from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from database import SessionLocal, PredictionHistory
from PIL import Image
import numpy as np
import shutil
import os
import io

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load Models
mnist_model = load_model("models/mnist/mnist_cnn_model.h5")
brain_tumor_model = load_model("models/brain_tumor/brain_tumor_model.h5")
cifar10_model = load_model("models/cifar10/best_model.h5")

# Class Labels
brain_tumor_classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


@app.get("/", response_class=HTMLResponse)
@app.get("/image_detector", response_class=HTMLResponse)
async def image_detector_home(request: Request):
    db = SessionLocal()
    try:
        history = db.query(PredictionHistory).filter(
            PredictionHistory.disease.in_(["mnist_digit", "brain_tumor", "cifar10"])
        ).order_by(PredictionHistory.created_at.desc()).limit(5).all()

        return templates.TemplateResponse("image_detector.html", {
            "request": request,
            "username": "Guest",
            "history": history
        })
    finally:
        db.close()


@app.get("/mnist", response_class=HTMLResponse)
async def mnist_form(request: Request):
    return templates.TemplateResponse("mnist.html", {"request": request, "result": None})


@app.post("/mnist_predict", response_class=HTMLResponse)
async def mnist_predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('L').resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = mnist_model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    result_text = f"Predicted digit: {predicted_class} with {confidence:.2f}% confidence."

    db = SessionLocal()
    new_history = PredictionHistory(
        user_id=0,
        disease="mnist_digit",
        result=result_text
    )
    db.add(new_history)
    db.commit()
    db.close()

    return templates.TemplateResponse("mnist.html", {"request": request, "result": result_text})


@app.get("/brain_tumor", response_class=HTMLResponse)
async def brain_tumor_form(request: Request):
    return templates.TemplateResponse("brain_tumor.html", {"request": request, "result": None})


@app.post("/brain_tumor_predict", response_class=HTMLResponse)
async def brain_tumor_predict(request: Request, file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        img = image.load_img(file_location, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = brain_tumor_model.predict(img_array)
        predicted_class = brain_tumor_classes[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        result_text = f"{predicted_class.capitalize()} detected with {confidence:.2f}% confidence."

    except Exception as e:
        result_text = f"Error during prediction: {str(e)}"

    finally:
        os.remove(file_location)

    db = SessionLocal()
    new_history = PredictionHistory(
        user_id=0,
        disease="brain_tumor",
        result=result_text
    )
    db.add(new_history)
    db.commit()
    db.close()

    return templates.TemplateResponse("brain_tumor.html", {"request": request, "result": result_text})


@app.get("/cifar10", response_class=HTMLResponse)
async def cifar10_form(request: Request):
    return templates.TemplateResponse("cifar10.html", {"request": request, "result": None})


@app.post("/cifar10_predict", response_class=HTMLResponse)
async def cifar10_predict(request: Request, file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        img = image.load_img(file_location, target_size=(32, 32))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = cifar10_model.predict(img_array)
        class_index = np.argmax(preds)
        confidence = float(np.max(preds)) * 100
        predicted_class = cifar10_classes[class_index]

        result_text = f"{predicted_class.capitalize()} detected with {confidence:.2f}% confidence."

    except Exception as e:
        result_text = f"Error during prediction: {str(e)}"

    finally:
        os.remove(file_location)

    db = SessionLocal()
    new_history = PredictionHistory(
        user_id=0,
        disease="cifar10",
        result=result_text
    )
    db.add(new_history)
    db.commit()
    db.close()

    return templates.TemplateResponse("cifar10.html", {"request": request, "result": result_text})