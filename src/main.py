from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Diabetes Prediction")

model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")


class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int


@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}


@app.post("/predict")
def predict(data: DiabetesInput):

    input_data = np.array([[
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.diabetes_pedigree_function,
        data.age
    ]])

    scaled_data = scaler.transform(input_data)

    prob = model.predict_proba(scaled_data)[0][1]
    prediction = 1 if prob >= 0.48 else 0

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return {
        "prediction": result,
        "probability": float(prob)
    }