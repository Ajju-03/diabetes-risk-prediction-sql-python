from fastapi import FastAPI
from app.risk_level import get_risk_level
import joblib
from pydantic import BaseModel
import numpy as np
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "src", "rf_model.joblib")

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"CRITICAL: Failed to load model at {model_path}. Error: {e}")

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/")
def read_root():
    return {"message": "Diabetes risk prediction API active"}

@app.post("/predict")
def predict_data(data: PatientData):

    patient_dict = data.model_dump()
    
    features = np.array([list(patient_dict.values())])

    if model is None:
        return {"error": "Model not loaded on server"}

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    risk_info = get_risk_level(patient_dict, probability) 
    
    return {
        "prediction": "Diabetic" if prediction == 1 else "Non-Diabetic",
        "probability": round(float(probability), 2),
        "Model Confidence": (f"{probability*100:.0f}%"),
        "risk_analysis": risk_info,
    }

