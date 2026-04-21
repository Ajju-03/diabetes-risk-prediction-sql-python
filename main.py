from fastapi import FastAPI
from app.risk_level import get_risk_level
import joblib
from pydantic import BaseModel
import numpy as np
import os
import pandas as pd

from config import CONFIG
from core.datapreprocessor import DataPreprocessor
from core.mlflowtracker import MLflowTracker
from Database.databasemanager import DatabaseManager

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "src", "rf_model.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, CONFIG.get("preprocessor_path", "preprocessor.pkl"))

model = None
preprocessor = None
tracker = None
db = None
use_production_model = False

@app.on_event("startup")
def load_model():
    global model, preprocessor, tracker, db, use_production_model
    
    db = DatabaseManager()

    try:
        preprocessor = DataPreprocessor.load(PREPROCESSOR_PATH)
    except Exception as e:
        print(f"WARNING: Failed to load preprocessor at {PREPROCESSOR_PATH}: {e}")
        preprocessor = None

    try:
        tracker = MLflowTracker(CONFIG["experiment_name"], CONFIG["tracking_uri"])
        model = tracker.load_production_model()
        use_production_model = True
        print("Loaded best model from MLflow production registry.")
    except Exception as e:
        print(f"WARNING: Failed to load production model from MLflow: {e}")
        use_production_model = False
        try:
            model = joblib.load(MODEL_PATH)
            print(f"Loaded fallback local model from {MODEL_PATH}.")
        except Exception as err:
            print(f"CRITICAL: Failed to load fallback model at {MODEL_PATH}. Error: {err}")

class PatientData(BaseModel):
    patient_id: str
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
    patient_data = {k.lower(): v for k, v in patient_dict.items()}

    if preprocessor is not None:
        df = preprocessor.clean_data(pd.DataFrame([patient_data]))
        df = preprocessor.feature_engineering(df)
        for col in preprocessor.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[preprocessor.feature_names]
        df_imputed = pd.DataFrame(
            preprocessor.imputer.transform(df),
            columns=preprocessor.feature_names
        )
        features = preprocessor.scaler.transform(df_imputed)
    else:
        features = np.array([list(patient_data.values())])

    if model is None:
        return {"error": "Model not loaded on server"}

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    risk_info = get_risk_level(patient_dict, probability)

    # Save to SQL database
    try:
        db.save_predictions(
            patient_id = data.patient_id,
            model_name = "FastAPI-Production",
            model_version = "latest",
            risk_score = float(probability),
            prediction = int(prediction)
        )
    except Exception as e:
        print(f"ERROR: Failed to save prediction to DB: {e}")

    return {
        "patient_id": data.patient_id,
        "prediction": "Diabetic" if prediction == 1 else "Non-Diabetic",
        "probability": round(float(probability), 4),
        "risk_level": risk_info["risk_level"],
        "risk_analysis": risk_info,
    }

