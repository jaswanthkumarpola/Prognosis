import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel, Field

# Initialize FastAPI app instance
app = FastAPI(
    title="Diabetes Prediction API",
    description="REST API for predicting diabetes risk using a Random Forest model with StandardScaler.",
    version="1.0.0",
)

# Enable CORS for local testing and live deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Load Model and Scaler
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# Custom optimal threshold from model training
BEST_THRESHOLD = 0.2424

# Input Validation Schema
class PatientData(BaseModel):
    pregnancies: int = Field(..., ge=0, le=20)
    glucose: float = Field(..., ge=0.0, le=300.0)
    blood_pressure: float = Field(..., ge=0.0, le=200.0)
    skin_thickness: float = Field(..., ge=0.0, le=100.0)
    insulin: float = Field(..., ge=0.0, le=900.0)
    bmi: float = Field(..., ge=0.0, le=70.0)
    diabetes_pedigree: float = Field(..., ge=0.0, le=3.0)
    age: int = Field(..., ge=1, le=120)


@app.get("/")
def home():
    return {"message": "Diabetes Risk API is running."}


@app.post("/predict")
def predict_diabetes(patient: PatientData):
    if model is None or scaler is None:
        raise HTTPException(
            status_code=500,
            detail="Model or Scaler file missing/failed to load on server.",
        )

    try:
        # 1. Feature Engineering (BMI * Age & Glucose * BMI)
        bmi_age = float(patient.bmi * patient.age)
        glucose_bmi = float(patient.glucose * patient.bmi)

        # 2. Raw Input Vector (10 features matching model order)
        raw_features = np.array(
            [
                [
                    patient.pregnancies,
                    patient.glucose,
                    patient.blood_pressure,
                    patient.skin_thickness,
                    patient.insulin,
                    patient.bmi,
                    patient.diabetes_pedigree,
                    patient.age,
                    bmi_age,
                    glucose_bmi,
                ]
            ],
            dtype=np.float64,
        )

        # 3. Apply StandardScaler
        scaled_features = scaler.transform(raw_features)

        # 4. Predict Risk Probability & Apply Custom Threshold
        proba = float(model.predict_proba(scaled_features)[0][1])
        is_diabetic = int(proba >= BEST_THRESHOLD)

        return {
            "risk_probability": round(proba, 4),
            "prediction": is_diabetic,
            "label": "Diabetic" if is_diabetic == 1 else "Non-Diabetic"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")