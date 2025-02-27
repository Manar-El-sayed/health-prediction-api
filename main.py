#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
import joblib
from typing import Dict, Any, List
import uvicorn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define model paths - Update these paths to your model locations

DIABETES_MODEL_PATH = Path("models/diabetes.pkl")
DIABETES_SCALER_PATH = Path("models/scalerdiabetes.pkl")
ANEMIA_MODEL_PATH = Path("models/Ml_Anemia_graduation.pkl")
ANEMIA_SCALER_PATH = Path("models/ScalerAnemia.pkl")

# -------------------- Diabetes Models --------------------

class DiabetesInput(BaseModel):
    age: int = Field(..., description="Age of the patient")
    hypertension: int = Field(..., description="Hypertension (0 for No, 1 for Yes)")
    bmi: float = Field(..., description="Body Mass Index")
    hbA1c_level: float = Field(..., description="HbA1c Level")
    blood_glucose_level: float = Field(..., description="Blood Glucose Level")
    
    @model_validator(mode='after')
    def validate_fields(self) -> 'DiabetesInput':
        # Validate age
        if self.age < 0 or self.age > 120:
            raise ValueError('Age must be between 0 and 120 years')
        
        # Validate hypertension
        if self.hypertension not in [0, 1]:
            raise ValueError('Hypertension must be 0 (No) or 1 (Yes)')
        
        # Validate BMI
        if self.bmi <= 0 or self.bmi > 100:
            raise ValueError('BMI must be between 0 and 100')
        
        # Validate HbA1c
        if self.hbA1c_level <= 0 or self.hbA1c_level > 20:
            raise ValueError('HbA1c level must be between 0 and 20')
        
        # Validate blood glucose
        if self.blood_glucose_level <= 0:
            raise ValueError('Blood glucose level must be positive')
        
        return self
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 45,
                "hypertension": 0,
                "bmi": 28.5,
                "hbA1c_level": 6.5,
                "blood_glucose_level": 140
            }
        }
    }

class DiabetesPrediction(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    recommendation: str
    input_values: Dict[str, Any]

def get_diabetes_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

def get_diabetes_recommendation(risk_level: str, age: int, bmi: float) -> str:
    recommendations = {
        "Low Risk": (
            "Your diabetes risk appears to be low. To maintain this: \n"
            "- Continue maintaining a healthy lifestyle\n"
            "- Regular exercise for 30 minutes daily\n"
            "- Maintain a balanced diet"
        ),
        "Moderate Risk": (
            "You have moderate risk factors for diabetes: \n"
            "- Schedule a consultation with your healthcare provider\n"
            "- Consider lifestyle modifications\n"
            "- Monitor your blood sugar regularly"
        ),
        "High Risk": (
            "You show high risk factors for diabetes: \n"
            "- Immediate consultation with a healthcare provider is recommended\n"
            "- Comprehensive diabetes screening may be needed\n"
            "- Start monitoring blood sugar levels closely"
        )
    }
    
    base_recommendation = recommendations.get(risk_level, "Please consult a healthcare provider for proper evaluation.")
    
    # Add BMI-specific advice
    if bmi >= 30:
        base_recommendation += "\nConsider weight management strategies as your BMI indicates obesity."
    elif bmi >= 25:
        base_recommendation += "\nConsider weight management strategies as your BMI indicates overweight."
    
    return base_recommendation

# -------------------- Anemia Models --------------------

class AnemiaInput(BaseModel):
    gender: int = Field(..., description="Gender (0 for Female, 1 for Male)")
    hemoglobin: float = Field(..., description="Hemoglobin level (g/dL)")
    mch: float = Field(..., description="Mean Corpuscular Hemoglobin (pg)")
    mchc: float = Field(..., description="Mean Corpuscular Hemoglobin Concentration (g/dL)")
    mcv: float = Field(..., description="Mean Corpuscular Volume (fL)")
    
    @model_validator(mode='after')
    def validate_fields(self) -> 'AnemiaInput':
        if self.gender not in [0, 1]:
            raise ValueError('Gender must be 0 (Female) or 1 (Male)')
            
        for field_name, value in {
            'hemoglobin': self.hemoglobin,
            'mch': self.mch,
            'mchc': self.mchc,
            'mcv': self.mcv
        }.items():
            if value <= 0:
                raise ValueError(f'{field_name} must be positive')
        
        return self
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": 1,
                "hemoglobin": 11.5,
                "mch": 28.0,
                "mchc": 33.0,
                "mcv": 85.0
            }
        }
    }

class DetailedAnalysis(BaseModel):
    parameter: str
    value: float
    status: str
    interpretation: str
    recommendations: List[str]

class AnemiaPrediction(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    detailed_analysis: List[DetailedAnalysis]
    overall_recommendation: str
    input_values: Dict[str, Any]

class ParameterRanges:
    def __init__(self, gender: int):
        self.gender = gender
        self.ranges = {
            'hemoglobin': {
                'male': {'low': 13.5, 'high': 17.5},
                'female': {'low': 12.0, 'high': 15.5}
            },
            'mch': {'low': 27.0, 'high': 32.0},
            'mchc': {'low': 32.0, 'high': 36.0},
            'mcv': {'low': 80.0, 'high': 100.0}
        }

    def get_range(self, parameter: str) -> Dict[str, float]:
        if parameter == 'hemoglobin':
            return self.ranges[parameter]['male' if self.gender == 1 else 'female']
        return self.ranges[parameter]

def analyze_parameter(param_name: str, value: float, ranges: ParameterRanges) -> DetailedAnalysis:
    param_range = ranges.get_range(param_name)
    status = "Normal"
    interpretation = ""
    recommendations = []

    if value < param_range['low']:
        status = "Low"
        if param_name == 'hemoglobin':
            interpretation = "Low hemoglobin indicates reduced oxygen-carrying capacity in the blood."
            recommendations = [
                "Increase iron-rich foods in your diet (red meat, leafy greens, beans)",
                "Consider vitamin C supplements to enhance iron absorption",
                "Get tested for iron deficiency",
                "Discuss iron supplementation with your healthcare provider"
            ]
        elif param_name == 'mch':
            interpretation = "Low MCH suggests possible iron deficiency or chronic disease."
            recommendations = [
                "Include more iron-rich foods in your diet",
                "Consider B12 and folate supplementation",
                "Get tested for nutritional deficiencies"
            ]
        elif param_name == 'mchc':
            interpretation = "Low MCHC might indicate iron deficiency or thalassemia."
            recommendations = [
                "Seek further testing for iron deficiency",
                "Consider genetic testing for thalassemia",
                "Consult a hematologist"
            ]
        elif param_name == 'mcv':
            interpretation = "Low MCV indicates microcytic anemia, commonly due to iron deficiency."
            recommendations = [
                "Increase iron intake through diet or supplements",
                "Get tested for iron deficiency and lead exposure",
                "Consider testing for thalassemia"
            ]
    elif value > param_range['high']:
        status = "High"
        if param_name == 'hemoglobin':
            interpretation = "Elevated hemoglobin might indicate polycythemia or dehydration."
            recommendations = [
                "Increase water intake",
                "Avoid smoking and altitude exposure",
                "Get tested for polycythemia vera",
                "Consider sleep study for sleep apnea"
            ]
        elif param_name == 'mch':
            interpretation = "High MCH often indicates macrocytic anemia."
            recommendations = [
                "Get tested for vitamin B12 and folate levels",
                "Consider liver function tests",
                "Discuss alcohol consumption with healthcare provider"
            ]
        elif param_name == 'mchc':
            interpretation = "Elevated MCHC might indicate spherocytosis or lab error."
            recommendations = [
                "Verify results with repeat testing",
                "Consider hereditary spherocytosis testing",
                "Consult a hematologist"
            ]
        elif param_name == 'mcv':
            interpretation = "High MCV suggests macrocytic anemia, often due to B12 or folate deficiency."
            recommendations = [
                "Get tested for vitamin B12 and folate deficiency",
                "Evaluate alcohol consumption",
                "Consider thyroid function tests"
            ]
    else:
        interpretation = f"{param_name.upper()} is within normal range."
        recommendations = ["Continue current dietary and lifestyle habits"]

    return DetailedAnalysis(
        parameter=param_name,
        value=value,
        status=status,
        interpretation=interpretation,
        recommendations=recommendations
    )

def get_anemia_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

def generate_overall_recommendation(detailed_analyses: List[DetailedAnalysis], risk_level: str) -> str:
    abnormal_params = [analysis for analysis in detailed_analyses if analysis.status != "Normal"]
    
    if not abnormal_params:
        return ("Your blood parameters are within normal ranges. Continue maintaining a healthy diet "
                "and lifestyle. Regular check-ups are recommended for monitoring.")
    
    recommendation_parts = []
    
    if risk_level == "High Risk":
        recommendation_parts.append("URGENT: Schedule an appointment with a healthcare provider for comprehensive evaluation.")
    
    if any(analysis.parameter == "hemoglobin" and analysis.status == "Low" for analysis in abnormal_params):
        recommendation_parts.append(
            "Priority: Address low hemoglobin levels through dietary changes and possible supplementation. "
            "Include iron-rich foods and vitamin C to enhance absorption."
        )
    
    nutritional_issues = any(analysis.status == "Low" for analysis in abnormal_params)
    if nutritional_issues:
        recommendation_parts.append(
            "Consider comprehensive nutritional assessment and supplementation based on deficiencies. "
            "Focus on a balanced diet rich in iron, B12, and folate."
        )
    
    lifestyle_changes = [
        "Maintain regular exercise within your comfort level",
        "Ensure adequate sleep and stress management",
        "Stay well-hydrated",
        "Avoid smoking and limit alcohol consumption"
    ]
    recommendation_parts.append("Lifestyle Recommendations: " + " ".join(lifestyle_changes))
    
    monitoring = "Schedule regular follow-up blood tests to monitor your progress."
    recommendation_parts.append(monitoring)
    
    return " ".join(recommendation_parts)

# -------------------- Model Manager --------------------

class ModelManager:
    def __init__(self):
        self.diabetes_model = None
        self.diabetes_scaler = None
        self.anemia_model = None
        self.anemia_scaler = None
        self.load_models()
    
    def load_models(self):
        try:
            # Load diabetes models
            if not DIABETES_MODEL_PATH.exists():
                raise FileNotFoundError(f"Diabetes model file not found at {DIABETES_MODEL_PATH}")
            if not DIABETES_SCALER_PATH.exists():
                raise FileNotFoundError(f"Diabetes scaler file not found at {DIABETES_SCALER_PATH}")
            
            self.diabetes_model = joblib.load(DIABETES_MODEL_PATH)
            self.diabetes_scaler = joblib.load(DIABETES_SCALER_PATH)
            logger.info("Diabetes models loaded successfully")
            
            # Load anemia models
            if not ANEMIA_MODEL_PATH.exists():
                raise FileNotFoundError(f"Anemia model file not found at {ANEMIA_MODEL_PATH}")
            if not ANEMIA_SCALER_PATH.exists():
                raise FileNotFoundError(f"Anemia scaler file not found at {ANEMIA_SCALER_PATH}")
            
            self.anemia_model = joblib.load(ANEMIA_MODEL_PATH)
            self.anemia_scaler = joblib.load(ANEMIA_SCALER_PATH)
            logger.info("Anemia models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

# -------------------- FastAPI Application --------------------

app = FastAPI(
    title="Health Prediction API",
    description="API for predicting diabetes and anemia risk using trained machine learning models",
    version="1.0.0"
)

# Initialize model manager
model_manager = None

@app.on_event("startup")
async def startup_event():
    global model_manager
    model_manager = ModelManager()
    logger.info("Model manager initialized during startup")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Health Prediction API",
        "diabetes_model_status": "loaded" if model_manager and model_manager.diabetes_model is not None else "not loaded",
        "anemia_model_status": "loaded" if model_manager and model_manager.anemia_model is not None else "not loaded",
        "documentation": "/docs",
        "endpoints": {
            "/predict/diabetes": "POST - Make diabetes predictions",
            "/predict/anemia": "POST - Make anemia predictions",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health_check():
    if not model_manager or model_manager.diabetes_model is None or model_manager.diabetes_scaler is None:
        raise HTTPException(status_code=503, detail="Diabetes model or scaler not loaded")
    
    if not model_manager or model_manager.anemia_model is None or model_manager.anemia_scaler is None:
        raise HTTPException(status_code=503, detail="Anemia model or scaler not loaded")
    
    diabetes_features = []
    if hasattr(model_manager.diabetes_scaler, 'feature_names_in_'):
        diabetes_features = model_manager.diabetes_scaler.feature_names_in_.tolist()
    
    anemia_features = []
    if hasattr(model_manager.anemia_scaler, 'feature_names_in_'):
        anemia_features = model_manager.anemia_scaler.feature_names_in_.tolist()
    
    return {
        "status": "healthy",
        "diabetes_model": {
            "loaded": True,
            "model_path": str(DIABETES_MODEL_PATH),
            "scaler_path": str(DIABETES_SCALER_PATH),
            "features": diabetes_features
        },
        "anemia_model": {
            "loaded": True,
            "model_path": str(ANEMIA_MODEL_PATH),
            "scaler_path": str(ANEMIA_SCALER_PATH),
            "features": anemia_features
        }
    }

@app.post("/predict/diabetes", response_model=DiabetesPrediction)
async def predict_diabetes(data: DiabetesInput):
    logger.info("Received diabetes prediction request")
    
    if not model_manager or model_manager.diabetes_model is None or model_manager.diabetes_scaler is None:
        raise HTTPException(status_code=503, detail="Diabetes model or scaler not loaded")
    
    try:
        features = np.array([[
            data.age,
            data.hypertension,
            data.bmi,
            data.hbA1c_level,
            data.blood_glucose_level
        ]])
        
        logger.debug(f"Original input data: {features}")
        features_scaled = model_manager.diabetes_scaler.transform(features)
        logger.debug(f"Scaled input data: {features_scaled}")
        
        prediction = model_manager.diabetes_model.predict(features_scaled)[0]
        
        if hasattr(model_manager.diabetes_model, "predict_proba"):
            probability = model_manager.diabetes_model.predict_proba(features_scaled)[0][1]
        else:
            decision_value = model_manager.diabetes_model.decision_function(features_scaled)
            probability = 1 / (1 + np.exp(-decision_value))
            probability = float(probability[0])
        
        risk_level = get_diabetes_risk_level(probability)
        recommendation = get_diabetes_recommendation(risk_level, data.age, data.bmi)
        
        response = DiabetesPrediction(
            prediction=int(prediction),
            probability=probability,
            risk_level=risk_level,
            recommendation=recommendation,
            input_values=data.model_dump()
        )
        
        logger.info(f"Diabetes prediction complete: {prediction}, Risk Level: {risk_level}")
        return response
        
    except Exception as e:
        logger.error(f"Diabetes prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/anemia", response_model=AnemiaPrediction)
async def predict_anemia(data: AnemiaInput):
    logger.info("Received anemia prediction request")
    
    if not model_manager or model_manager.anemia_model is None or model_manager.anemia_scaler is None:
        raise HTTPException(status_code=503, detail="Anemia model or scaler not loaded")
    
    try:
        features = np.array([[
            data.gender,
            data.hemoglobin,
            data.mch,
            data.mchc,
            data.mcv
        ]])
        
        features_scaled = model_manager.anemia_scaler.transform(features)
        prediction = model_manager.anemia_model.predict(features_scaled)[0]
        
        if hasattr(model_manager.anemia_model, "predict_proba"):
            probability = model_manager.anemia_model.predict_proba(features_scaled)[0][1]
        else:
            decision_value = model_manager.anemia_model.decision_function(features_scaled)
            probability = 1 / (1 + np.exp(-decision_value))
            probability = float(probability[0])
        
        risk_level = get_anemia_risk_level(probability)
        
        # Perform detailed analysis of each parameter
        ranges = ParameterRanges(data.gender)
        detailed_analyses = [
            analyze_parameter('hemoglobin', data.hemoglobin, ranges),
            analyze_parameter('mch', data.mch, ranges),
            analyze_parameter('mchc', data.mchc, ranges),
            analyze_parameter('mcv', data.mcv, ranges)
        ]
        
        overall_recommendation = generate_overall_recommendation(detailed_analyses, risk_level)
        
        response = AnemiaPrediction(
            prediction=int(prediction),
            probability=probability,
            risk_level=risk_level,
            detailed_analysis=detailed_analyses,
            overall_recommendation=overall_recommendation,
            input_values=data.model_dump()
        )
        
        logger.info(f"Anemia prediction complete: {prediction}, Risk Level: {risk_level}")
        return response
        
    except Exception as e:
        logger.error(f"Anemia prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


# In[ ]:




