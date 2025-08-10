# app.py - FastAPI No-Show Prediction Service

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Appointment No-Show Prediction API",
    description="Predict the likelihood of patient no-shows for medical appointments",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scaler
model = None
scaler = None
feature_names = None

# Pydantic models for request/response
class AppointmentData(BaseModel):
    """Input data for appointment no-show prediction"""
    
    # Patient demographics
    gender: str = Field(..., description="Patient gender: 'M' or 'F'")
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    
    # Appointment details
    scheduled_day: str = Field(..., description="When appointment was scheduled (YYYY-MM-DD HH:MM:SS)")
    appointment_day: str = Field(..., description="Appointment date and time (YYYY-MM-DD HH:MM:SS)")
    
    # Patient conditions and benefits
    scholarship: int = Field(0, ge=0, le=1, description="Has scholarship (0=No, 1=Yes)")
    hipertension: int = Field(0, ge=0, le=1, description="Has hypertension (0=No, 1=Yes)")
    diabetes: int = Field(0, ge=0, le=1, description="Has diabetes (0=No, 1=Yes)")
    alcoholism: int = Field(0, ge=0, le=1, description="Has alcoholism (0=No, 1=Yes)")
    handcap: int = Field(0, ge=0, le=4, description="Handicap level (0-4)")
    sms_received: int = Field(0, ge=0, le=1, description="Received SMS reminder (0=No, 1=Yes)")
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "F",
                "age": 35,
                "scheduled_day": "2024-01-15 09:30:00",
                "appointment_day": "2024-01-22 14:00:00",
                "scholarship": 1,
                "hipertension": 0,
                "diabetes": 0,
                "alcoholism": 0,
                "handcap": 0,
                "sms_received": 1
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    
    no_show_probability: float = Field(..., description="Probability of no-show (0-1)")
    prediction: str = Field(..., description="Predicted outcome: 'Show' or 'No-Show'")
    confidence: str = Field(..., description="Confidence level: 'Low', 'Medium', 'High'")
    risk_factors: Dict[str, Any] = Field(..., description="Key risk factors contributing to prediction")
    recommendation: str = Field(..., description="Recommended action based on prediction")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str

# Utility functions
def load_models():
    """Load the trained model and scaler"""
    global model, scaler, feature_names
    
    try:
        model_path = Path("improved_model.pkl")
        scaler_path = Path("scaler.pkl")
        
        if not model_path.exists():
            raise FileNotFoundError("Model file 'improved_model.pkl' not found")
        if not scaler_path.exists():
            raise FileNotFoundError("Scaler file 'scaler.pkl' not found")
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Define feature names (must match training data)
        feature_names = [
            'Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes', 
            'Alcoholism', 'Handcap', 'SMS_received', 'DaysAhead', 
            'ScheduledDayOfWeek', 'AppointmentDayOfWeek', 'ScheduledHour', 
            'AgeGroup', 'HealthConditionsCount'
        ]
        
        logger.info("Model and scaler loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def preprocess_appointment_data(data: AppointmentData) -> np.ndarray:
    """Preprocess appointment data to match training format"""
    
    try:
        # Parse datetime strings
        scheduled_dt = pd.to_datetime(data.scheduled_day)
        appointment_dt = pd.to_datetime(data.appointment_day)
        
        # Calculate derived features
        days_ahead = (appointment_dt - scheduled_dt).days
        scheduled_day_of_week = scheduled_dt.dayofweek
        appointment_day_of_week = appointment_dt.dayofweek
        scheduled_hour = scheduled_dt.hour
        
        # Create age group (0=Child, 1=Adult, 2=Senior)
        if data.age < 18:
            age_group = 0
        elif data.age < 65:
            age_group = 1
        else:
            age_group = 2
            
        # Count health conditions
        health_conditions_count = (
            data.hipertension + data.diabetes + 
            data.alcoholism + (1 if data.handcap > 0 else 0)
        )
        
        # Convert gender to numeric (F=0, M=1)
        gender_numeric = 1 if data.gender.upper() == 'M' else 0
        
        # Create feature array in the correct order
        features = np.array([
            gender_numeric,           # Gender
            data.age,                 # Age
            data.scholarship,         # Scholarship
            data.hipertension,        # Hipertension
            data.diabetes,            # Diabetes
            data.alcoholism,          # Alcoholism
            data.handcap,             # Handcap
            data.sms_received,        # SMS_received
            days_ahead,               # DaysAhead
            scheduled_day_of_week,    # ScheduledDayOfWeek
            appointment_day_of_week,  # AppointmentDayOfWeek
            scheduled_hour,           # ScheduledHour
            age_group,                # AgeGroup
            health_conditions_count   # HealthConditionsCount
        ]).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        return features_scaled, {
            'days_ahead': days_ahead,
            'scheduled_hour': scheduled_hour,
            'age_group': ['Child', 'Adult', 'Senior'][age_group],
            'health_conditions_count': health_conditions_count
        }
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Data preprocessing error: {str(e)}")

def analyze_risk_factors(data: AppointmentData, derived_features: Dict) -> Dict[str, Any]:
    """Analyze key risk factors"""
    
    risk_factors = {}
    
    # Days ahead analysis
    days_ahead = derived_features['days_ahead']
    if days_ahead <= 0:
        risk_factors['scheduling'] = "Same-day appointment - HIGH RISK"
    elif days_ahead <= 7:
        risk_factors['scheduling'] = "Short notice appointment - MEDIUM RISK"
    elif days_ahead > 30:
        risk_factors['scheduling'] = "Far future appointment - MEDIUM RISK"
    else:
        risk_factors['scheduling'] = "Normal scheduling window - LOW RISK"
    
    # Age analysis
    if data.age < 18:
        risk_factors['age'] = "Young patient - may need parent coordination"
    elif data.age > 65:
        risk_factors['age'] = "Senior patient - may have mobility/health issues"
    else:
        risk_factors['age'] = "Adult patient - standard risk"
    
    # Health conditions
    if derived_features['health_conditions_count'] >= 2:
        risk_factors['health'] = "Multiple health conditions - higher complexity"
    elif derived_features['health_conditions_count'] == 1:
        risk_factors['health'] = "Single health condition - moderate complexity"
    else:
        risk_factors['health'] = "No major health conditions reported"
    
    # SMS reminder
    if data.sms_received == 0:
        risk_factors['communication'] = "No SMS reminder sent - consider sending"
    else:
        risk_factors['communication'] = "SMS reminder sent - good"
    
    # Appointment timing
    scheduled_hour = derived_features['scheduled_hour']
    if scheduled_hour < 8 or scheduled_hour > 17:
        risk_factors['timing'] = "Outside normal hours - may affect attendance"
    else:
        risk_factors['timing'] = "Normal business hours - standard risk"
    
    return risk_factors

def get_recommendation(probability: float, risk_factors: Dict) -> str:
    """Generate recommendation based on prediction"""
    
    if probability >= 0.7:
        return "HIGH RISK: Consider calling patient to confirm. Send additional reminders."
    elif probability >= 0.5:
        return "MEDIUM RISK: Send SMS reminder 24h before appointment."
    elif probability >= 0.3:
        return "LOW-MEDIUM RISK: Standard SMS reminder should suffice."
    else:
        return "LOW RISK: Patient likely to show up. Standard procedures."

# API Routes
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    success = load_models()
    if not success:
        logger.error("Failed to load models on startup")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "Medical Appointment No-Show Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None and scaler is not None else "unhealthy",
        model_loaded=model is not None and scaler is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_no_show(appointment: AppointmentData):
    """Predict no-show probability for an appointment"""
    
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Preprocess the data
        features_scaled, derived_features = preprocess_appointment_data(appointment)
        
        # Make prediction
        probability = model.predict_proba(features_scaled)[0][1]  # Probability of no-show
        prediction = "No-Show" if probability >= 0.5 else "Show"
        
        # Determine confidence level
        if probability >= 0.8 or probability <= 0.2:
            confidence = "High"
        elif probability >= 0.6 or probability <= 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Analyze risk factors
        risk_factors = analyze_risk_factors(appointment, derived_features)
        
        # Get recommendation
        recommendation = get_recommendation(probability, risk_factors)
        
        logger.info(f"Prediction made: {prediction} ({probability:.3f})")
        
        return PredictionResponse(
            no_show_probability=round(probability, 3),
            prediction=prediction,
            confidence=confidence,
            risk_factors=risk_factors,
            recommendation=recommendation
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(appointments: list[AppointmentData]):
    """Predict no-show probability for multiple appointments"""
    
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    if len(appointments) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 100 appointments per request."
        )
    
    try:
        results = []
        for appointment in appointments:
            # Reuse the single prediction logic
            features_scaled, derived_features = preprocess_appointment_data(appointment)
            probability = model.predict_proba(features_scaled)[0][1]
            prediction = "No-Show" if probability >= 0.5 else "Show"
            
            if probability >= 0.8 or probability <= 0.2:
                confidence = "High"
            elif probability >= 0.6 or probability <= 0.4:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            risk_factors = analyze_risk_factors(appointment, derived_features)
            recommendation = get_recommendation(probability, risk_factors)
            
            results.append({
                "no_show_probability": round(probability, 3),
                "prediction": prediction,
                "confidence": confidence,
                "risk_factors": risk_factors,
                "recommendation": recommendation
            })
        
        logger.info(f"Batch prediction made for {len(appointments)} appointments")
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# For running with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)