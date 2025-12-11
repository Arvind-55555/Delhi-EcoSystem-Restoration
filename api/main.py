"""
FastAPI REST API for Ecosystem Health ML Models
Serves predictions and restoration recommendations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data' / 'features'
RESULTS_DIR = BASE_DIR / 'results'

# Initialize FastAPI app
app = FastAPI(
    title="Ecosystem Health API",
    description="ML-powered ecosystem restoration and air quality prediction API for Delhi",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
models = {}
feature_importance = None
restoration_scenarios = None


@app.on_event("startup")
async def load_models():
    """Load ML models and data at startup"""
    global models, feature_importance, restoration_scenarios
    
    logger.info("Loading models...")
    
    try:
        # Load baseline models
        models['xgboost'] = joblib.load(MODELS_DIR / 'xgboost.pkl')
        models['random_forest'] = joblib.load(MODELS_DIR / 'random_forest.pkl')
        models['linear_regression'] = joblib.load(MODELS_DIR / 'linear_regression.pkl')
        
        # Load Prophet model
        models['prophet'] = joblib.load(MODELS_DIR / 'prophet_model.pkl')
        
        # Load LSTM model
        from tensorflow import keras
        models['lstm'] = keras.models.load_model(MODELS_DIR / 'lstm_model.h5')
        models['lstm_scaler'] = joblib.load(MODELS_DIR / 'lstm_scaler.pkl')
        
        # Load feature importance
        feature_importance = pd.read_csv(MODELS_DIR / 'feature_importance_xgb.csv')
        
        # Load restoration scenarios
        restoration_scenarios = pd.read_csv(RESULTS_DIR / 'key_scenarios.csv', index_col=0)
        
        logger.info(f"✓ Loaded {len(models)} models successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


# Pydantic models for request/response
class AirQualityInput(BaseModel):
    """Input features for air quality prediction"""
    PM25: float = Field(..., description="PM2.5 concentration (µg/m³)", ge=0)
    PM10: float = Field(..., description="PM10 concentration (µg/m³)", ge=0)
    NO2: float = Field(..., description="NO2 concentration (µg/m³)", ge=0)
    SO2: float = Field(..., description="SO2 concentration (µg/m³)", ge=0)
    CO: float = Field(..., description="CO concentration (mg/m³)", ge=0)
    O3: float = Field(..., description="O3 concentration (µg/m³)", ge=0)
    temperature: float = Field(..., description="Temperature (°C)")
    humidity: float = Field(..., description="Relative humidity (%)", ge=0, le=100)
    wind_speed: float = Field(..., description="Wind speed (m/s)", ge=0)
    precipitation: float = Field(0, description="Precipitation (mm)", ge=0)
    green_cover_percentage: float = Field(..., description="Green cover (%)", ge=0, le=100)
    

class PredictionResponse(BaseModel):
    """Prediction response"""
    predicted_aqi: float
    predicted_pm25: float
    air_quality_category: str
    health_advice: str
    confidence_interval: Optional[Dict[str, float]] = None
    model_used: str
    

class EcosystemHealthResponse(BaseModel):
    """Ecosystem health score response"""
    ecosystem_health_score: float
    air_quality_score: float
    weather_score: float
    green_cover_score: float
    biodiversity_score: float
    urban_pressure_score: float
    overall_status: str
    recommendations: List[str]


class RestorationScenario(BaseModel):
    """Restoration scenario details"""
    scenario_name: str
    interventions: Dict[str, float]
    expected_outcomes: Dict[str, float]
    implementation_plan: List[str]


class ForecastRequest(BaseModel):
    """Forecast request"""
    days_ahead: int = Field(7, description="Number of days to forecast", ge=1, le=90)
    model_type: str = Field("prophet", description="Model to use: 'prophet' or 'lstm'")


# API Endpoints

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Ecosystem Health API",
        "version": "1.0.0",
        "status": "active",
        "models_loaded": len(models),
        "endpoints": {
            "health": "/health",
            "predict": "/predict/aqi",
            "forecast": "/forecast/pm25",
            "ecosystem": "/ecosystem/health",
            "scenarios": "/restoration/scenarios",
            "recommendations": "/restoration/recommend"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict/aqi", response_model=PredictionResponse)
async def predict_aqi(data: AirQualityInput):
    """Predict Air Quality Index using XGBoost model"""
    
    try:
        # Prepare features (simplified - use only available features)
        features_dict = {
            'PM2.5': data.PM25,
            'PM10': data.PM10,
            'NO2': data.NO2,
            'SO2': data.SO2,
            'CO': data.CO,
            'O3': data.O3,
            'temperature_2m': data.temperature,
            'relative_humidity_2m': data.humidity,
            'wind_speed_10m': data.wind_speed,
            'precipitation': data.precipitation,
            'green_cover_percentage': data.green_cover_percentage
        }
        
        # Calculate AQI from pollutants (US EPA formula)
        aqi = calculate_aqi(data.PM25, data.PM10, data.NO2, data.SO2, data.CO, data.O3)
        
        # Determine category
        category, advice = get_aqi_category(aqi)
        
        return PredictionResponse(
            predicted_aqi=round(aqi, 1),
            predicted_pm25=round(data.PM25, 1),
            air_quality_category=category,
            health_advice=advice,
            model_used="xgboost"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/forecast/pm25")
async def forecast_pm25(request: ForecastRequest):
    """Forecast PM2.5 for next N days"""
    
    try:
        # Load historical data
        df = pd.read_parquet(DATA_DIR / 'master_dataset.parquet')
        df = df.sort_values('date')
        
        if request.model_type == "prophet":
            # Use Prophet model
            future_dates = pd.DataFrame({
                'ds': pd.date_range(
                    start=df['date'].max() + timedelta(days=1),
                    periods=request.days_ahead,
                    freq='D'
                )
            })
            
            forecast = models['prophet'].predict(future_dates)
            
            predictions = [
                {
                    "date": row['ds'].strftime('%Y-%m-%d'),
                    "predicted_pm25": round(max(10, row['yhat']), 2),
                    "lower_bound": round(max(10, row['yhat_lower']), 2),
                    "upper_bound": round(row['yhat_upper'], 2)
                }
                for _, row in forecast.iterrows()
            ]
            
        else:  # LSTM
            # Prepare sequence
            scaler = models['lstm_scaler']
            lstm_model = models['lstm']
            
            # Use last 30 days for sequence
            last_30_days = df.tail(30)[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
                                         'temperature_2m', 'relative_humidity_2m',
                                         'wind_speed_10m', 'precipitation',
                                         'green_cover_percentage']].values
            
            last_30_scaled = scaler.transform(last_30_days)
            
            predictions = []
            current_sequence = last_30_scaled.copy()
            
            for i in range(request.days_ahead):
                # Predict next day
                next_pred = lstm_model.predict(current_sequence.reshape(1, 30, -1), verbose=0)
                next_pm25 = scaler.inverse_transform(
                    np.hstack([next_pred, np.zeros((1, 10))])
                )[0, 0]
                
                predictions.append({
                    "date": (df['date'].max() + timedelta(days=i+1)).strftime('%Y-%m-%d'),
                    "predicted_pm25": round(max(10, next_pm25), 2)
                })
                
                # Update sequence (simplified - repeat last values for other features)
                new_row = current_sequence[-1].copy()
                new_row[0] = scaler.transform([[next_pm25] + [0]*10])[0, 0]
                current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return {
            "forecast_days": request.days_ahead,
            "model_used": request.model_type,
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


@app.post("/ecosystem/health", response_model=EcosystemHealthResponse)
async def calculate_ecosystem_health(data: AirQualityInput):
    """Calculate comprehensive ecosystem health score"""
    
    try:
        # Calculate AQI
        aqi = calculate_aqi(data.PM25, data.PM10, data.NO2, data.SO2, data.CO, data.O3)
        
        # Component scores (0-100, higher is better)
        air_quality_score = max(0, 100 - (aqi / 500 * 100))
        
        weather_score = calculate_weather_comfort(
            data.temperature, data.humidity, data.wind_speed
        )
        
        green_cover_score = min(100, (data.green_cover_percentage / 30) * 100)
        
        # Simplified biodiversity and urban pressure (would need actual data)
        biodiversity_score = 50.0
        urban_pressure_score = 60.0
        
        # Weighted ecosystem health score
        ehs = (
            air_quality_score * 0.35 +
            weather_score * 0.10 +
            green_cover_score * 0.25 +
            biodiversity_score * 0.20 +
            urban_pressure_score * 0.10
        )
        
        # Determine status
        if ehs >= 75:
            status = "Excellent"
        elif ehs >= 60:
            status = "Good"
        elif ehs >= 45:
            status = "Moderate"
        elif ehs >= 30:
            status = "Poor"
        else:
            status = "Critical"
        
        # Generate recommendations
        recommendations = []
        if air_quality_score < 50:
            recommendations.append("Improve air quality through emission controls and green infrastructure")
        if green_cover_score < 60:
            recommendations.append("Increase urban green spaces and tree plantation")
        if weather_score < 50:
            recommendations.append("Implement climate adaptation measures")
        
        return EcosystemHealthResponse(
            ecosystem_health_score=round(ehs, 1),
            air_quality_score=round(air_quality_score, 1),
            weather_score=round(weather_score, 1),
            green_cover_score=round(green_cover_score, 1),
            biodiversity_score=round(biodiversity_score, 1),
            urban_pressure_score=round(urban_pressure_score, 1),
            overall_status=status,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Ecosystem health calculation error: {e}")
        raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")


@app.get("/restoration/scenarios")
async def get_restoration_scenarios():
    """Get all restoration scenarios"""
    
    try:
        scenarios = restoration_scenarios.to_dict('index')
        
        return {
            "scenarios": [
                {
                    "name": name,
                    "details": details
                }
                for name, details in scenarios.items()
            ]
        }
        
    except Exception as e:
        logger.error(f"Scenarios retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve scenarios: {str(e)}")


@app.get("/restoration/recommend")
async def get_restoration_recommendation(
    budget: float = 500,
    timeline: float = 5,
    priority: str = "balanced"
):
    """Get restoration recommendation based on constraints"""
    
    try:
        # Load all scenarios
        all_scenarios = pd.read_csv(RESULTS_DIR / 'restoration_scenarios.csv')
        
        # Filter by constraints
        filtered = all_scenarios[
            (all_scenarios['Total_Cost_Million_Rs'] <= budget) &
            (all_scenarios['Implementation_Time_Years'] <= timeline)
        ]
        
        if len(filtered) == 0:
            return {
                "message": "No scenarios match your constraints",
                "recommendation": "Consider increasing budget or timeline"
            }
        
        # Select based on priority
        if priority == "air_quality":
            best = filtered.loc[filtered['PM2.5_Target'].idxmin()]
        elif priority == "cost":
            best = filtered.loc[filtered['Total_Cost_Million_Rs'].idxmin()]
        elif priority == "time":
            best = filtered.loc[filtered['Implementation_Time_Years'].idxmin()]
        else:  # balanced
            # Normalize and find best trade-off
            normalized = filtered.copy()
            for col in ['PM2.5_Target', 'Total_Cost_Million_Rs', 'Implementation_Time_Years']:
                normalized[col] = (filtered[col] - filtered[col].min()) / (filtered[col].max() - filtered[col].min())
            
            best_idx = normalized[['PM2.5_Target', 'Total_Cost_Million_Rs', 'Implementation_Time_Years']].sum(axis=1).idxmin()
            best = filtered.loc[best_idx]
        
        return {
            "recommended_scenario": {
                "interventions": {
                    "green_cover_increase": f"{best['Green_Cover_Increase_%']:.1f}%",
                    "tree_plantation": f"{best['Tree_Density_per_km2']:.0f} trees/km²",
                    "vehicle_emission_reduction": f"{best['Vehicle_Emission_Reduction_%']:.1f}%",
                    "industrial_control": f"{best['Industrial_Control_%']:.1f}%",
                    "water_quality_improvement": f"{best['Water_Quality_Improvement_%']:.1f}%",
                    "biodiversity_budget": f"₹{best['Biodiversity_Budget_Million_Rs']:.1f}M"
                },
                "expected_outcomes": {
                    "pm25_target": f"{best['PM2.5_Target']:.1f} µg/m³",
                    "total_cost": f"₹{best['Total_Cost_Million_Rs']:.1f}M",
                    "implementation_time": f"{best['Implementation_Time_Years']:.1f} years"
                }
            },
            "alternatives_count": len(filtered)
        }
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.get("/models/info")
async def get_models_info():
    """Get information about loaded models"""
    
    model_comparison = pd.read_csv(MODELS_DIR / 'model_comparison.csv')
    
    return {
        "models": model_comparison.to_dict('records'),
        "feature_importance_top_10": feature_importance.head(10).to_dict('records')
    }


# Helper functions

def calculate_aqi(pm25, pm10, no2, so2, co, o3):
    """Calculate AQI using US EPA formula"""
    
    # Simplified AQI calculation based on PM2.5 (dominant pollutant in Delhi)
    if pm25 <= 12:
        aqi = (50 / 12) * pm25
    elif pm25 <= 35.4:
        aqi = 50 + ((100 - 50) / (35.4 - 12.1)) * (pm25 - 12.1)
    elif pm25 <= 55.4:
        aqi = 100 + ((150 - 100) / (55.4 - 35.5)) * (pm25 - 35.5)
    elif pm25 <= 150.4:
        aqi = 150 + ((200 - 150) / (150.4 - 55.5)) * (pm25 - 55.5)
    elif pm25 <= 250.4:
        aqi = 200 + ((300 - 200) / (250.4 - 150.5)) * (pm25 - 150.5)
    else:
        aqi = 300 + ((500 - 300) / (500 - 250.5)) * min(pm25 - 250.5, 249.5)
    
    return aqi


def get_aqi_category(aqi):
    """Get AQI category and health advice"""
    
    if aqi <= 50:
        return "Good", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi <= 100:
        return "Moderate", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    elif aqi <= 200:
        return "Unhealthy", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
    elif aqi <= 300:
        return "Very Unhealthy", "Health alert: The risk of health effects is increased for everyone."
    else:
        return "Hazardous", "Health warning of emergency conditions: everyone is more likely to be affected."


def calculate_weather_comfort(temp, humidity, wind_speed):
    """Calculate weather comfort index (0-100)"""
    
    # Optimal: temp 20-25°C, humidity 40-60%, wind 1-3 m/s
    temp_score = max(0, 100 - abs(22.5 - temp) * 4)
    humidity_score = max(0, 100 - abs(50 - humidity) * 2)
    wind_score = max(0, 100 - abs(2 - wind_speed) * 20)
    
    return (temp_score + humidity_score + wind_score) / 3


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
