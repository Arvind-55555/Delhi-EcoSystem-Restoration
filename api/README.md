# Ecosystem Health ML API

FastAPI REST API for ecosystem restoration and air quality prediction

## Installation

```bash
pip install fastapi uvicorn
```

## Running the API

```bash
# From project root
python api/main.py

# Or using uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, access interactive API docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

### Health Check
- `GET /health` - Check API status

### Predictions
- `POST /predict/aqi` - Predict Air Quality Index
- `POST /forecast/pm25` - Forecast PM2.5 for next N days

### Ecosystem Health
- `POST /ecosystem/health` - Calculate comprehensive ecosystem health score

### Restoration Scenarios
- `GET /restoration/scenarios` - Get all restoration scenarios
- `GET /restoration/recommend` - Get personalized recommendation

### Model Information
- `GET /models/info` - Get loaded models and feature importance

## Example Usage

### Predict AQI
```bash
curl -X POST "http://localhost:8000/predict/aqi" \
  -H "Content-Type: application/json" \
  -d '{
    "PM25": 120,
    "PM10": 200,
    "NO2": 50,
    "SO2": 10,
    "CO": 1.5,
    "O3": 40,
    "temperature": 25,
    "humidity": 60,
    "wind_speed": 2,
    "precipitation": 0,
    "green_cover_percentage": 20
  }'
```

### Get Restoration Recommendation
```bash
curl "http://localhost:8000/restoration/recommend?budget=500&timeline=5&priority=balanced"
```
