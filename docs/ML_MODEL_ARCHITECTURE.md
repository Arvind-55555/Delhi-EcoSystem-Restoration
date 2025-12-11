# Machine Learning Model Architecture for Delhi Ecosystem Restoration

## COMPREHENSIVE ML FRAMEWORK

---

## 1. MODEL OVERVIEW

### 1.1 Predictive Models
1. **Air Quality Forecasting Model**
2. **Water Quality Prediction Model**
3. **Urban Heat Island (UHI) Mitigation Model**
4. **Vegetation Health Monitoring Model**
5. **Biodiversity Recovery Model**
6. **Integrated Ecosystem Health Model**

### 1.2 Optimization Models
1. **Multi-Objective Restoration Optimizer**
2. **Spatial Prioritization Engine**
3. **Cost-Benefit Analyzer**

---

## 2. MODEL 1: AIR QUALITY FORECASTING

### 2.1 Problem Formulation
**Type**: Time-series forecasting + Regression  
**Input**: Historical air quality, meteorological, spatial features  
**Output**: Next-day/week/month PM2.5, PM10, NO2, SO2, CO, O3, AQI predictions

### 2.2 Feature Set

#### Input Features (X)
1. **Temporal Features**
   - Hour of day (0-23)
   - Day of week (1-7)
   - Month (1-12)
   - Season (Winter/Summer/Monsoon/Autumn)
   - Is_weekend (binary)
   - Is_holiday (binary)

2. **Lagged Air Quality Features**
   - PM2.5_{t-1}, PM2.5_{t-2}, ..., PM2.5_{t-7} (last 7 days)
   - PM10_{t-1}, PM10_{t-2}, ..., PM10_{t-7}
   - NO2, SO2, CO, O3 lags (1-7 days)
   - AQI_{t-1}, AQI_{t-7}, AQI_{t-30}

3. **Rolling Statistics**
   - PM2.5_rolling_mean_7d
   - PM2.5_rolling_std_7d
   - PM10_rolling_mean_7d
   - PM10_rolling_max_7d

4. **Meteorological Features**
   - Temperature (°C)
   - Humidity (%)
   - Wind speed (km/h)
   - Wind direction (degrees)
   - Rainfall (mm)
   - Atmospheric pressure (hPa)

5. **Spatial Features**
   - Station location (latitude, longitude)
   - Distance to major roads
   - Distance to industrial zones
   - Neighboring station PM2.5 average
   - Urban density (population/sq km)

6. **Auxiliary Features**
   - Vehicle count (if available)
   - Construction activity indicator
   - Biomass burning events (satellite hotspots)
   - Festival days (Diwali, etc.)

#### Target Variables (y)
- PM2.5_{t+1} (next day)
- PM10_{t+1}
- AQI_{t+1}
- Multi-step: PM2.5_{t+1}, PM2.5_{t+2}, ..., PM2.5_{t+7}

### 2.3 Model Architecture

#### Approach 1: Ensemble of Tree-Based Models

**Algorithm**: XGBoost + LightGBM + Random Forest (Voting/Stacking)

**XGBoost Configuration**:
```python
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    objective='reg:squarederror',
    eval_metric='rmse',
    early_stopping_rounds=50
)
```

**LightGBM Configuration**:
```python
import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='regression',
    metric='rmse'
)
```

**Ensemble (Stacking)**:
```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

ensemble = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('rf', RandomForestRegressor(n_estimators=500, max_depth=10))
    ],
    final_estimator=Ridge(alpha=1.0)
)
```

#### Approach 2: Deep Learning (LSTM)

**Architecture**:
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

def build_lstm_model(sequence_length, n_features):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        
        Dense(1)  # Output: PM2.5 prediction
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae', 'mse']
    )
    
    return model

lstm_model = build_lstm_model(sequence_length=7, n_features=20)
```

#### Approach 3: Prophet (for trend + seasonality)

```python
from prophet import Prophet

def train_prophet_model(df):
    prophet_df = df[['timestamp', 'PM2.5']].rename(columns={'timestamp': 'ds', 'PM2.5': 'y'})
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    
    # Add custom seasonalities
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Add regressors (weather, etc.)
    model.add_regressor('temperature')
    model.add_regressor('humidity')
    model.add_regressor('wind_speed')
    
    model.fit(prophet_df)
    return model
```

### 2.4 Training Strategy

**Data Split**:
- Training: 2019-01-01 to 2023-06-30 (70%)
- Validation: 2023-07-01 to 2023-12-31 (15%)
- Test: 2024-01-01 to 2024-12-31 (15%)

**Cross-Validation**: Time-series cross-validation (expanding window)
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, val_index in tscv.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    # Train and evaluate
```

**Hyperparameter Tuning**: Bayesian Optimization
```python
from skopt import BayesSearchCV

param_space = {
    'n_estimators': (100, 1000),
    'max_depth': (3, 12),
    'learning_rate': (0.01, 0.3, 'log-uniform'),
    'subsample': (0.6, 1.0, 'uniform')
}

opt = BayesSearchCV(
    xgb_model,
    param_space,
    n_iter=50,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
opt.fit(X_train, y_train)
```

### 2.5 Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"RMSE: {rmse:.2f} µg/m³")
    print(f"MAE: {mae:.2f} µg/m³")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}
```

**Target Performance**:
- RMSE < 15 µg/m³ for PM2.5
- R² > 0.80
- MAE < 10 µg/m³

---

## 3. MODEL 2: WATER QUALITY PREDICTION

### 3.1 Problem Formulation
**Type**: Multi-output regression + Classification  
**Input**: Water quality parameters, rainfall, discharge, season  
**Output**: BOD, COD, DO, pH, fecal coliform predictions; Water quality class

### 3.2 Feature Set

#### Input Features
1. **Lagged Water Quality**
   - BOD_{t-1}, BOD_{t-2}, BOD_{t-3} (last 3 months)
   - COD_{t-1}, DO_{t-1}, pH_{t-1}
   - Fecal_coliform_{t-1}

2. **Environmental Features**
   - Rainfall (monthly total)
   - River discharge (cumecs)
   - Temperature (°C)
   - Season (Monsoon/Winter/Summer)

3. **Spatial Features**
   - Monitoring location (Palla/Nizamuddin/Okhla)
   - Upstream pollution load
   - Distance from sewage outfalls

4. **Temporal Features**
   - Month, season, year

#### Target Variables
- BOD (mg/L) - continuous
- COD (mg/L) - continuous
- DO (mg/L) - continuous
- Fecal coliform (MPN/100ml) - continuous (log-transformed)
- Water quality class (A/B/C/D/E) - categorical

### 3.3 Model Architecture

**Multi-Output Regressor**:
```python
from sklearn.multioutput import MultiOutputRegressor

multi_model = MultiOutputRegressor(
    xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1)
)

# Fit on multiple targets
multi_model.fit(X_train, y_train[['BOD', 'COD', 'DO', 'pH']])
```

**Classification Model** (for water quality class):
```python
from sklearn.ensemble import GradientBoostingClassifier

classifier = GradientBoostingClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.1
)

classifier.fit(X_train, y_train['water_quality_class'])
```

### 3.4 Evaluation
- **Regression**: RMSE, MAE for each parameter
- **Classification**: Accuracy, F1-score, Confusion matrix

**Target**: Classification accuracy > 85%

---

## 4. MODEL 3: URBAN HEAT ISLAND (UHI) MITIGATION

### 4.1 Problem Formulation
**Type**: Spatial regression + Scenario modeling  
**Input**: LST, NDVI, built-up area, albedo  
**Output**: Temperature reduction potential from green interventions

### 4.2 Feature Set

#### Input Features
1. **Remote Sensing**
   - Land Surface Temperature (LST) from Landsat
   - NDVI (vegetation index)
   - NDBI (built-up index)
   - Albedo

2. **Land Use**
   - % Built-up area in 1km radius
   - % Green cover in 1km radius
   - % Water bodies in 1km radius

3. **Spatial**
   - Latitude, longitude
   - Distance to parks/forests
   - Distance to water bodies

4. **Temporal**
   - Month, season (summer focus)

#### Target Variable
- LST (°C) - continuous
- UHI intensity (LST_urban - LST_rural)

### 4.3 Model Architecture

**Spatial Regression**:
```python
from sklearn.ensemble import RandomForestRegressor

rf_spatial = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_split=10,
    n_jobs=-1
)

rf_spatial.fit(X_train, y_train)
```

**Scenario Simulation**:
```python
def simulate_green_intervention(X, ndvi_increase_pct):
    X_scenario = X.copy()
    X_scenario['NDVI'] *= (1 + ndvi_increase_pct / 100)
    X_scenario['pct_green_cover'] += ndvi_increase_pct
    X_scenario['pct_built_up'] -= ndvi_increase_pct
    
    LST_predicted = rf_spatial.predict(X_scenario)
    LST_baseline = rf_spatial.predict(X)
    
    temperature_reduction = LST_baseline - LST_predicted
    return temperature_reduction

# Simulate 10% increase in green cover
temp_reduction = simulate_green_intervention(X_test, ndvi_increase_pct=10)
print(f"Expected temperature reduction: {temp_reduction.mean():.2f} °C")
```

---

## 5. MODEL 4: VEGETATION HEALTH MONITORING

### 5.1 Problem Formulation
**Type**: Time-series forecasting  
**Input**: Historical NDVI, rainfall, temperature  
**Output**: Future NDVI trends

### 5.2 Model Architecture

**ARIMA Model**:
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima_model = SARIMAX(
    ndvi_timeseries,
    order=(1, 1, 1),  # (p, d, q)
    seasonal_order=(1, 1, 1, 12),  # (P, D, Q, s) - 12 months
    exog=weather_data[['rainfall', 'temperature']]
)

sarima_fit = sarima_model.fit(disp=False)
forecast = sarima_fit.forecast(steps=12, exog=weather_forecast)
```

---

## 6. MODEL 5: BIODIVERSITY RECOVERY

### 6.1 Problem Formulation
**Type**: Classification + Regression  
**Input**: Habitat features, green cover, connectivity  
**Output**: Species richness, biodiversity index

### 6.2 Feature Set

#### Input Features
1. **Habitat Quality**
   - Green cover (%)
   - Forest patch size (ha)
   - Edge density
   - Habitat connectivity index

2. **Environmental**
   - NDVI
   - Distance to water bodies
   - Air quality index

3. **Anthropogenic**
   - Human population density
   - Urbanization rate

#### Target Variable
- Species richness (count)
- Shannon diversity index (continuous)

### 6.3 Model Architecture

**Random Forest Regressor**:
```python
rf_biodiversity = RandomForestRegressor(n_estimators=300, max_depth=8)
rf_biodiversity.fit(X_train, y_train['species_richness'])
```

---

## 7. MODEL 6: INTEGRATED ECOSYSTEM HEALTH MODEL

### 7.1 Composite Ecosystem Health Score

**Formula**:
```
EHS = w1 * AQI_score + w2 * WQI_score + w3 * GCI_score + w4 * BDI_score + w5 * CRI_score

Where:
- AQI_score: Air Quality Index (normalized 0-100, inverted so higher is better)
- WQI_score: Water Quality Index (0-100)
- GCI_score: Green Cover Index (0-100)
- BDI_score: Biodiversity Index (0-100)
- CRI_score: Climate Resilience Index (0-100)
- w1, w2, w3, w4, w5: Weights (sum to 1.0)
```

**Weight Optimization**: Use Analytic Hierarchy Process (AHP) or expert surveys

**Example Weights**:
- w1 (AQI) = 0.30
- w2 (WQI) = 0.20
- w3 (GCI) = 0.25
- w4 (BDI) = 0.15
- w5 (CRI) = 0.10

### 7.2 Model Architecture

**Deep Neural Network (Multi-Output)**:
```python
def build_ecosystem_health_model(n_features):
    inputs = tf.keras.Input(shape=(n_features,))
    
    # Shared layers
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    
    # Output heads
    aqi_out = Dense(1, name='aqi_score')(x)
    wqi_out = Dense(1, name='wqi_score')(x)
    gci_out = Dense(1, name='gci_score')(x)
    bdi_out = Dense(1, name='bdi_score')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=[aqi_out, wqi_out, gci_out, bdi_out])
    
    model.compile(
        optimizer='adam',
        loss={
            'aqi_score': 'mse',
            'wqi_score': 'mse',
            'gci_score': 'mse',
            'bdi_score': 'mse'
        },
        loss_weights={
            'aqi_score': 0.3,
            'wqi_score': 0.2,
            'gci_score': 0.25,
            'bdi_score': 0.25
        },
        metrics=['mae']
    )
    
    return model

ehs_model = build_ecosystem_health_model(n_features=50)
```

### 7.3 Training
```python
history = ehs_model.fit(
    X_train,
    {
        'aqi_score': y_train_aqi,
        'wqi_score': y_train_wqi,
        'gci_score': y_train_gci,
        'bdi_score': y_train_bdi
    },
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
)
```

---

## 8. OPTIMIZATION MODELS

### 8.1 Multi-Objective Restoration Optimizer

**Problem Formulation**:
```
Maximize: f1(x) = Ecosystem Health Score
Maximize: f2(x) = Climate Resilience
Minimize: f3(x) = Implementation Cost
Minimize: f4(x) = Implementation Time

Subject to:
- Budget constraint: Cost(x) ≤ B
- Land availability: Area(x) ≤ A
- Policy constraints: x ∈ feasible_set

Decision variables x:
- x1: Number of trees to plant
- x2: Area for urban parks (ha)
- x3: Wetland restoration area (ha)
- x4: Sewage treatment capacity upgrade (MLD)
- x5: Green roof area (sq km)
- x6: Electric vehicle fleet size
- ...
```

**Algorithm**: NSGA-II (Non-dominated Sorting Genetic Algorithm)

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import Problem
from pymoo.optimize import minimize

class EcosystemRestorationProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=10,  # 10 decision variables
            n_obj=3,   # 3 objectives
            n_constr=2,  # 2 constraints
            xl=np.zeros(10),  # Lower bounds
            xu=np.array([1e6, 1000, 500, 200, 50, 10000, 1e4, 500, 100, 50])  # Upper bounds
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        # Objective 1: Maximize ecosystem health (convert to minimization: -f1)
        ehs = self.predict_ecosystem_health(x)
        f1 = -ehs
        
        # Objective 2: Minimize cost
        f2 = self.calculate_cost(x)
        
        # Objective 3: Minimize implementation time
        f3 = self.calculate_time(x)
        
        # Constraints
        g1 = self.calculate_cost(x) - self.budget_limit  # Cost <= budget
        g2 = self.calculate_area(x) - self.land_availability  # Area <= available
        
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2])
    
    def predict_ecosystem_health(self, x):
        # Use trained ML model to predict EHS given intervention x
        features = self.construct_features(x)
        ehs_predicted = ehs_model.predict(features)[0]
        return ehs_predicted
    
    def calculate_cost(self, x):
        # Cost model
        tree_cost = x[0] * 500  # ₹500 per tree
        park_cost = x[1] * 5e6  # ₹5 crore per hectare
        wetland_cost = x[2] * 3e6
        sewage_cost = x[3] * 10e6  # ₹10 crore per MLD capacity
        # ...
        total_cost = tree_cost + park_cost + wetland_cost + sewage_cost
        return total_cost

# Run optimization
problem = EcosystemRestorationProblem()
algorithm = NSGA2(pop_size=100)

res = minimize(
    problem,
    algorithm,
    ('n_gen', 200),  # 200 generations
    seed=1,
    verbose=True
)

# Extract Pareto-optimal solutions
pareto_front = res.F
pareto_solutions = res.X
```

**Output**: Set of Pareto-optimal restoration strategies

### 8.2 Spatial Prioritization Engine

**Method**: Multi-Criteria Decision Analysis (MCDA) with GIS

```python
import geopandas as gpd
from scipy.stats import rankdata

def spatial_prioritization(gdf, weights):
    """
    gdf: GeoDataFrame with grid cells and criteria scores
    weights: dict of criteria weights
    """
    criteria = ['pollution_severity', 'population_exposed', 'restoration_potential', 
                'cost_effectiveness', 'co_benefits']
    
    # Normalize criteria (0-1 scale)
    for criterion in criteria:
        gdf[f'{criterion}_norm'] = (gdf[criterion] - gdf[criterion].min()) / \
                                     (gdf[criterion].max() - gdf[criterion].min())
    
    # Calculate weighted score
    gdf['priority_score'] = sum(
        gdf[f'{criterion}_norm'] * weights[criterion] for criterion in criteria
    )
    
    # Rank
    gdf['priority_rank'] = rankdata(-gdf['priority_score'], method='min')
    
    # Classify
    gdf['priority_class'] = pd.cut(
        gdf['priority_score'],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    return gdf

# Example weights
weights = {
    'pollution_severity': 0.30,
    'population_exposed': 0.25,
    'restoration_potential': 0.20,
    'cost_effectiveness': 0.15,
    'co_benefits': 0.10
}

priority_map = spatial_prioritization(delhi_grid, weights)
priority_map.plot(column='priority_class', legend=True, cmap='RdYlGn_r')
```

---

## 9. MODEL DEPLOYMENT PIPELINE

### 9.1 Model Serving Architecture

```
[Frontend Dashboard (React)] 
         ↓ HTTP Request
[API Gateway (FastAPI/Flask)]
         ↓
[Model Inference Service]
  - Load trained models
  - Preprocess input
  - Generate predictions
  - Post-process output
         ↓
[Response JSON]
         ↓
[Frontend Visualization]
```

### 9.2 FastAPI Backend

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Delhi Ecosystem Restoration API")

# Load trained models
air_quality_model = joblib.load('models/air_quality_xgb.pkl')
ecosystem_health_model = tf.keras.models.load_model('models/ecosystem_health_nn.h5')

class AirQualityRequest(BaseModel):
    station: str
    features: dict  # {PM2.5_t1: 150, PM10_t1: 200, temperature: 30, ...}

class AirQualityResponse(BaseModel):
    predicted_pm25: float
    predicted_aqi: int
    forecast_7day: list

@app.post("/predict/air_quality", response_model=AirQualityResponse)
def predict_air_quality(request: AirQualityRequest):
    try:
        # Preprocess
        X_input = preprocess_features(request.features)
        
        # Predict
        pm25_pred = air_quality_model.predict(X_input)[0]
        aqi_pred = calculate_aqi(pm25_pred)
        
        # 7-day forecast (iterative)
        forecast = []
        for day in range(1, 8):
            # Update features with previous prediction
            pm25_day = air_quality_model.predict(X_input)[0]
            forecast.append(float(pm25_day))
            X_input = update_features(X_input, pm25_day)
        
        return AirQualityResponse(
            predicted_pm25=float(pm25_pred),
            predicted_aqi=int(aqi_pred),
            forecast_7day=forecast
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/restoration")
def optimize_restoration(budget: float, objectives: list):
    # Run NSGA-II optimizer
    pareto_solutions = run_optimization(budget, objectives)
    return {"pareto_solutions": pareto_solutions.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 9.3 Docker Deployment

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**requirements.txt**:
```
fastapi==0.104.1
uvicorn==0.24.0
scikit-learn==1.3.2
xgboost==2.0.2
lightgbm==4.1.0
tensorflow==2.14.0
pandas==2.1.3
numpy==1.26.2
joblib==1.3.2
pydantic==2.5.0
```

**Build and Run**:
```bash
docker build -t delhi-ecosystem-ml .
docker run -p 8000:8000 delhi-ecosystem-ml
```

---

## 10. MODEL MONITORING & MAINTENANCE

### 10.1 Performance Monitoring

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("delhi_ecosystem_restoration")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params(xgb_model.get_params())
    
    # Log metrics
    mlflow.log_metric("rmse_pm25", rmse)
    mlflow.log_metric("r2_score", r2)
    
    # Log model
    mlflow.sklearn.log_model(xgb_model, "air_quality_model")
    
    # Log artifacts
    mlflow.log_artifact("feature_importance.png")
```

### 10.2 Drift Detection

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def detect_drift(reference_data, current_data):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("drift_report.html")
    
    drift_detected = report.as_dict()['metrics'][0]['result']['dataset_drift']
    return drift_detected

# Check weekly
if detect_drift(X_train, X_current_week):
    print("Data drift detected! Consider retraining model.")
    # Trigger retraining pipeline
```

### 10.3 Retraining Pipeline

**Scheduled Retraining** (every 3 months):
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'ecosystem_model_retrain',
    default_args=default_args,
    schedule_interval='0 0 1 */3 *',  # Every 3 months
    catchup=False
)

def fetch_new_data():
    # Download latest data from CPCB, IMD, etc.
    pass

def retrain_models():
    # Retrain with updated data
    pass

def evaluate_new_model():
    # Compare with current production model
    pass

def deploy_if_better():
    # Deploy new model if performance improvement > 5%
    pass

task1 = PythonOperator(task_id='fetch_data', python_callable=fetch_new_data, dag=dag)
task2 = PythonOperator(task_id='retrain', python_callable=retrain_models, dag=dag)
task3 = PythonOperator(task_id='evaluate', python_callable=evaluate_new_model, dag=dag)
task4 = PythonOperator(task_id='deploy', python_callable=deploy_if_better, dag=dag)

task1 >> task2 >> task3 >> task4
```

---

## 11. SUMMARY OF ML TECHNIQUES

| **Model** | **Algorithm** | **Input Features** | **Output** | **Evaluation Metric** |
|-----------|---------------|---------------------|------------|----------------------|
| Air Quality Forecasting | XGBoost + LSTM | Lagged pollutants, weather, spatial | PM2.5, AQI (next day) | RMSE < 15 µg/m³, R² > 0.80 |
| Water Quality Prediction | Multi-Output XGBoost | Lagged BOD/COD, rainfall, discharge | BOD, COD, DO, pH | RMSE per parameter |
| UHI Mitigation | Random Forest Regressor | LST, NDVI, LULC | Temperature reduction | MAE < 0.5 °C |
| Vegetation Health | SARIMA + Prophet | NDVI time-series, weather | Future NDVI | MASE < 1.0 |
| Biodiversity Recovery | Random Forest | Habitat quality, green cover | Species richness | R² > 0.70 |
| Integrated Ecosystem Health | Deep Neural Network | All indicators | Composite EHS | Correlation > 0.90 with expert |
| Multi-Objective Optimization | NSGA-II | Intervention parameters | Pareto-optimal strategies | Hypervolume indicator |
| Spatial Prioritization | MCDA + GIS | Criteria scores (pollution, cost, etc.) | Priority zones | Accuracy vs. expert ranking |

---

## 12. EXPECTED MODEL PERFORMANCE

**Air Quality Forecasting**:
- 1-day ahead: RMSE = 12-15 µg/m³, R² = 0.82-0.85
- 7-day ahead: RMSE = 20-25 µg/m³, R² = 0.70-0.75

**Water Quality Prediction**:
- BOD: RMSE = 5-8 mg/L
- Classification: Accuracy = 85-90%

**UHI Mitigation**:
- Temperature prediction: MAE = 0.3-0.5 °C
- Scenario simulation: ±10% uncertainty

**Ecosystem Health Model**:
- Overall R² = 0.88-0.92
- Sub-component correlation > 0.85

**Optimization**:
- 50-100 Pareto-optimal solutions generated
- Cost savings: 15-25% vs. heuristic approaches

---

**END OF ML MODEL ARCHITECTURE DOCUMENT**
