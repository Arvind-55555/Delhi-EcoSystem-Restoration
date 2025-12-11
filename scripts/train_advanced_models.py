#!/usr/bin/env python3
"""
Advanced Model Training: LSTM and Prophet for Time-Series Forecasting
Delhi Ecosystem Restoration ML Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available, LSTM training will be skipped")

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available, Prophet training will be skipped")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/advanced_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_FEATURES_DIR = BASE_DIR / 'data' / 'features'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class AdvancedModelTrainer:
    """Train advanced time-series forecasting models"""
    
    def __init__(self):
        self.df = None
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        """Load master dataset"""
        logger.info("=" * 70)
        logger.info("LOADING DATA FOR ADVANCED MODELS")
        logger.info("=" * 70)
        
        self.df = pd.read_parquet(DATA_FEATURES_DIR / 'master_dataset.parquet')
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"✓ Loaded: {len(self.df):,} records")
        logger.info(f"  Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        
        return True
    
    def prepare_lstm_data(self, sequence_length=30, target='PM2.5'):
        """Prepare sequences for LSTM"""
        logger.info("\n" + "=" * 70)
        logger.info(f"PREPARING LSTM DATA (sequence_length={sequence_length})")
        logger.info("=" * 70)
        
        # Select features for LSTM
        feature_cols = [
            'PM2.5', 'PM10', 'AQI', 'temp_mean_C', 'humidity_percent',
            'wind_speed_ms', 'precipitation_mm', 'month', 'day_of_year',
            'PM2.5_rolling_mean_7d', 'AQI_rolling_mean_7d'
        ]
        
        data = self.df[feature_cols].values
        
        # Normalize
        data_scaled = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(data_scaled)):
            X.append(data_scaled[i-sequence_length:i])
            y.append(data_scaled[i, 0])  # Predict PM2.5 (first column)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train/test split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(f"✓ Sequences created:")
        logger.info(f"  Train: {len(X_train)} sequences")
        logger.info(f"  Test: {len(X_test)} sequences")
        logger.info(f"  Sequence shape: {X_train.shape}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_lstm(self):
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, skipping LSTM")
            return None
        
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING LSTM MODEL")
        logger.info("=" * 70)
        
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_lstm_data()
        
        # Build LSTM model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        logger.info("Model architecture:")
        model.summary(print_fn=logger.info)
        
        # Train
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        y_train_pred = model.predict(X_train, verbose=0)
        y_test_pred = model.predict(X_test, verbose=0)
        
        # Inverse transform predictions
        y_train_actual = self.scaler.inverse_transform(
            np.concatenate([y_train.reshape(-1, 1), np.zeros((len(y_train), len(feature_cols)-1))], axis=1)
        )[:, 0]
        
        y_test_actual = self.scaler.inverse_transform(
            np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), len(feature_cols)-1))], axis=1)
        )[:, 0]
        
        y_train_pred_actual = self.scaler.inverse_transform(
            np.concatenate([y_train_pred, np.zeros((len(y_train_pred), len(feature_cols)-1))], axis=1)
        )[:, 0]
        
        y_test_pred_actual = self.scaler.inverse_transform(
            np.concatenate([y_test_pred, np.zeros((len(y_test_pred), len(feature_cols)-1))], axis=1)
        )[:, 0]
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_actual))
        train_r2 = r2_score(y_train_actual, y_train_pred_actual)
        test_r2 = r2_score(y_test_actual, y_test_pred_actual)
        
        logger.info("\n✓ LSTM Model trained:")
        logger.info(f"  Train RMSE: {train_rmse:.2f} µg/m³")
        logger.info(f"  Test RMSE: {test_rmse:.2f} µg/m³")
        logger.info(f"  Train R²: {train_r2:.4f}")
        logger.info(f"  Test R²: {test_r2:.4f}")
        logger.info(f"  Epochs trained: {len(history.history['loss'])}")
        
        # Save model
        model.save(MODELS_DIR / 'lstm_model.h5')
        joblib.dump(self.scaler, MODELS_DIR / 'lstm_scaler.pkl')
        
        return {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'history': history.history
        }
    
    def prepare_prophet_data(self):
        """Prepare data for Prophet"""
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING PROPHET DATA")
        logger.info("=" * 70)
        
        # Prophet requires 'ds' (date) and 'y' (target) columns
        prophet_df = self.df[['date', 'PM2.5']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Train/test split (80/20)
        split_idx = int(len(prophet_df) * 0.8)
        train_df = prophet_df.iloc[:split_idx]
        test_df = prophet_df.iloc[split_idx:]
        
        logger.info(f"✓ Data prepared:")
        logger.info(f"  Train: {len(train_df)} days")
        logger.info(f"  Test: {len(test_df)} days")
        
        return train_df, test_df
    
    def train_prophet(self):
        """Train Prophet model"""
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available, skipping")
            return None
        
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING PROPHET MODEL")
        logger.info("=" * 70)
        
        train_df, test_df = self.prepare_prophet_data()
        
        # Build Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )
        
        # Add custom seasonalities
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Fit model
        model.fit(train_df)
        
        # Make predictions
        future_train = model.make_future_dataframe(periods=0)
        forecast_train = model.predict(future_train)
        
        future_test = model.make_future_dataframe(periods=len(test_df))
        forecast_test = model.predict(future_test)
        
        # Extract predictions
        train_pred = forecast_train['yhat'].values
        test_pred = forecast_test.tail(len(test_df))['yhat'].values
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(train_df['y'], train_pred))
        test_rmse = np.sqrt(mean_squared_error(test_df['y'], test_pred))
        train_r2 = r2_score(train_df['y'], train_pred)
        test_r2 = r2_score(test_df['y'], test_pred)
        
        logger.info("\n✓ Prophet Model trained:")
        logger.info(f"  Train RMSE: {train_rmse:.2f} µg/m³")
        logger.info(f"  Test RMSE: {test_rmse:.2f} µg/m³")
        logger.info(f"  Train R²: {train_r2:.4f}")
        logger.info(f"  Test R²: {test_r2:.4f}")
        
        # Save model
        joblib.dump(model, MODELS_DIR / 'prophet_model.pkl')
        
        return {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    
    def generate_comparison_report(self, lstm_results, prophet_results):
        """Generate comparison report"""
        logger.info("\n" + "=" * 70)
        logger.info("ADVANCED MODELS COMPARISON")
        logger.info("=" * 70)
        
        results = []
        
        # Load baseline results
        baseline_comparison = pd.read_csv(MODELS_DIR / 'model_comparison.csv')
        
        for _, row in baseline_comparison.iterrows():
            results.append({
                'Model': row['Model'],
                'Type': 'Baseline',
                'Train RMSE': row['Train RMSE'],
                'Test RMSE': row['Test RMSE'],
                'Train R²': row['Train R²'],
                'Test R²': row['Test R²']
            })
        
        if lstm_results:
            results.append({
                'Model': 'LSTM',
                'Type': 'Advanced',
                'Train RMSE': lstm_results['train_rmse'],
                'Test RMSE': lstm_results['test_rmse'],
                'Train R²': lstm_results['train_r2'],
                'Test R²': lstm_results['test_r2']
            })
        
        if prophet_results:
            results.append({
                'Model': 'Prophet',
                'Type': 'Advanced',
                'Train RMSE': prophet_results['train_rmse'],
                'Test RMSE': prophet_results['test_rmse'],
                'Train R²': prophet_results['train_r2'],
                'Test R²': prophet_results['test_r2']
            })
        
        comparison_df = pd.DataFrame(results).sort_values('Test RMSE')
        
        logger.info("\n" + comparison_df.to_string(index=False))
        
        # Save
        comparison_df.to_csv(MODELS_DIR / 'all_models_comparison.csv', index=False)
        
        logger.info(f"\n✓ Best Model: {comparison_df.iloc[0]['Model']}")
        logger.info(f"  Test RMSE: {comparison_df.iloc[0]['Test RMSE']:.2f} µg/m³")
        logger.info(f"  Test R²: {comparison_df.iloc[0]['Test R²']:.4f}")
        
        return comparison_df

def main():
    """Main training pipeline"""
    logger.info("Starting advanced model training...")
    
    trainer = AdvancedModelTrainer()
    
    try:
        # Load data
        trainer.load_data()
        
        # Train LSTM
        lstm_results = trainer.train_lstm()
        
        # Train Prophet
        prophet_results = trainer.train_prophet()
        
        # Generate comparison
        if lstm_results or prophet_results:
            comparison = trainer.generate_comparison_report(lstm_results, prophet_results)
        
        logger.info("\n" + "=" * 70)
        logger.info("ADVANCED MODEL TRAINING COMPLETE ✓")
        logger.info("=" * 70)
        logger.info(f"\nModels saved in: {MODELS_DIR}")
        
    except Exception as e:
        logger.error(f"Error in advanced model training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
