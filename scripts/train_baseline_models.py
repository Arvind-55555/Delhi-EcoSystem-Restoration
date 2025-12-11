#!/usr/bin/env python3
"""
Baseline Model Training for Delhi Ecosystem Restoration ML Project
Trains Linear Regression, Random Forest, and XGBoost models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_FEATURES_DIR = BASE_DIR / 'data' / 'features'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class BaselineModelTrainer:
    """Train and evaluate baseline ML models"""
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_cols = None
        self.results = {}
        
    def load_data(self):
        """Load feature-engineered dataset"""
        logger.info("=" * 70)
        logger.info("LOADING MASTER DATASET")
        logger.info("=" * 70)
        
        # Load parquet (faster than CSV)
        file_path = DATA_FEATURES_DIR / 'master_dataset.parquet'
        self.df = pd.read_parquet(file_path)
        
        logger.info(f"✓ Loaded dataset: {len(self.df):,} records")
        logger.info(f"  Features: {len(self.df.columns)}")
        logger.info(f"  Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        
        return True
    
    def prepare_features(self, target='PM2.5'):
        """Prepare features and target for modeling"""
        logger.info("\n" + "=" * 70)
        logger.info(f"PREPARING FEATURES FOR TARGET: {target}")
        logger.info("=" * 70)
        
        df = self.df.copy()
        
        # Exclude non-feature columns
        exclude_cols = [
            'date', target, 'AQI_Category', 'EHS_Category',
            'season', 'PM2.5_trend_7d', 'PM10_trend_7d', 'AQI_trend_7d'  # Categorical
        ]
        
        # Additional targets to exclude
        other_targets = ['PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI', 'Ecosystem_Health_Score']
        if target not in other_targets:
            exclude_cols.extend(other_targets)
        else:
            other_targets.remove(target)
            exclude_cols.extend(other_targets)
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove any remaining NaN values
        df_clean = df[feature_cols + [target]].dropna()
        
        X = df_clean[feature_cols]
        y = df_clean[target]
        
        logger.info(f"✓ Features selected: {len(feature_cols)}")
        logger.info(f"  Records after cleaning: {len(df_clean):,}")
        logger.info(f"  Target stats:")
        logger.info(f"    Min: {y.min():.2f}")
        logger.info(f"    Max: {y.max():.2f}")
        logger.info(f"    Mean: {y.mean():.2f}")
        logger.info(f"    Std: {y.std():.2f}")
        
        self.feature_cols = feature_cols
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """Split data into train/test sets (temporal split)"""
        logger.info("\n" + "=" * 70)
        logger.info("SPLITTING DATA")
        logger.info("=" * 70)
        
        # Temporal split (last 20% for testing)
        split_index = int(len(X) * (1 - test_size))
        
        self.X_train = X.iloc[:split_index]
        self.X_test = X.iloc[split_index:]
        self.y_train = y.iloc[:split_index]
        self.y_test = y.iloc[split_index:]
        
        logger.info(f"✓ Train set: {len(self.X_train):,} records ({(1-test_size)*100:.0f}%)")
        logger.info(f"✓ Test set: {len(self.X_test):,} records ({test_size*100:.0f}%)")
        logger.info(f"  Train period: {self.df.iloc[:split_index]['date'].min()} to {self.df.iloc[:split_index]['date'].max()}")
        logger.info(f"  Test period: {self.df.iloc[split_index:]['date'].min()} to {self.df.iloc[split_index:]['date'].max()}")
        
        return True
    
    def train_linear_regression(self):
        """Train Linear Regression baseline"""
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING LINEAR REGRESSION")
        logger.info("=" * 70)
        
        model = Ridge(alpha=1.0)
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        logger.info(f"✓ Linear Regression trained:")
        logger.info(f"  Train RMSE: {train_rmse:.2f}")
        logger.info(f"  Test RMSE: {test_rmse:.2f}")
        logger.info(f"  Train R²: {train_r2:.4f}")
        logger.info(f"  Test R²: {test_r2:.4f}")
        logger.info(f"  Test MAE: {test_mae:.2f}")
        
        # Save model
        joblib.dump(model, MODELS_DIR / 'linear_regression.pkl')
        
        self.results['Linear Regression'] = {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae
        }
        
        return model
    
    def train_random_forest(self):
        """Train Random Forest model"""
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING RANDOM FOREST")
        logger.info("=" * 70)
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        logger.info(f"✓ Random Forest trained:")
        logger.info(f"  Train RMSE: {train_rmse:.2f}")
        logger.info(f"  Test RMSE: {test_rmse:.2f}")
        logger.info(f"  Train R²: {train_r2:.4f}")
        logger.info(f"  Test R²: {test_r2:.4f}")
        logger.info(f"  Test MAE: {test_mae:.2f}")
        
        # Feature importance (top 10)
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\n  Top 10 Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")
        
        # Save model and feature importance
        joblib.dump(model, MODELS_DIR / 'random_forest.pkl')
        feature_importance.to_csv(MODELS_DIR / 'feature_importance_rf.csv', index=False)
        
        self.results['Random Forest'] = {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'feature_importance': feature_importance
        }
        
        return model
    
    def train_xgboost(self):
        """Train XGBoost model"""
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING XGBOOST")
        logger.info("=" * 70)
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        logger.info(f"✓ XGBoost trained:")
        logger.info(f"  Train RMSE: {train_rmse:.2f}")
        logger.info(f"  Test RMSE: {test_rmse:.2f}")
        logger.info(f"  Train R²: {train_r2:.4f}")
        logger.info(f"  Test R²: {test_r2:.4f}")
        logger.info(f"  Test MAE: {test_mae:.2f}")
        
        # Feature importance (top 10)
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\n  Top 10 Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")
        
        # Save model and feature importance
        joblib.dump(model, MODELS_DIR / 'xgboost.pkl')
        feature_importance.to_csv(MODELS_DIR / 'feature_importance_xgb.csv', index=False)
        
        self.results['XGBoost'] = {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'feature_importance': feature_importance
        }
        
        return model
    
    def generate_model_comparison_report(self):
        """Generate comparison report for all models"""
        logger.info("\n" + "=" * 70)
        logger.info("MODEL COMPARISON REPORT")
        logger.info("=" * 70)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Train RMSE': [self.results[m]['train_rmse'] for m in self.results.keys()],
            'Test RMSE': [self.results[m]['test_rmse'] for m in self.results.keys()],
            'Train R²': [self.results[m]['train_r2'] for m in self.results.keys()],
            'Test R²': [self.results[m]['test_r2'] for m in self.results.keys()],
            'Test MAE': [self.results[m]['test_mae'] for m in self.results.keys()]
        })
        
        comparison = comparison.sort_values('Test RMSE')
        
        logger.info("\n" + comparison.to_string(index=False))
        
        # Save comparison
        comparison.to_csv(MODELS_DIR / 'model_comparison.csv', index=False)
        
        # Generate report file
        report_path = BASE_DIR / 'logs' / 'model_training_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MODEL TRAINING REPORT - DELHI ECOSYSTEM RESTORATION\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Dataset:\n")
            f.write(f"  Records: {len(self.X_train) + len(self.X_test):,}\n")
            f.write(f"  Features: {len(self.feature_cols)}\n")
            f.write(f"  Train/Test Split: {len(self.X_train)}/{len(self.X_test)}\n\n")
            
            f.write("Model Comparison:\n")
            f.write(comparison.to_string(index=False) + "\n\n")
            
            f.write("Best Model: " + comparison.iloc[0]['Model'] + "\n")
            f.write(f"  Test RMSE: {comparison.iloc[0]['Test RMSE']:.2f}\n")
            f.write(f"  Test R²: {comparison.iloc[0]['Test R²']:.4f}\n")
            f.write(f"  Test MAE: {comparison.iloc[0]['Test MAE']:.2f}\n\n")
            
            f.write("Target Performance:\n")
            f.write("  Goal: Test RMSE < 15 µg/m³, R² > 0.80\n")
            if comparison.iloc[0]['Test RMSE'] < 15 and comparison.iloc[0]['Test R²'] > 0.80:
                f.write("  Status: ✓ ACHIEVED\n")
            else:
                f.write("  Status: Not yet achieved (advanced models needed)\n")
        
        logger.info(f"\n✓ Report saved: {report_path}")
        
        return comparison

def main():
    """Main training pipeline"""
    logger.info("Starting baseline model training...")
    
    trainer = BaselineModelTrainer()
    
    try:
        # Load data
        trainer.load_data()
        
        # Prepare features (target: PM2.5)
        X, y = trainer.prepare_features(target='PM2.5')
        
        # Split data
        trainer.split_data(X, y, test_size=0.2)
        
        # Train models
        trainer.train_linear_regression()
        trainer.train_random_forest()
        trainer.train_xgboost()
        
        # Generate comparison
        comparison = trainer.generate_model_comparison_report()
        
        logger.info("\n" + "=" * 70)
        logger.info("BASELINE MODEL TRAINING COMPLETE ✓")
        logger.info("=" * 70)
        logger.info(f"\nBest Model: {comparison.iloc[0]['Model']}")
        logger.info(f"Test RMSE: {comparison.iloc[0]['Test RMSE']:.2f} µg/m³")
        logger.info(f"Test R²: {comparison.iloc[0]['Test R²']:.4f}")
        logger.info(f"\nModels saved in: {MODELS_DIR}")
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
