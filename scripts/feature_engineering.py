#!/usr/bin/env python3
"""
Feature Engineering for Delhi Ecosystem Restoration ML Project
Creates lag features, rolling statistics, and composite indices
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_engineering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
DATA_FEATURES_DIR = BASE_DIR / 'data' / 'features'
DATA_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

class FeatureEngineer:
    """Create advanced features for ML models"""
    
    def __init__(self):
        self.citywide_df = None
        self.annual_df = None
        
    def load_processed_data(self):
        """Load preprocessed data"""
        logger.info("=" * 70)
        logger.info("LOADING PREPROCESSED DATA")
        logger.info("=" * 70)
        
        # Load city-wide daily average
        self.citywide_df = pd.read_csv(DATA_PROCESSED_DIR / 'citywide_daily_average.csv')
        self.citywide_df['date'] = pd.to_datetime(self.citywide_df['date'])
        logger.info(f"✓ City-wide daily: {len(self.citywide_df):,} records")
        
        # Load annual data
        self.annual_df = pd.read_csv(DATA_PROCESSED_DIR / 'annual_merged.csv')
        logger.info(f"✓ Annual data: {len(self.annual_df):,} records")
        
        return True
    
    def create_lag_features(self):
        """Create lag features for time-series prediction"""
        logger.info("\n" + "=" * 70)
        logger.info("CREATING LAG FEATURES")
        logger.info("=" * 70)
        
        df = self.citywide_df.copy().sort_values('date')
        
        # Pollutants to create lags for
        pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
        
        # Lag periods (1 day, 7 days, 30 days ago)
        lag_periods = [1, 7, 30]
        
        features_created = 0
        
        for pollutant in pollutants:
            for lag in lag_periods:
                col_name = f'{pollutant}_lag_{lag}d'
                df[col_name] = df[pollutant].shift(lag)
                features_created += 1
        
        logger.info(f"✓ Created {features_created} lag features")
        logger.info(f"  Pollutants: {len(pollutants)}")
        logger.info(f"  Lag periods: {lag_periods}")
        
        self.citywide_df = df
        return df
    
    def create_rolling_features(self):
        """Create rolling window statistics"""
        logger.info("\n" + "=" * 70)
        logger.info("CREATING ROLLING STATISTICS")
        logger.info("=" * 70)
        
        df = self.citywide_df.copy()
        
        pollutants = ['PM2.5', 'PM10', 'AQI']
        windows = [7, 30]  # 7-day and 30-day windows
        
        features_created = 0
        
        for pollutant in pollutants:
            for window in windows:
                # Rolling mean
                df[f'{pollutant}_rolling_mean_{window}d'] = df[pollutant].rolling(window=window, min_periods=1).mean()
                
                # Rolling max
                df[f'{pollutant}_rolling_max_{window}d'] = df[pollutant].rolling(window=window, min_periods=1).max()
                
                # Rolling std
                df[f'{pollutant}_rolling_std_{window}d'] = df[pollutant].rolling(window=window, min_periods=1).std()
                
                features_created += 3
        
        logger.info(f"✓ Created {features_created} rolling features")
        logger.info(f"  Statistics: mean, max, std")
        logger.info(f"  Windows: {windows} days")
        
        self.citywide_df = df
        return df
    
    def create_trend_features(self):
        """Create trend indicators"""
        logger.info("\n" + "=" * 70)
        logger.info("CREATING TREND FEATURES")
        logger.info("=" * 70)
        
        df = self.citywide_df.copy()
        
        pollutants = ['PM2.5', 'PM10', 'AQI']
        
        features_created = 0
        
        for pollutant in pollutants:
            # 7-day change
            df[f'{pollutant}_change_7d'] = df[pollutant] - df[f'{pollutant}_lag_7d']
            
            # 30-day change
            df[f'{pollutant}_change_30d'] = df[pollutant] - df[f'{pollutant}_lag_30d']
            
            # Trend direction (improving/stable/worsening)
            df[f'{pollutant}_trend_7d'] = np.where(
                df[f'{pollutant}_change_7d'] < -5, 'Improving',
                np.where(df[f'{pollutant}_change_7d'] > 5, 'Worsening', 'Stable')
            )
            
            features_created += 3
        
        logger.info(f"✓ Created {features_created} trend features")
        logger.info(f"  Change periods: 7d, 30d")
        logger.info(f"  Trend categories: Improving/Stable/Worsening")
        
        self.citywide_df = df
        return df
    
    def create_composite_indices(self):
        """Create composite environmental indices"""
        logger.info("\n" + "=" * 70)
        logger.info("CREATING COMPOSITE INDICES")
        logger.info("=" * 70)
        
        df = self.citywide_df.copy()
        
        # 1. Air Quality Health Index (normalized 0-100, higher is worse)
        df['AQHI'] = (
            (df['PM2.5'] / 250) * 0.4 +  # PM2.5 weight: 40%
            (df['PM10'] / 400) * 0.3 +   # PM10 weight: 30%
            (df['NO2'] / 80) * 0.15 +    # NO2 weight: 15%
            (df['O3'] / 100) * 0.15      # O3 weight: 15%
        ) * 100
        df['AQHI'] = df['AQHI'].clip(0, 100)
        
        # 2. Weather Comfort Index (0-100, higher is better)
        ideal_temp = 25  # Ideal temperature for Delhi
        df['Weather_Comfort_Index'] = (
            100 - abs(df['temp_mean_C'] - ideal_temp) * 2 -  # Temperature deviation
            (df['humidity_percent'] - 50).abs() * 0.5         # Humidity deviation from 50%
        ).clip(0, 100)
        
        # 3. Pollution Severity Score (0-10 scale)
        df['Pollution_Severity'] = pd.cut(
            df['PM2.5'],
            bins=[0, 30, 60, 90, 120, 250, 1000],
            labels=[1, 2, 3, 4, 5, 6]
        ).astype(float)
        
        # 4. Seasonal Pollution Index (comparison to seasonal average)
        season_avg = df.groupby('season')['PM2.5'].transform('mean')
        df['Seasonal_Pollution_Index'] = (df['PM2.5'] / season_avg) * 100
        
        logger.info("✓ Created 4 composite indices:")
        logger.info(f"  - AQHI (Air Quality Health Index): {df['AQHI'].mean():.1f} avg")
        logger.info(f"  - Weather Comfort Index: {df['Weather_Comfort_Index'].mean():.1f} avg")
        logger.info(f"  - Pollution Severity: {df['Pollution_Severity'].mean():.1f} avg (1-6 scale)")
        logger.info(f"  - Seasonal Pollution Index: {df['Seasonal_Pollution_Index'].mean():.1f} avg")
        
        self.citywide_df = df
        return df
    
    def merge_annual_features(self):
        """Merge annual features (forest, biodiversity, population) into daily data"""
        logger.info("\n" + "=" * 70)
        logger.info("MERGING ANNUAL FEATURES")
        logger.info("=" * 70)
        
        df = self.citywide_df.copy()
        
        # Add year column if not present
        if 'year' not in df.columns:
            df['year'] = df['date'].dt.year
        
        # Merge annual data
        df = df.merge(
            self.annual_df[[
                'year', 'forest_cover_sq_km', 'total_green_cover_sq_km', 
                'green_cover_percentage', 'total_species', 'threatened_species',
                'population', 'density_per_sq_km', 'total_vehicles'
            ]],
            on='year',
            how='left'
        )
        
        # Forward fill for missing years
        annual_cols = [
            'forest_cover_sq_km', 'total_green_cover_sq_km', 'green_cover_percentage',
            'total_species', 'threatened_species', 'population', 
            'density_per_sq_km', 'total_vehicles'
        ]
        
        for col in annual_cols:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"✓ Merged {len(annual_cols)} annual features")
        logger.info(f"  Forest cover: {df['green_cover_percentage'].mean():.2f}% avg")
        logger.info(f"  Population: {df['population'].mean():.0f} avg")
        logger.info(f"  Vehicles: {df['total_vehicles'].mean():.0f} avg")
        
        self.citywide_df = df
        return df
    
    def create_interaction_features(self):
        """Create interaction features between variables"""
        logger.info("\n" + "=" * 70)
        logger.info("CREATING INTERACTION FEATURES")
        logger.info("=" * 70)
        
        df = self.citywide_df.copy()
        
        # 1. Temperature * Humidity (heat stress)
        df['Heat_Stress_Index'] = df['temp_mean_C'] * (df['humidity_percent'] / 100)
        
        # 2. Wind * Pollution (dispersion effect)
        df['Wind_Dispersion_Effect'] = df['wind_speed_ms'] / (df['PM2.5'] + 1)  # +1 to avoid div by zero
        
        # 3. Population density * Vehicles (emission pressure)
        df['Emission_Pressure'] = (df['density_per_sq_km'] / 1000) * (df['total_vehicles'] / 1e6)
        
        # 4. Green cover * Temperature (cooling effect)
        df['Green_Cooling_Effect'] = df['green_cover_percentage'] * (50 - df['temp_mean_C']) / 100
        
        logger.info("✓ Created 4 interaction features:")
        logger.info("  - Heat Stress Index (temp × humidity)")
        logger.info("  - Wind Dispersion Effect (wind / pollution)")
        logger.info("  - Emission Pressure (density × vehicles)")
        logger.info("  - Green Cooling Effect (green cover × temp)")
        
        self.citywide_df = df
        return df
    
    def create_cyclical_features(self):
        """Create cyclical features for temporal patterns"""
        logger.info("\n" + "=" * 70)
        logger.info("CREATING CYCLICAL FEATURES")
        logger.info("=" * 70)
        
        df = self.citywide_df.copy()
        
        # Create day_of_year if not present
        if 'day_of_year' not in df.columns:
            df['day_of_year'] = df['date'].dt.dayofyear
        
        # Convert day_of_year to sine/cosine for cyclical encoding
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Convert month to sine/cosine
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        logger.info("✓ Created 4 cyclical features:")
        logger.info("  - Day of year (sin/cos)")
        logger.info("  - Month (sin/cos)")
        
        self.citywide_df = df
        return df
    
    def calculate_ecosystem_health_score(self):
        """Calculate comprehensive Ecosystem Health Score (EHS)"""
        logger.info("\n" + "=" * 70)
        logger.info("CALCULATING ECOSYSTEM HEALTH SCORE")
        logger.info("=" * 70)
        
        df = self.citywide_df.copy()
        
        # Normalize each component to 0-100 scale (higher is better)
        
        # 1. Air Quality Score (invert - lower pollution is better)
        aqi_score = 100 - (df['AQI'] / 500 * 100).clip(0, 100)
        
        # 2. Weather Quality Score
        weather_score = df['Weather_Comfort_Index']
        
        # 3. Green Cover Score
        green_score = df['green_cover_percentage'] / 30 * 100  # Assume 30% is ideal
        green_score = green_score.clip(0, 100)
        
        # 4. Biodiversity Score
        biodiversity_score = (df['total_species'] - 300) / 100 * 100  # 300 baseline, 400 ideal
        biodiversity_score = biodiversity_score.clip(0, 100)
        
        # 5. Urban Pressure Score (inverse - less is better)
        urban_pressure = (df['density_per_sq_km'] / 20000) * 100  # 20k per sq km as reference
        urban_pressure_score = 100 - urban_pressure.clip(0, 100)
        
        # Weighted Ecosystem Health Score
        df['Ecosystem_Health_Score'] = (
            aqi_score * 0.35 +           # Air quality: 35%
            weather_score * 0.10 +        # Weather: 10%
            green_score * 0.25 +          # Green cover: 25%
            biodiversity_score * 0.20 +   # Biodiversity: 20%
            urban_pressure_score * 0.10   # Urban pressure: 10%
        )
        
        # Categorize EHS
        df['EHS_Category'] = pd.cut(
            df['Ecosystem_Health_Score'],
            bins=[0, 25, 40, 55, 70, 100],
            labels=['Critical', 'Poor', 'Moderate', 'Good', 'Excellent']
        )
        
        logger.info("✓ Ecosystem Health Score calculated:")
        logger.info(f"  Average EHS: {df['Ecosystem_Health_Score'].mean():.1f}")
        logger.info(f"  Min EHS: {df['Ecosystem_Health_Score'].min():.1f}")
        logger.info(f"  Max EHS: {df['Ecosystem_Health_Score'].max():.1f}")
        logger.info(f"\n  Category distribution:")
        for cat, count in df['EHS_Category'].value_counts().items():
            logger.info(f"    {cat}: {count} ({count/len(df)*100:.1f}%)")
        
        self.citywide_df = df
        return df
    
    def save_feature_dataset(self):
        """Save final feature-engineered dataset"""
        logger.info("\n" + "=" * 70)
        logger.info("SAVING FEATURE DATASET")
        logger.info("=" * 70)
        
        df = self.citywide_df.copy()
        
        # Drop rows with missing values in key features (from lag features)
        key_features = ['PM2.5', 'PM10', 'AQI', 'temp_mean_C']
        df_clean = df.dropna(subset=key_features)
        
        logger.info(f"Rows before cleaning: {len(df):,}")
        logger.info(f"Rows after cleaning: {len(df_clean):,}")
        logger.info(f"Total features: {len(df_clean.columns)}")
        
        # Save as CSV
        output_csv = DATA_FEATURES_DIR / 'master_dataset.csv'
        df_clean.to_csv(output_csv, index=False)
        logger.info(f"✓ Saved CSV: {output_csv} ({output_csv.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Save as Parquet (more efficient)
        output_parquet = DATA_FEATURES_DIR / 'master_dataset.parquet'
        df_clean.to_parquet(output_parquet, index=False)
        logger.info(f"✓ Saved Parquet: {output_parquet} ({output_parquet.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Save feature list
        feature_list_file = DATA_FEATURES_DIR / 'feature_list.txt'
        with open(feature_list_file, 'w') as f:
            f.write("FEATURE LIST\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total Features: {len(df_clean.columns)}\n")
            f.write(f"Total Records: {len(df_clean):,}\n")
            f.write(f"Date Range: {df_clean['date'].min()} to {df_clean['date'].max()}\n\n")
            
            # Group features by category
            categories = {
                'Target Variables': ['PM2.5', 'PM10', 'AQI', 'Ecosystem_Health_Score'],
                'Weather Features': [col for col in df_clean.columns if 'temp' in col or 'precipitation' in col or 'humidity' in col or 'wind' in col],
                'Lag Features': [col for col in df_clean.columns if 'lag_' in col],
                'Rolling Features': [col for col in df_clean.columns if 'rolling_' in col],
                'Trend Features': [col for col in df_clean.columns if 'trend' in col or 'change_' in col],
                'Composite Indices': [col for col in df_clean.columns if 'Index' in col or 'Score' in col],
                'Annual Features': [col for col in df_clean.columns if any(x in col for x in ['forest', 'green', 'species', 'population', 'vehicles'])],
                'Temporal Features': ['year', 'month', 'day_of_year', 'day_of_week', 'quarter', 'season', 'is_weekend'],
                'Cyclical Features': [col for col in df_clean.columns if 'sin' in col or 'cos' in col]
            }
            
            for category, features in categories.items():
                features = [f for f in features if f in df_clean.columns]
                if features:
                    f.write(f"\n{category} ({len(features)}):\n")
                    f.write("-" * 70 + "\n")
                    for feature in sorted(features):
                        f.write(f"  - {feature}\n")
        
        logger.info(f"✓ Saved feature list: {feature_list_file}")
        
        return df_clean

def main():
    """Main feature engineering pipeline"""
    logger.info("Starting feature engineering pipeline...")
    
    engineer = FeatureEngineer()
    
    try:
        # Load data
        engineer.load_processed_data()
        
        # Create features
        engineer.create_lag_features()
        engineer.create_rolling_features()
        engineer.create_trend_features()
        engineer.create_composite_indices()
        engineer.merge_annual_features()
        engineer.create_interaction_features()
        engineer.create_cyclical_features()
        engineer.calculate_ecosystem_health_score()
        
        # Save final dataset
        final_df = engineer.save_feature_dataset()
        
        logger.info("\n" + "=" * 70)
        logger.info("FEATURE ENGINEERING COMPLETE ✓")
        logger.info("=" * 70)
        logger.info(f"\nFinal dataset:")
        logger.info(f"  Records: {len(final_df):,}")
        logger.info(f"  Features: {len(final_df.columns)}")
        logger.info(f"  Date range: {final_df['date'].min()} to {final_df['date'].max()}")
        logger.info(f"\nReady for model training!")
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
