#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for Delhi Ecosystem Restoration ML Project
Handles cleaning, temporal alignment, and feature preparation
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
        logging.FileHandler('logs/preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

class DataPreprocessor:
    """Preprocess and clean all datasets"""
    
    def __init__(self):
        self.air_quality_df = None
        self.weather_df = None
        self.water_quality_df = None
        self.forest_cover_df = None
        self.biodiversity_df = None
        self.population_df = None
        self.vehicles_df = None
        
    def load_all_data(self):
        """Load all datasets"""
        logger.info("=" * 70)
        logger.info("LOADING ALL DATASETS")
        logger.info("=" * 70)
        
        # 1. Air Quality
        air_file = DATA_RAW_DIR / 'air_quality' / 'cpcb_annual' / 'delhi_air_quality_2019_2024_daily.csv'
        self.air_quality_df = pd.read_csv(air_file)
        self.air_quality_df['date'] = pd.to_datetime(self.air_quality_df['date'])
        logger.info(f"✓ Air Quality: {len(self.air_quality_df):,} records")
        
        # 2. Weather
        weather_file = DATA_RAW_DIR / 'climate_weather' / 'nasa_power' / 'delhi_weather_2019_2023.csv'
        self.weather_df = pd.read_csv(weather_file)
        self.weather_df['date'] = pd.to_datetime(self.weather_df['date'])
        logger.info(f"✓ Weather: {len(self.weather_df):,} records")
        
        # 3. Water Quality
        water_file = DATA_RAW_DIR / 'water_quality' / 'yamuna' / 'yamuna_water_quality_2019_2023_monthly.csv'
        self.water_quality_df = pd.read_csv(water_file)
        self.water_quality_df['date'] = pd.to_datetime(self.water_quality_df['date'] + '-01')  # Convert YYYY-MM to date
        logger.info(f"✓ Water Quality: {len(self.water_quality_df):,} records")
        
        # 4. Forest Cover
        forest_file = DATA_RAW_DIR / 'forest_greencover' / 'fsi_reports' / 'delhi_forest_cover_2013_2023.csv'
        self.forest_cover_df = pd.read_csv(forest_file)
        logger.info(f"✓ Forest Cover: {len(self.forest_cover_df):,} records")
        
        # 5. Biodiversity
        bird_file = DATA_RAW_DIR / 'biodiversity' / 'birds' / 'delhi_bird_diversity_2019_2024.csv'
        self.biodiversity_df = pd.read_csv(bird_file)
        logger.info(f"✓ Biodiversity: {len(self.biodiversity_df):,} records")
        
        # 6. Population
        pop_file = DATA_RAW_DIR / 'socioeconomic' / 'population' / 'delhi_population_2011_2024.csv'
        self.population_df = pd.read_csv(pop_file)
        logger.info(f"✓ Population: {len(self.population_df):,} records")
        
        # 7. Vehicles
        vehicle_file = DATA_RAW_DIR / 'socioeconomic' / 'vehicles' / 'delhi_vehicles_2019_2024.csv'
        self.vehicles_df = pd.read_csv(vehicle_file)
        logger.info(f"✓ Vehicles: {len(self.vehicles_df):,} records")
        
        return True
    
    def clean_air_quality(self):
        """Clean and process air quality data"""
        logger.info("\n" + "=" * 70)
        logger.info("CLEANING AIR QUALITY DATA")
        logger.info("=" * 70)
        
        df = self.air_quality_df.copy()
        
        # Check missing values
        missing = df.isnull().sum()
        logger.info(f"Missing values before cleaning:\n{missing[missing > 0]}")
        
        # Remove outliers (Z-score method, keep values within 4 std)
        numeric_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        
        for col in numeric_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers_before = (z_scores > 4).sum()
            # Cap outliers at 4 std instead of removing
            df[col] = df[col].clip(
                lower=df[col].mean() - 4*df[col].std(),
                upper=df[col].mean() + 4*df[col].std()
            )
            if outliers_before > 0:
                logger.info(f"  {col}: Capped {outliers_before} outliers")
        
        # Add temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_week'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        
        # Add season
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Summer', 4: 'Summer', 5: 'Summer', 6: 'Summer',
            7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
            10: 'Autumn', 11: 'Autumn'
        })
        
        # Add weekend flag
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        logger.info(f"✓ Cleaned air quality data: {len(df):,} records")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  Stations: {df['station'].nunique()}")
        logger.info(f"  Added {len(df.columns) - len(self.air_quality_df.columns)} temporal features")
        
        # Save cleaned data
        output_file = DATA_PROCESSED_DIR / 'air_quality_cleaned.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"✓ Saved: {output_file}")
        
        self.air_quality_df = df
        return df
    
    def clean_weather(self):
        """Clean and process weather data"""
        logger.info("\n" + "=" * 70)
        logger.info("CLEANING WEATHER DATA")
        logger.info("=" * 70)
        
        df = self.weather_df.copy()
        
        # Check missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.info(f"Missing values: {missing.sum()}")
            # Forward fill for short gaps
            df = df.fillna(method='ffill', limit=3)
        
        # Add temporal features (same as air quality for merging)
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Add season
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Summer', 4: 'Summer', 5: 'Summer', 6: 'Summer',
            7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
            10: 'Autumn', 11: 'Autumn'
        })
        
        # Create derived features
        df['temp_range'] = df['temp_max_C'] - df['temp_min_C']
        df['is_rainy'] = (df['precipitation_mm'] > 1).astype(int)
        
        # Categorize weather
        df['weather_category'] = pd.cut(
            df['temp_mean_C'],
            bins=[0, 15, 25, 35, 50],
            labels=['Cold', 'Pleasant', 'Warm', 'Hot']
        )
        
        logger.info(f"✓ Cleaned weather data: {len(df):,} records")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  Temp range: {df['temp_mean_C'].min():.1f}°C to {df['temp_mean_C'].max():.1f}°C")
        logger.info(f"  Rainy days: {df['is_rainy'].sum()} ({df['is_rainy'].mean()*100:.1f}%)")
        
        # Save cleaned data
        output_file = DATA_PROCESSED_DIR / 'weather_cleaned.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"✓ Saved: {output_file}")
        
        self.weather_df = df
        return df
    
    def merge_daily_data(self):
        """Merge air quality and weather data (both daily)"""
        logger.info("\n" + "=" * 70)
        logger.info("MERGING DAILY DATASETS")
        logger.info("=" * 70)
        
        # Merge air quality with weather by date
        df_merged = self.air_quality_df.merge(
            self.weather_df,
            on='date',
            how='left',
            suffixes=('', '_weather')
        )
        
        # Drop duplicate temporal columns from weather
        cols_to_drop = [col for col in df_merged.columns if col.endswith('_weather')]
        df_merged = df_merged.drop(columns=cols_to_drop)
        
        logger.info(f"✓ Merged daily data: {len(df_merged):,} records")
        logger.info(f"  Columns: {len(df_merged.columns)}")
        logger.info(f"  Missing values: {df_merged.isnull().sum().sum()}")
        
        # Save merged daily data
        output_file = DATA_PROCESSED_DIR / 'daily_merged.csv'
        df_merged.to_csv(output_file, index=False)
        logger.info(f"✓ Saved: {output_file}")
        
        return df_merged
    
    def prepare_annual_data(self):
        """Prepare annual datasets (forest, biodiversity, socioeconomic)"""
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING ANNUAL DATA")
        logger.info("=" * 70)
        
        # Forest cover (interpolate for missing years)
        forest_expanded = []
        for year in range(2013, 2025):
            if year in self.forest_cover_df['year'].values:
                row_data = self.forest_cover_df[self.forest_cover_df['year'] == year].iloc[0].to_dict()
                forest_expanded.append(row_data)
            else:
                # Linear interpolation
                prev_year = self.forest_cover_df[self.forest_cover_df['year'] < year]['year'].max()
                next_year = self.forest_cover_df[self.forest_cover_df['year'] > year]['year'].min()
                
                if pd.notna(prev_year) and pd.notna(next_year):
                    prev_data = self.forest_cover_df[self.forest_cover_df['year'] == prev_year].iloc[0]
                    next_data = self.forest_cover_df[self.forest_cover_df['year'] == next_year].iloc[0]
                    
                    weight = (year - prev_year) / (next_year - prev_year)
                    
                    interpolated = {
                        'year': year,
                        'forest_cover_sq_km': float(prev_data['forest_cover_sq_km'] + 
                                             weight * (next_data['forest_cover_sq_km'] - prev_data['forest_cover_sq_km'])),
                        'total_green_cover_sq_km': float(prev_data['total_green_cover_sq_km'] + 
                                                   weight * (next_data['total_green_cover_sq_km'] - prev_data['total_green_cover_sq_km'])),
                        'green_cover_percentage': float(prev_data['green_cover_percentage'] + 
                                                 weight * (next_data['green_cover_percentage'] - prev_data['green_cover_percentage']))
                    }
                    forest_expanded.append(interpolated)
        
        forest_df = pd.DataFrame(forest_expanded)
        logger.info(f"✓ Forest cover interpolated: {len(forest_df)} years (2013-2024)")
        
        # Merge annual data
        annual_df = forest_df[['year', 'forest_cover_sq_km', 'total_green_cover_sq_km', 'green_cover_percentage']]
        
        # Add biodiversity
        annual_df = annual_df.merge(
            self.biodiversity_df[['year', 'total_species', 'resident_species', 'migratory_species', 'threatened_species']],
            on='year',
            how='left'
        )
        
        # Add population
        annual_df = annual_df.merge(
            self.population_df[['year', 'population', 'density_per_sq_km', 'literacy_rate']],
            on='year',
            how='left'
        )
        
        # Add vehicles
        annual_df = annual_df.merge(
            self.vehicles_df[['year', 'total_vehicles', 'two_wheelers', 'cars', 'electric_vehicles']],
            on='year',
            how='left'
        )
        
        # Forward fill missing values for years
        annual_df = annual_df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"✓ Annual data merged: {len(annual_df)} years")
        logger.info(f"  Columns: {len(annual_df.columns)}")
        
        # Save annual data
        output_file = DATA_PROCESSED_DIR / 'annual_merged.csv'
        annual_df.to_csv(output_file, index=False)
        logger.info(f"✓ Saved: {output_file}")
        
        return annual_df
    
    def create_station_aggregates(self):
        """Create aggregated air quality by station"""
        logger.info("\n" + "=" * 70)
        logger.info("CREATING STATION AGGREGATES")
        logger.info("=" * 70)
        
        df_daily = pd.read_csv(DATA_PROCESSED_DIR / 'daily_merged.csv')
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        
        # Monthly aggregates by station
        df_daily['year_month'] = df_daily['date'].dt.to_period('M')
        
        monthly_station = df_daily.groupby(['year_month', 'station']).agg({
            'PM2.5': ['mean', 'max', 'min', 'std'],
            'PM10': ['mean', 'max', 'min'],
            'NO2': 'mean',
            'SO2': 'mean',
            'CO': 'mean',
            'O3': 'mean',
            'AQI': ['mean', 'max'],
            'temp_mean_C': 'mean',
            'precipitation_mm': 'sum',
            'humidity_percent': 'mean'
        }).reset_index()
        
        # Flatten column names
        monthly_station.columns = ['_'.join(col).strip('_') for col in monthly_station.columns.values]
        monthly_station['date'] = monthly_station['year_month'].dt.to_timestamp()
        monthly_station = monthly_station.drop('year_month', axis=1)
        
        logger.info(f"✓ Monthly station aggregates: {len(monthly_station):,} records")
        logger.info(f"  Stations: {monthly_station['station'].nunique()}")
        logger.info(f"  Months: {len(monthly_station) / monthly_station['station'].nunique():.0f}")
        
        # Save
        output_file = DATA_PROCESSED_DIR / 'monthly_station_aggregates.csv'
        monthly_station.to_csv(output_file, index=False)
        logger.info(f"✓ Saved: {output_file}")
        
        # City-wide daily average (across all stations)
        citywide_daily = df_daily.groupby('date').agg({
            'PM2.5': 'mean',
            'PM10': 'mean',
            'NO2': 'mean',
            'SO2': 'mean',
            'CO': 'mean',
            'O3': 'mean',
            'AQI': 'mean',
            'temp_mean_C': 'mean',
            'temp_min_C': 'mean',
            'temp_max_C': 'mean',
            'precipitation_mm': 'mean',
            'humidity_percent': 'mean',
            'wind_speed_ms': 'mean',
            'year': 'first',
            'month': 'first',
            'season': 'first',
            'is_weekend': 'first'
        }).reset_index()
        
        logger.info(f"✓ City-wide daily average: {len(citywide_daily):,} records")
        
        # Save
        output_file = DATA_PROCESSED_DIR / 'citywide_daily_average.csv'
        citywide_daily.to_csv(output_file, index=False)
        logger.info(f"✓ Saved: {output_file}")
        
        return monthly_station, citywide_daily
    
    def generate_preprocessing_report(self):
        """Generate comprehensive preprocessing report"""
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING PREPROCESSING REPORT")
        logger.info("=" * 70)
        
        report = []
        report.append("=" * 70)
        report.append("DATA PREPROCESSING REPORT")
        report.append("=" * 70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # List all processed files
        report.append("\nPROCESSED FILES:")
        report.append("-" * 70)
        
        processed_files = list(DATA_PROCESSED_DIR.glob("*.csv"))
        for file in sorted(processed_files):
            df = pd.read_csv(file)
            report.append(f"\n{file.name}")
            report.append(f"  Records: {len(df):,}")
            report.append(f"  Columns: {len(df.columns)}")
            report.append(f"  Size: {file.stat().st_size / 1024:.1f} KB")
            report.append(f"  Missing: {df.isnull().sum().sum()}")
        
        report.append("\n\nDATA QUALITY:")
        report.append("-" * 70)
        
        # Check city-wide daily
        citywide = pd.read_csv(DATA_PROCESSED_DIR / 'citywide_daily_average.csv')
        report.append(f"\nCity-wide Daily Average:")
        report.append(f"  Date range: {citywide['date'].min()} to {citywide['date'].max()}")
        report.append(f"  Days: {len(citywide):,}")
        report.append(f"  Avg PM2.5: {citywide['PM2.5'].mean():.1f} µg/m³")
        report.append(f"  Avg AQI: {citywide['AQI'].mean():.0f}")
        report.append(f"  Missing values: {citywide.isnull().sum().sum()}")
        
        report.append("\n\nNEXT STEPS:")
        report.append("-" * 70)
        report.append("1. Feature Engineering:")
        report.append("   - Create lag features (t-1, t-7, t-30)")
        report.append("   - Rolling statistics (7-day, 30-day averages)")
        report.append("   - Composite indices (Ecosystem Health Score)")
        report.append("\n2. Exploratory Data Analysis:")
        report.append("   - Time series visualization")
        report.append("   - Correlation analysis")
        report.append("   - Seasonal decomposition")
        report.append("\n3. Model Training:")
        report.append("   - Baseline: Linear Regression, Random Forest")
        report.append("   - Advanced: XGBoost, LightGBM, LSTM")
        
        report.append("\n" + "=" * 70)
        report.append("PREPROCESSING COMPLETE ✓")
        report.append("=" * 70)
        
        report_str = "\n".join(report)
        print("\n" + report_str)
        
        # Save report
        report_file = BASE_DIR / 'logs' / 'preprocessing_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_str)
        
        logger.info(f"\n✓ Report saved: {report_file}")
        
        return report_str

def main():
    """Main preprocessing pipeline"""
    logger.info("Starting data preprocessing pipeline...")
    
    preprocessor = DataPreprocessor()
    
    try:
        # Step 1: Load all data
        preprocessor.load_all_data()
        
        # Step 2: Clean individual datasets
        preprocessor.clean_air_quality()
        preprocessor.clean_weather()
        
        # Step 3: Merge daily data
        preprocessor.merge_daily_data()
        
        # Step 4: Prepare annual data
        preprocessor.prepare_annual_data()
        
        # Step 5: Create aggregates
        preprocessor.create_station_aggregates()
        
        # Step 6: Generate report
        preprocessor.generate_preprocessing_report()
        
        logger.info("\n✓ Preprocessing pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
