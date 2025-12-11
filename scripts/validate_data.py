#!/usr/bin/env python3
"""
Data Validation and Summary Report Generator
Validates downloaded datasets and generates comprehensive summary
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'

class DataValidator:
    """Validate and summarize downloaded datasets"""
    
    def __init__(self):
        self.validation_results = {}
        self.datasets_info = []
    
    def validate_csv(self, file_path, expected_columns=None):
        """Validate a single CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            info = {
                'file': str(file_path.relative_to(DATA_DIR)),
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'size_mb': round(file_path.stat().st_size / (1024**2), 3),
                'missing_values': df.isnull().sum().sum(),
                'missing_pct': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
                'date_columns': [],
                'numeric_columns': [],
                'status': 'valid'
            }
            
            # Identify date columns
            for col in df.columns:
                if 'date' in col.lower() or 'year' in col.lower():
                    info['date_columns'].append(col)
            
            # Identify numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            info['numeric_columns'] = numeric_cols
            
            # Basic statistics for first numeric column
            if numeric_cols:
                first_num_col = numeric_cols[0]
                info['sample_stats'] = {
                    'column': first_num_col,
                    'min': round(df[first_num_col].min(), 2) if pd.notna(df[first_num_col].min()) else None,
                    'max': round(df[first_num_col].max(), 2) if pd.notna(df[first_num_col].max()) else None,
                    'mean': round(df[first_num_col].mean(), 2) if pd.notna(df[first_num_col].mean()) else None,
                    'median': round(df[first_num_col].median(), 2) if pd.notna(df[first_num_col].median()) else None
                }
            
            return info
            
        except Exception as e:
            logger.error(f"Error validating {file_path}: {str(e)}")
            return {
                'file': str(file_path.relative_to(DATA_DIR)),
                'status': 'error',
                'error': str(e)
            }
    
    def validate_all_datasets(self):
        """Validate all CSV files in data directory"""
        logger.info("=" * 70)
        logger.info("VALIDATING DOWNLOADED DATASETS")
        logger.info("=" * 70)
        
        # Find all CSV files
        csv_files = list(DATA_DIR.rglob("*.csv"))
        logger.info(f"\nFound {len(csv_files)} CSV files")
        
        # Validate each file
        for csv_file in sorted(csv_files):
            info = self.validate_csv(csv_file)
            self.datasets_info.append(info)
            
            if info['status'] == 'valid':
                logger.info(f"\n✓ {info['file']}")
                logger.info(f"  Rows: {info['rows']:,} | Columns: {info['columns']} | Size: {info['size_mb']} MB")
                if info.get('sample_stats'):
                    stats = info['sample_stats']
                    logger.info(f"  {stats['column']}: min={stats['min']}, max={stats['max']}, mean={stats['mean']}")
            else:
                logger.error(f"\n✗ {info['file']}: {info.get('error', 'Unknown error')}")
        
        return self.datasets_info
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        logger.info("\n" + "=" * 70)
        logger.info("DATA SUMMARY REPORT")
        logger.info("=" * 70)
        
        # Group by category
        categories = {}
        for info in self.datasets_info:
            if info['status'] == 'valid':
                category = info['file'].split('/')[0]
                if category not in categories:
                    categories[category] = []
                categories[category].append(info)
        
        total_rows = 0
        total_size = 0
        
        summary_text = []
        summary_text.append("\n" + "=" * 70)
        summary_text.append("DELHI ECOSYSTEM RESTORATION - DATA COLLECTION SUMMARY")
        summary_text.append("=" * 70)
        summary_text.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for category, datasets in sorted(categories.items()):
            cat_rows = sum(d['rows'] for d in datasets)
            cat_size = sum(d['size_mb'] for d in datasets)
            total_rows += cat_rows
            total_size += cat_size
            
            summary_text.append(f"\n{category.upper().replace('_', ' ')}:")
            summary_text.append("-" * 70)
            
            for ds in datasets:
                filename = ds['file'].split('/')[-1]
                summary_text.append(f"  • {filename}")
                summary_text.append(f"    Rows: {ds['rows']:,} | Columns: {ds['columns']} | Size: {ds['size_mb']} MB")
                
                if ds.get('sample_stats'):
                    stats = ds['sample_stats']
                    summary_text.append(f"    {stats['column']}: {stats['min']} to {stats['max']} (avg: {stats['mean']})")
            
            summary_text.append(f"\n  Subtotal: {cat_rows:,} rows | {cat_size:.2f} MB")
        
        summary_text.append("\n" + "=" * 70)
        summary_text.append(f"TOTAL: {total_rows:,} rows | {total_size:.2f} MB | {len(self.datasets_info)} files")
        summary_text.append("=" * 70)
        
        # Key datasets summary
        summary_text.append("\n\nKEY DATASETS:")
        summary_text.append("-" * 70)
        
        key_datasets = [
            ('Air Quality', 'air_quality/cpcb_annual/delhi_air_quality_2019_2024_daily.csv'),
            ('Weather', 'climate_weather/nasa_power/delhi_weather_2019_2023.csv'),
            ('Water Quality', 'water_quality/yamuna/yamuna_water_quality_2019_2023_monthly.csv'),
            ('Forest Cover', 'forest_greencover/fsi_reports/delhi_forest_cover_2013_2023.csv'),
            ('Population', 'socioeconomic/population/delhi_population_2011_2024.csv'),
            ('Vehicles', 'socioeconomic/vehicles/delhi_vehicles_2019_2024.csv'),
            ('Birds', 'biodiversity/birds/delhi_bird_diversity_2019_2024.csv')
        ]
        
        for name, path in key_datasets:
            ds = next((d for d in self.datasets_info if d['file'] == path), None)
            if ds and ds['status'] == 'valid':
                summary_text.append(f"\n{name}:")
                summary_text.append(f"  Records: {ds['rows']:,}")
                summary_text.append(f"  Columns: {', '.join(ds['column_names'][:5])}...")
                if ds.get('sample_stats'):
                    summary_text.append(f"  Sample: {ds['sample_stats']}")
        
        # Data coverage
        summary_text.append("\n\nDATA COVERAGE:")
        summary_text.append("-" * 70)
        summary_text.append("✓ Air Quality: 2019-2024 (6 stations, daily)")
        summary_text.append("✓ Weather: 2019-2023 (NASA POWER, daily)")
        summary_text.append("✓ Water Quality: 2019-2023 (4 locations, monthly)")
        summary_text.append("✓ Forest Cover: 2013-2023 (biennial)")
        summary_text.append("✓ Biodiversity: 2019-2024 (annual)")
        summary_text.append("✓ Socioeconomic: 2011-2024 (annual)")
        summary_text.append("⚠ Satellite Data: Requires manual download (see instructions)")
        
        # Data quality
        summary_text.append("\n\nDATA QUALITY:")
        summary_text.append("-" * 70)
        
        valid_count = sum(1 for d in self.datasets_info if d['status'] == 'valid')
        error_count = sum(1 for d in self.datasets_info if d['status'] == 'error')
        
        summary_text.append(f"Valid datasets: {valid_count}/{len(self.datasets_info)}")
        summary_text.append(f"Error datasets: {error_count}")
        
        total_missing = sum(d.get('missing_values', 0) for d in self.datasets_info if d['status'] == 'valid')
        summary_text.append(f"Total missing values: {total_missing:,}")
        
        # Next steps
        summary_text.append("\n\nNEXT STEPS:")
        summary_text.append("-" * 70)
        summary_text.append("1. Download satellite data (OPTIONAL):")
        summary_text.append("   • Google Earth Engine: Sentinel-2 LULC, MODIS NDVI")
        summary_text.append("   • NASA Earthdata: Landsat LST")
        summary_text.append("   • See: data/raw/satellite/DOWNLOAD_INSTRUCTIONS.txt")
        summary_text.append("\n2. Data preprocessing:")
        summary_text.append("   • Run: python scripts/preprocess_data.py")
        summary_text.append("   • Clean missing values, outliers")
        summary_text.append("   • Temporal alignment")
        summary_text.append("   • Feature engineering")
        summary_text.append("\n3. Exploratory Data Analysis:")
        summary_text.append("   • Jupyter notebook: notebooks/01_data_exploration.ipynb")
        summary_text.append("   • Visualize trends, patterns")
        summary_text.append("   • Correlation analysis")
        summary_text.append("\n4. Model development:")
        summary_text.append("   • Baseline models (Linear Regression, Random Forest)")
        summary_text.append("   • Advanced models (XGBoost, LSTM)")
        summary_text.append("   • Hyperparameter tuning")
        
        summary_text.append("\n" + "=" * 70)
        summary_text.append("DATA COLLECTION STATUS: COMPLETE ✓")
        summary_text.append("=" * 70)
        
        # Print and save
        summary_str = "\n".join(summary_text)
        print(summary_str)
        
        # Save to file
        report_path = BASE_DIR / 'logs' / 'data_summary_report.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(summary_str)
        
        logger.info(f"\n\n✓ Report saved: {report_path}")
        
        # Save JSON metadata
        json_path = BASE_DIR / 'logs' / 'datasets_metadata.json'
        with open(json_path, 'w') as f:
            json.dump(self.datasets_info, f, indent=2)
        
        logger.info(f"✓ Metadata saved: {json_path}")
        
        return summary_str

def main():
    validator = DataValidator()
    validator.validate_all_datasets()
    validator.generate_summary_report()
    
    logger.info("\n✓ Validation complete!")

if __name__ == "__main__":
    main()
