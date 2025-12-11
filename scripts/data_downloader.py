#!/usr/bin/env python3
"""
Data Downloader for Delhi Ecosystem Restoration Project
Downloads data from various government and public sources
"""

import os
import sys
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import time
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'

class DataDownloader:
    """Main class for downloading environmental data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        
    def download_file(self, url, output_path, description=""):
        """Download file with progress bar"""
        try:
            logger.info(f"Downloading {description}: {url}")
            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=description
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Successfully downloaded: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return False
    
    def download_data_gov_in(self, resource_id, output_path, api_key=None):
        """Download dataset from data.gov.in API"""
        try:
            # data.gov.in CKAN API endpoint
            base_url = "https://api.data.gov.in/resource"
            
            params = {
                'resource_id': resource_id,
                'format': 'json',
                'limit': 10000
            }
            
            if api_key:
                params['api-key'] = api_key
            
            logger.info(f"Fetching data.gov.in resource: {resource_id}")
            response = self.session.get(base_url, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Save raw JSON
                with open(output_path.with_suffix('.json'), 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Convert to CSV if records exist
                if 'records' in data and len(data['records']) > 0:
                    df = pd.DataFrame(data['records'])
                    df.to_csv(output_path, index=False)
                    logger.info(f"Saved {len(df)} records to {output_path}")
                    return True
                else:
                    logger.warning(f"No records found in resource {resource_id}")
                    return False
            else:
                logger.error(f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading from data.gov.in: {str(e)}")
            return False
    
    def download_air_quality_data(self):
        """Download air quality datasets"""
        logger.info("=" * 60)
        logger.info("DOWNLOADING AIR QUALITY DATA")
        logger.info("=" * 60)
        
        air_quality_dir = DATA_DIR / 'air_quality' / 'data_gov_in'
        air_quality_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset 1: Category-wise Air Quality Index - Delhi
        datasets = [
            {
                'name': 'Delhi_AQI_Category_2019_2021',
                'url': 'https://www.data.gov.in/resource/category-wise-air-quality-index-major-metropolitan-city-delhi-comparative-air-quality',
                'resource_id': None,  # Will need actual resource ID
                'description': 'Category-wise AQI for Delhi 2019-2021'
            }
        ]
        
        # Note: Direct CSV download links from data.gov.in (if available)
        # These need to be updated with actual download links
        
        # Try to fetch CPCB real-time data (web scraping alternative)
        logger.info("Attempting to fetch CPCB data...")
        self.fetch_cpcb_realtime_data()
        
        return True
    
    def fetch_cpcb_realtime_data(self):
        """Fetch CPCB real-time air quality data"""
        try:
            # CPCB CCR API endpoint (may require authentication)
            # This is a placeholder - actual API endpoints may vary
            
            cpcb_dir = DATA_DIR / 'air_quality' / 'cpcb_realtime'
            cpcb_dir.mkdir(parents=True, exist_ok=True)
            
            # Delhi stations
            delhi_stations = [
                'Anand Vihar', 'RK Puram', 'Punjabi Bagh', 'Dwarka',
                'ITO', 'Lodhi Road', 'Major Dhyan Chand Stadium'
            ]
            
            # For demonstration, create a template CSV
            logger.info("Note: CPCB real-time data requires API access or web scraping")
            logger.info("Creating data collection template...")
            
            template_data = {
                'timestamp': [datetime.now() - timedelta(days=i) for i in range(7)],
                'station': ['Anand Vihar'] * 7,
                'PM2.5': [150, 160, 145, 155, 170, 165, 158],
                'PM10': [200, 210, 195, 205, 220, 215, 208],
                'NO2': [45, 48, 43, 46, 50, 47, 45],
                'SO2': [12, 13, 11, 12, 14, 13, 12],
                'CO': [1.2, 1.3, 1.1, 1.2, 1.4, 1.3, 1.2],
                'O3': [35, 37, 33, 36, 38, 36, 35],
                'AQI': [200, 210, 195, 205, 220, 215, 208]
            }
            
            df_template = pd.DataFrame(template_data)
            template_path = cpcb_dir / 'delhi_aqi_template.csv'
            df_template.to_csv(template_path, index=False)
            
            logger.info(f"Template saved: {template_path}")
            logger.info("Manual steps required:")
            logger.info("1. Visit: https://cpcb.nic.in/real-time-data/")
            logger.info("2. Download historical data for Delhi stations")
            logger.info("3. Place CSV files in: data/raw/air_quality/cpcb_realtime/")
            
            return True
            
        except Exception as e:
            logger.error(f"Error fetching CPCB data: {str(e)}")
            return False
    
    def download_water_quality_data(self):
        """Download Yamuna River water quality data"""
        logger.info("=" * 60)
        logger.info("DOWNLOADING WATER QUALITY DATA")
        logger.info("=" * 60)
        
        water_dir = DATA_DIR / 'water_quality' / 'yamuna'
        water_dir.mkdir(parents=True, exist_ok=True)
        
        # Create template for Yamuna water quality
        template_data = {
            'date': pd.date_range('2019-01-01', '2023-12-31', freq='M'),
            'location': ['Palla'] * 60,
            'BOD_mg_L': [3.5, 3.8, 3.2, 3.6, 4.0] * 12,
            'COD_mg_L': [20, 22, 18, 21, 24] * 12,
            'DO_mg_L': [6.5, 6.2, 6.8, 6.4, 6.0] * 12,
            'pH': [7.8, 7.9, 7.7, 7.8, 7.9] * 12,
            'Fecal_Coliform_MPN': [500, 550, 480, 520, 580] * 12
        }
        
        df_water = pd.DataFrame(template_data)
        water_template = water_dir / 'yamuna_water_quality_template.csv'
        df_water.to_csv(water_template, index=False)
        
        logger.info(f"Template saved: {water_template}")
        logger.info("Manual steps required:")
        logger.info("1. Visit: https://www.data.gov.in/keywords/Yamuna")
        logger.info("2. Download 'Water Quality Data of River Yamuna (2019-2023)'")
        logger.info("3. Place CSV in: data/raw/water_quality/yamuna/")
        
        return True
    
    def download_forest_cover_data(self):
        """Download forest cover and green cover data"""
        logger.info("=" * 60)
        logger.info("DOWNLOADING FOREST & GREEN COVER DATA")
        logger.info("=" * 60)
        
        forest_dir = DATA_DIR / 'forest_greencover' / 'fsi_reports'
        forest_dir.mkdir(parents=True, exist_ok=True)
        
        # Create template
        template_data = {
            'year': [2019, 2021, 2023],
            'district': ['Delhi'] * 3,
            'forest_cover_sq_km': [176.14, 176.50, 177.00],
            'tree_cover_sq_km': [67.00, 68.50, 70.00],
            'total_green_cover_sq_km': [243.14, 245.00, 247.00],
            'green_cover_percentage': [16.4, 16.5, 16.7]
        }
        
        df_forest = pd.DataFrame(template_data)
        forest_template = forest_dir / 'delhi_forest_cover_template.csv'
        df_forest.to_csv(forest_template, index=False)
        
        logger.info(f"Template saved: {forest_template}")
        logger.info("Manual steps required:")
        logger.info("1. Visit: https://www.data.gov.in/resource/district-wise-forest-cover-delhi")
        logger.info("2. Visit: https://forest.delhi.gov.in/forest/extent-forest-and-tree-cover")
        logger.info("3. Download India State of Forest Report (ISFR) 2019, 2021, 2023")
        logger.info("4. Place data in: data/raw/forest_greencover/fsi_reports/")
        
        return True
    
    def download_climate_data_kaggle(self):
        """Download Delhi climate data from Kaggle"""
        logger.info("=" * 60)
        logger.info("DOWNLOADING CLIMATE DATA")
        logger.info("=" * 60)
        
        climate_dir = DATA_DIR / 'climate_weather' / 'kaggle'
        climate_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Kaggle dataset download requires Kaggle API setup")
        logger.info("\nSetup instructions:")
        logger.info("1. Install: pip install kaggle")
        logger.info("2. Create Kaggle account and get API token from https://www.kaggle.com/account")
        logger.info("3. Place kaggle.json in ~/.kaggle/")
        logger.info("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        logger.info("\nTo download:")
        logger.info("kaggle datasets download -d yug201/daily-climate-time-series-data-delhi-india")
        logger.info(f"Extract to: {climate_dir}")
        
        # Create sample template
        import numpy as np
        date_range = pd.date_range('2013-01-01', '2024-12-31', freq='D')
        template_data = {
            'date': date_range,
            'meantemp_C': [20 + 10*np.sin(2*np.pi*i/365) for i in range(len(date_range))],
            'humidity': [60 + 20*np.sin(2*np.pi*i/365) for i in range(len(date_range))],
            'wind_speed_kmph': [10 + 5*np.random.rand() for _ in range(len(date_range))],
            'meanpressure': [1013] * len(date_range)
        }
        
        # Sample only (too large for full template)
        df_climate_sample = pd.DataFrame(template_data).head(1000)
        climate_template = climate_dir / 'delhi_climate_sample_template.csv'
        df_climate_sample.to_csv(climate_template, index=False)
        
        logger.info(f"Sample template saved: {climate_template}")
        
        return True
    
    def download_satellite_data_info(self):
        """Provide instructions for satellite data download"""
        logger.info("=" * 60)
        logger.info("SATELLITE DATA DOWNLOAD INSTRUCTIONS")
        logger.info("=" * 60)
        
        satellite_dir = DATA_DIR / 'satellite'
        satellite_dir.mkdir(parents=True, exist_ok=True)
        
        instructions = """
SATELLITE DATA ACQUISITION GUIDE
=================================

1. GOOGLE EARTH ENGINE (Sentinel-2, MODIS)
   - Register: https://earthengine.google.com/
   - Install: pip install earthengine-api
   - Authenticate: earthengine authenticate
   - Use provided scripts in scripts/satellite_download.py

2. SENTINEL-2 (10m LULC)
   - Option A: Google Earth Engine (recommended)
   - Option B: Copernicus Open Access Hub: https://scihub.copernicus.eu/
   - Option C: AWS S3: aws s3 ls s3://sentinel-s2-l2a/tiles/43/R/GM/

3. MODIS NDVI
   - NASA Earthdata: https://earthdata.nasa.gov/
   - Register and use AppEEARS: https://appeears.earthdatacloud.nasa.gov/
   - Product: MOD13Q1 (250m, 16-day NDVI)
   - Tiles covering Delhi: h25v06

4. LANDSAT 8/9 (Land Surface Temperature)
   - USGS EarthExplorer: https://earthexplorer.usgs.gov/
   - Datasets: Landsat 8-9 Level-2, Collection 2
   - Path/Row: 146/40 (covers Delhi)
   - Download thermal bands (ST_B10)

Delhi Coordinates:
   - Latitude: 28.7041° N
   - Longitude: 77.1025° E
   - Bounding Box: [76.8, 28.4, 77.4, 28.9]

Recommended Time Period: 2019-2024
        """
        
        info_path = satellite_dir / 'DOWNLOAD_INSTRUCTIONS.txt'
        with open(info_path, 'w') as f:
            f.write(instructions)
        
        logger.info(f"Instructions saved: {info_path}")
        logger.info("\nSatellite data requires separate download due to file sizes")
        logger.info("Follow instructions in: data/raw/satellite/DOWNLOAD_INSTRUCTIONS.txt")
        
        return True
    
    def download_biodiversity_data(self):
        """Download biodiversity data"""
        logger.info("=" * 60)
        logger.info("DOWNLOADING BIODIVERSITY DATA")
        logger.info("=" * 60)
        
        bio_dir = DATA_DIR / 'biodiversity'
        bio_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Biodiversity data sources:")
        logger.info("1. eBird India API: https://ebird.org/india/region/IN-DL")
        logger.info("2. India Biodiversity Portal: https://indiabiodiversity.org/")
        logger.info("3. NMHS Fauna Database: https://delhi.data.gov.in/datasets_webservices/datasets/7466786")
        logger.info("\nNote: Most biodiversity data requires API keys or manual download")
        
        # Create sample bird diversity template
        template_data = {
            'year': [2019, 2020, 2021, 2022, 2023],
            'species_count': [250, 255, 260, 265, 270],
            'observations': [5000, 5500, 6000, 6200, 6500],
            'endemic_species': [5, 5, 5, 5, 5],
            'threatened_species': [12, 12, 11, 11, 10]
        }
        
        df_birds = pd.DataFrame(template_data)
        birds_template = bio_dir / 'birds' / 'delhi_bird_diversity_template.csv'
        birds_template.parent.mkdir(parents=True, exist_ok=True)
        df_birds.to_csv(birds_template, index=False)
        
        logger.info(f"Template saved: {birds_template}")
        
        return True
    
    def download_socioeconomic_data(self):
        """Download socioeconomic data"""
        logger.info("=" * 60)
        logger.info("DOWNLOADING SOCIOECONOMIC DATA")
        logger.info("=" * 60)
        
        socio_dir = DATA_DIR / 'socioeconomic'
        socio_dir.mkdir(parents=True, exist_ok=True)
        
        # Population template
        pop_data = {
            'year': [2011, 2015, 2019, 2020, 2021, 2022, 2023],
            'population': [16787941, 18500000, 19814000, 20000000, 20100000, 20300000, 20500000],
            'area_sq_km': [1484] * 7,
            'density_per_sq_km': [11313, 12466, 13350, 13478, 13545, 13680, 13815]
        }
        
        df_pop = pd.DataFrame(pop_data)
        pop_template = socio_dir / 'population' / 'delhi_population_template.csv'
        pop_template.parent.mkdir(parents=True, exist_ok=True)
        df_pop.to_csv(pop_template, index=False)
        
        # Vehicle data template
        vehicle_data = {
            'year': [2019, 2020, 2021, 2022, 2023],
            'total_vehicles': [11200000, 11500000, 11800000, 12100000, 12400000],
            'cars': [3360000, 3450000, 3540000, 3630000, 3720000],
            'two_wheelers': [6720000, 6900000, 7080000, 7260000, 7440000],
            'commercial': [560000, 575000, 590000, 605000, 620000]
        }
        
        df_vehicles = pd.DataFrame(vehicle_data)
        vehicles_template = socio_dir / 'vehicles' / 'delhi_vehicles_template.csv'
        vehicles_template.parent.mkdir(parents=True, exist_ok=True)
        df_vehicles.to_csv(vehicles_template, index=False)
        
        logger.info(f"Population template: {pop_template}")
        logger.info(f"Vehicles template: {vehicles_template}")
        logger.info("\nData sources:")
        logger.info("1. Census India: https://censusindia.gov.in/")
        logger.info("2. Economic Survey of Delhi: https://delhi.gov.in/")
        logger.info("3. Delhi Transport Department")
        
        return True
    
    def create_metadata_files(self):
        """Create metadata files for each data category"""
        logger.info("=" * 60)
        logger.info("CREATING METADATA FILES")
        logger.info("=" * 60)
        
        categories = {
            'air_quality': {
                'sources': [
                    'CPCB Real-Time Data: https://cpcb.nic.in/real-time-data/',
                    'data.gov.in: https://www.data.gov.in/resource/real-time-air-quality-index-various-locations',
                    'NAMP Data: https://cpcb.nic.in/namp-data/'
                ],
                'parameters': ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI'],
                'time_range': '2019-2024',
                'stations': ['Anand Vihar', 'RK Puram', 'Punjabi Bagh', 'Dwarka', 'ITO', 'Lodhi Road']
            },
            'water_quality': {
                'sources': [
                    'data.gov.in Yamuna: https://www.data.gov.in/keywords/Yamuna',
                    'CPCB Water Quality Monitoring'
                ],
                'parameters': ['BOD', 'COD', 'DO', 'pH', 'Fecal Coliform', 'Nitrate', 'Phosphate'],
                'time_range': '2019-2023',
                'locations': ['Palla', 'Nizamuddin Bridge', 'Okhla Barrage']
            },
            'forest_greencover': {
                'sources': [
                    'data.gov.in: https://www.data.gov.in/resource/district-wise-forest-cover-delhi',
                    'Delhi Forest Dept: https://forest.delhi.gov.in/',
                    'India State of Forest Report (FSI)'
                ],
                'parameters': ['Forest Cover (sq km)', 'Tree Cover (sq km)', 'Green Cover %'],
                'time_range': '2019-2023'
            },
            'climate_weather': {
                'sources': [
                    'IMD: https://mausam.imd.gov.in/',
                    'data.gov.in Rainfall: https://www.data.gov.in/catalog/rainfall-india',
                    'Kaggle: https://www.kaggle.com/datasets/yug201/daily-climate-time-series-data-delhi-india'
                ],
                'parameters': ['Temperature', 'Humidity', 'Rainfall', 'Wind Speed', 'Pressure'],
                'time_range': '2013-2024'
            }
        }
        
        for category, info in categories.items():
            metadata_path = DATA_DIR / category / 'metadata.txt'
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metadata_path, 'w') as f:
                f.write(f"METADATA: {category.upper().replace('_', ' ')}\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Date Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Data Sources:\n")
                for source in info['sources']:
                    f.write(f"  - {source}\n")
                f.write("\n")
                
                f.write(f"Time Coverage: {info['time_range']}\n\n")
                
                f.write("Parameters:\n")
                for param in info['parameters']:
                    f.write(f"  - {param}\n")
                f.write("\n")
                
                if 'stations' in info:
                    f.write("Monitoring Stations:\n")
                    for station in info['stations']:
                        f.write(f"  - {station}\n")
                    f.write("\n")
                
                if 'locations' in info:
                    f.write("Monitoring Locations:\n")
                    for loc in info['locations']:
                        f.write(f"  - {loc}\n")
                    f.write("\n")
                
                f.write("License: National Data Sharing and Accessibility Policy (NDSAP)\n")
                f.write("Format: CSV, JSON, Excel\n")
            
            logger.info(f"Created metadata: {metadata_path}")
        
        return True
    
    def generate_download_summary(self):
        """Generate summary report of download status"""
        logger.info("=" * 60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 60)
        
        summary_path = BASE_DIR / 'logs' / 'download_summary.txt'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w') as f:
            f.write("DELHI ECOSYSTEM DATA DOWNLOAD SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("AUTOMATED DOWNLOADS:\n")
            f.write("  ✓ Project structure created\n")
            f.write("  ✓ Template files generated\n")
            f.write("  ✓ Metadata files created\n")
            f.write("  ✓ Download scripts prepared\n\n")
            
            f.write("MANUAL DOWNLOADS REQUIRED:\n")
            f.write("  ⚠ Air Quality Data (CPCB):\n")
            f.write("     https://cpcb.nic.in/real-time-data/\n")
            f.write("     → Place in: data/raw/air_quality/cpcb_realtime/\n\n")
            
            f.write("  ⚠ Water Quality Data (Yamuna):\n")
            f.write("     https://www.data.gov.in/keywords/Yamuna\n")
            f.write("     → Place in: data/raw/water_quality/yamuna/\n\n")
            
            f.write("  ⚠ Climate Data (Kaggle):\n")
            f.write("     Run: kaggle datasets download -d yug201/daily-climate-time-series-data-delhi-india\n")
            f.write("     → Extract to: data/raw/climate_weather/kaggle/\n\n")
            
            f.write("  ⚠ Satellite Data:\n")
            f.write("     See: data/raw/satellite/DOWNLOAD_INSTRUCTIONS.txt\n")
            f.write("     Requires: Google Earth Engine / NASA Earthdata account\n\n")
            
            f.write("NEXT STEPS:\n")
            f.write("  1. Complete manual downloads listed above\n")
            f.write("  2. Run data validation: python scripts/validate_data.py\n")
            f.write("  3. Start data preprocessing: python scripts/preprocess_data.py\n\n")
            
            f.write("For detailed instructions, see:\n")
            f.write("  - DATA_COLLECTION_STEPS.md\n")
            f.write("  - data/raw/*/metadata.txt\n")
        
        logger.info(f"Summary saved: {summary_path}")
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        with open(summary_path, 'r') as f:
            print(f.read())
        
        return True

def main():
    """Main execution function"""
    logger.info("Starting Delhi Ecosystem Data Download")
    logger.info(f"Base Directory: {BASE_DIR}")
    logger.info(f"Data Directory: {DATA_DIR}")
    
    downloader = DataDownloader()
    
    # Execute download sequence
    try:
        downloader.download_air_quality_data()
        time.sleep(1)
        
        downloader.download_water_quality_data()
        time.sleep(1)
        
        downloader.download_forest_cover_data()
        time.sleep(1)
        
        downloader.download_climate_data_kaggle()
        time.sleep(1)
        
        downloader.download_satellite_data_info()
        time.sleep(1)
        
        downloader.download_biodiversity_data()
        time.sleep(1)
        
        downloader.download_socioeconomic_data()
        time.sleep(1)
        
        downloader.create_metadata_files()
        time.sleep(1)
        
        downloader.generate_download_summary()
        
        logger.info("\n✓ Data download process completed successfully!")
        logger.info("Check logs/download_summary.txt for next steps")
        
    except Exception as e:
        logger.error(f"Error in download process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
