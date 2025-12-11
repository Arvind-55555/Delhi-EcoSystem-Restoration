#!/usr/bin/env python3
"""
Advanced Data Downloader - Attempts to fetch real data from public APIs
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'

class RealDataDownloader:
    """Download real data from public APIs and sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def download_aqicn_data(self):
        """Download air quality data from World Air Quality Index Project"""
        logger.info("Downloading air quality data from aqicn.org...")
        
        air_dir = DATA_DIR / 'air_quality' / 'aqicn'
        air_dir.mkdir(parents=True, exist_ok=True)
        
        # Delhi stations from aqicn.org
        delhi_stations = [
            'anand-vihar',
            'punjabi-bagh',
            'rk-puram',
            'dwarka-sector-8',
            'ito',
            'lodhi-road'
        ]
        
        all_data = []
        
        for station in delhi_stations:
            try:
                # Public feed URL (limited data)
                url = f"https://api.waqi.info/feed/{station}/?token=demo"
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'ok':
                        station_data = data['data']
                        
                        record = {
                            'timestamp': datetime.now().isoformat(),
                            'station': station,
                            'aqi': station_data.get('aqi', None),
                            'city': station_data.get('city', {}).get('name', ''),
                            'lat': station_data.get('city', {}).get('geo', [None, None])[0],
                            'lon': station_data.get('city', {}).get('geo', [None, None])[1]
                        }
                        
                        # Extract individual pollutants
                        iaqi = station_data.get('iaqi', {})
                        for pollutant in ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']:
                            if pollutant in iaqi:
                                record[pollutant] = iaqi[pollutant].get('v', None)
                        
                        all_data.append(record)
                        logger.info(f"✓ {station}: AQI = {record['aqi']}")
                        time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching {station}: {str(e)}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            output_file = air_dir / f'delhi_current_aqi_{datetime.now().strftime("%Y%m%d")}.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"Saved current AQI data: {output_file}")
            return True
        
        return False
    
    def download_openaq_data(self):
        """Download historical air quality data from OpenAQ"""
        logger.info("Downloading historical air quality from OpenAQ...")
        
        air_dir = DATA_DIR / 'air_quality' / 'openaq'
        air_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # OpenAQ API (open access, no key needed for basic queries)
            base_url = "https://api.openaq.org/v2"
            
            # Get Delhi locations
            locations_url = f"{base_url}/locations"
            params = {
                'country': 'IN',
                'city': 'Delhi',
                'limit': 100
            }
            
            response = self.session.get(locations_url, params=params, timeout=30)
            
            if response.status_code == 200:
                locations_data = response.json()
                
                if 'results' in locations_data:
                    locations = locations_data['results']
                    logger.info(f"Found {len(locations)} monitoring locations in Delhi")
                    
                    # Save locations info
                    df_locations = pd.DataFrame(locations)
                    df_locations.to_csv(air_dir / 'delhi_monitoring_locations.csv', index=False)
                    
                    # Fetch measurements for each location (last 30 days)
                    all_measurements = []
                    
                    for i, loc in enumerate(locations[:5]):  # Limit to 5 locations for demo
                        loc_id = loc.get('id')
                        logger.info(f"Fetching data for location {i+1}/5: {loc.get('name', 'Unknown')}")
                        
                        measurements_url = f"{base_url}/measurements"
                        date_from = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                        
                        meas_params = {
                            'location_id': loc_id,
                            'date_from': date_from,
                            'limit': 1000
                        }
                        
                        meas_response = self.session.get(measurements_url, params=meas_params, timeout=30)
                        
                        if meas_response.status_code == 200:
                            meas_data = meas_response.json()
                            if 'results' in meas_data:
                                all_measurements.extend(meas_data['results'])
                        
                        time.sleep(2)  # Rate limiting
                    
                    if all_measurements:
                        df_measurements = pd.DataFrame(all_measurements)
                        output_file = air_dir / f'delhi_measurements_last_30days_{datetime.now().strftime("%Y%m%d")}.csv'
                        df_measurements.to_csv(output_file, index=False)
                        logger.info(f"Saved {len(all_measurements)} air quality measurements: {output_file}")
                        return True
            
        except Exception as e:
            logger.error(f"Error downloading OpenAQ data: {str(e)}")
        
        return False
    
    def download_kaggle_climate_data(self):
        """Attempt to download Kaggle dataset if API is configured"""
        logger.info("Checking for Kaggle API configuration...")
        
        climate_dir = DATA_DIR / 'climate_weather' / 'kaggle'
        climate_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import subprocess
            
            # Check if kaggle is installed
            result = subprocess.run(['which', 'kaggle'], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Kaggle CLI found. Attempting download...")
                
                # Download dataset
                download_cmd = [
                    'kaggle', 'datasets', 'download', '-d',
                    'yug201/daily-climate-time-series-data-delhi-india',
                    '-p', str(climate_dir),
                    '--unzip'
                ]
                
                result = subprocess.run(download_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✓ Kaggle dataset downloaded successfully!")
                    return True
                else:
                    logger.warning(f"Kaggle download failed: {result.stderr}")
            else:
                logger.info("Kaggle CLI not found. Install with: pip install kaggle")
                logger.info("Then configure API token from https://www.kaggle.com/account")
        
        except Exception as e:
            logger.error(f"Error with Kaggle download: {str(e)}")
        
        return False
    
    def download_world_bank_data(self):
        """Download environmental indicators from World Bank API"""
        logger.info("Downloading environmental data from World Bank...")
        
        wb_dir = DATA_DIR / 'socioeconomic' / 'world_bank'
        wb_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # World Bank API (open access)
            base_url = "http://api.worldbank.org/v2"
            
            # Indicators for India
            indicators = {
                'EN.ATM.PM25.MC.M3': 'PM2.5_air_pollution',
                'EN.ATM.CO2E.PC': 'CO2_emissions_per_capita',
                'AG.LND.FRST.ZS': 'Forest_area_percent',
                'SP.POP.TOTL': 'Population_total',
                'EN.URB.MCTY': 'Population_largest_city'
            }
            
            all_data = []
            
            for indicator, name in indicators.items():
                url = f"{base_url}/country/IN/indicator/{indicator}"
                params = {
                    'format': 'json',
                    'date': '2010:2023',
                    'per_page': 100
                }
                
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if len(data) > 1 and data[1]:
                        for record in data[1]:
                            all_data.append({
                                'indicator': name,
                                'indicator_code': indicator,
                                'year': record.get('date'),
                                'value': record.get('value'),
                                'country': record.get('country', {}).get('value')
                            })
                        logger.info(f"✓ Downloaded {name}")
                
                time.sleep(1)
            
            if all_data:
                df = pd.DataFrame(all_data)
                output_file = wb_dir / 'india_environmental_indicators.csv'
                df.to_csv(output_file, index=False)
                logger.info(f"Saved World Bank data: {output_file}")
                return True
        
        except Exception as e:
            logger.error(f"Error downloading World Bank data: {str(e)}")
        
        return False
    
    def download_nasa_power_weather(self):
        """Download weather data from NASA POWER API"""
        logger.info("Downloading weather data from NASA POWER...")
        
        weather_dir = DATA_DIR / 'climate_weather' / 'nasa_power'
        weather_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # NASA POWER API (open access)
            base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
            
            # Delhi coordinates
            lat, lon = 28.7041, 77.1025
            
            # Parameters
            start_date = '20190101'
            end_date = '20231231'
            
            parameters = 'T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M,WS10M'
            
            params = {
                'parameters': parameters,
                'community': 'RE',
                'longitude': lon,
                'latitude': lat,
                'start': start_date,
                'end': end_date,
                'format': 'JSON'
            }
            
            logger.info("Fetching weather data (this may take a minute)...")
            response = self.session.get(base_url, params=params, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'properties' in data and 'parameter' in data['properties']:
                    param_data = data['properties']['parameter']
                    
                    # Convert to DataFrame
                    df_list = []
                    dates = list(param_data[list(param_data.keys())[0]].keys())
                    
                    for date in dates:
                        record = {'date': date}
                        for param in param_data:
                            record[param] = param_data[param].get(date)
                        df_list.append(record)
                    
                    df = pd.DataFrame(df_list)
                    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                    
                    # Rename columns
                    df.rename(columns={
                        'T2M': 'temp_mean_C',
                        'T2M_MIN': 'temp_min_C',
                        'T2M_MAX': 'temp_max_C',
                        'PRECTOTCORR': 'precipitation_mm',
                        'RH2M': 'humidity_percent',
                        'WS10M': 'wind_speed_ms'
                    }, inplace=True)
                    
                    output_file = weather_dir / 'delhi_weather_2019_2023.csv'
                    df.to_csv(output_file, index=False)
                    logger.info(f"✓ Saved NASA POWER weather data: {output_file}")
                    logger.info(f"  Records: {len(df)} days from 2019-2023")
                    return True
        
        except Exception as e:
            logger.error(f"Error downloading NASA POWER data: {str(e)}")
        
        return False

def main():
    logger.info("=" * 70)
    logger.info("DOWNLOADING REAL DATA FROM PUBLIC APIs")
    logger.info("=" * 70)
    
    downloader = RealDataDownloader()
    
    success_count = 0
    total_count = 0
    
    # 1. Air Quality Data
    logger.info("\n[1/5] Air Quality Data...")
    total_count += 2
    if downloader.download_aqicn_data():
        success_count += 1
    time.sleep(2)
    if downloader.download_openaq_data():
        success_count += 1
    time.sleep(2)
    
    # 2. Weather Data
    logger.info("\n[2/5] Weather Data...")
    total_count += 1
    if downloader.download_nasa_power_weather():
        success_count += 1
    time.sleep(2)
    
    # 3. Climate Data (Kaggle)
    logger.info("\n[3/5] Climate Data (Kaggle)...")
    total_count += 1
    if downloader.download_kaggle_climate_data():
        success_count += 1
    time.sleep(2)
    
    # 4. Environmental Indicators
    logger.info("\n[4/5] Environmental Indicators (World Bank)...")
    total_count += 1
    if downloader.download_world_bank_data():
        success_count += 1
    
    logger.info("\n" + "=" * 70)
    logger.info(f"DOWNLOAD COMPLETE: {success_count}/{total_count} sources successful")
    logger.info("=" * 70)
    
    logger.info("\nDownloaded data locations:")
    logger.info("  - Air Quality: data/raw/air_quality/aqicn/ and openaq/")
    logger.info("  - Weather: data/raw/climate_weather/nasa_power/")
    logger.info("  - Environmental: data/raw/socioeconomic/world_bank/")
    
    if success_count < total_count:
        logger.info("\nSome downloads failed. Check logs for details.")
        logger.info("Manual downloads may be required for some datasets.")

if __name__ == "__main__":
    main()
