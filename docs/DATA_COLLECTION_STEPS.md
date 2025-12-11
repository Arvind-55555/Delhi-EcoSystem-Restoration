# Data Collection Steps for Delhi Ecosystem Restoration ML Model

## STEP-BY-STEP DATA GATHERING GUIDE

---

## PHASE 1: AIR QUALITY DATA COLLECTION

### Step 1.1: CPCB Real-Time Air Quality Data

**Source**: Central Pollution Control Board (CPCB)  
**URL**: https://cpcb.nic.in/real-time-data/

**Procedure**:
1. Visit CPCB Real-Time Air Quality Index portal
2. Navigate to Delhi NCR section
3. Select monitoring stations in Delhi:
   - Anand Vihar
   - RK Puram
   - Punjabi Bagh
   - Dwarka
   - ITO
   - IGI Airport
   - Major Dhyan Chand Stadium
   - Lodhi Road
   - All other available stations

4. Download historical data:
   - **Method 1**: Use the download button for CSV files (if available)
   - **Method 2**: Use Python API scraping (code provided below)
   - **Method 3**: Access via data.gov.in portal

**Parameters to collect**:
- PM2.5 (µg/m³)
- PM10 (µg/m³)
- NO2 (µg/m³)
- SO2 (µg/m³)
- CO (mg/m³)
- O3 (µg/m³)
- AQI value
- AQI category (Good/Satisfactory/Moderate/Poor/Very Poor/Severe)
- Timestamp (hourly data)

**Time Range**: 2019-01-01 to 2024-12-31 (or latest available)

**Python Code for Data Collection**:
```python
import requests
import pandas as pd
from datetime import datetime, timedelta

# CPCB API endpoint (example - may need to be updated)
def fetch_cpcb_data(station_id, start_date, end_date):
    url = "https://api.cpcb.gov.in/aqi/data"  # Update with actual endpoint
    params = {
        'station': station_id,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d')
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        print(f"Error: {response.status_code}")
        return None

# Delhi stations
delhi_stations = [
    'Anand Vihar', 'RK Puram', 'Punjabi Bagh', 'Dwarka', 
    'ITO', 'IGI Airport', 'Major Dhyan Chand Stadium', 'Lodhi Road'
]

# Fetch data for each station
all_data = []
for station in delhi_stations:
    df = fetch_cpcb_data(station, datetime(2019, 1, 1), datetime(2024, 12, 31))
    if df is not None:
        df['station'] = station
        all_data.append(df)

# Combine and save
air_quality_df = pd.concat(all_data, ignore_index=True)
air_quality_df.to_csv('delhi_air_quality_2019_2024.csv', index=False)
```

**Alternative**: Download from data.gov.in
- URL: https://www.data.gov.in/resource/real-time-air-quality-index-various-locations
- Click "Download" and select format (CSV/JSON/XML)

### Step 1.2: NAMP (National Air Monitoring Programme) Data

**Source**: CPCB NAMP Portal  
**URL**: https://cpcb.nic.in/namp-data/

**Procedure**:
1. Visit NAMP Data portal
2. Select "Delhi" from state dropdown
3. Choose year range: 2019-2024
4. Download Excel/CSV files for each year
5. Consolidate into a single dataset

**Data Format**: Usually provided as Excel files with monthly averages

**Merge with real-time data** for comprehensive coverage

---

## PHASE 2: WATER QUALITY DATA COLLECTION

### Step 2.1: Yamuna River Water Quality Data

**Source**: Central Water Commission / CPCB  
**URL**: https://www.data.gov.in/keywords/Yamuna

**Procedure**:
1. Visit data.gov.in Yamuna datasets page
2. Download "State-wise water quality data of river Yamuna (2019-2023)"
3. Filter for Delhi locations:
   - Palla (upstream entry)
   - Nizamuddin Bridge
   - Okhla Barrage
   - Other monitoring points

**Parameters to collect**:
- BOD (Biochemical Oxygen Demand) - mg/L
- COD (Chemical Oxygen Demand) - mg/L
- DO (Dissolved Oxygen) - mg/L
- pH
- Fecal Coliform - MPN/100ml
- Total Coliform - MPN/100ml
- Nitrate - mg/L
- Phosphate - mg/L
- Heavy metals (Lead, Mercury, Cadmium, Chromium) - mg/L
- Turbidity - NTU
- Temperature - °C

**Time Range**: 2019-2023 (quarterly or monthly monitoring data)

### Step 2.2: Groundwater Quality Data

**Source**: Central Ground Water Board (CGWB) / Delhi Jal Board

**Procedure**:
1. Visit CGWB website: https://cgwb.gov.in/
2. Navigate to "Data Access" section
3. Download groundwater quality reports for Delhi
4. Extract parameters:
   - pH
   - Electrical Conductivity (EC) - µS/cm
   - Total Dissolved Solids (TDS) - mg/L
   - Hardness - mg/L
   - Nitrate - mg/L
   - Fluoride - mg/L
   - Arsenic - µg/L

**Time Range**: Annual reports 2019-2024

---

## PHASE 3: FOREST & GREEN COVER DATA COLLECTION

### Step 3.1: Forest Cover Data

**Source**: Forest Survey of India (FSI) / Delhi Forest Department  
**URL**: https://www.data.gov.in/resource/district-wise-forest-cover-delhi

**Procedure**:
1. Download district-wise forest cover dataset
2. Visit Delhi Forest Department: https://forest.delhi.gov.in/
3. Navigate to "Extent of Forest and Tree Cover" section
4. Download India State of Forest Report (ISFR) data for Delhi (2019, 2021, 2023 editions)

**Data Points**:
- Total forest cover area (sq km)
- Very Dense Forest (VDF) - sq km
- Moderately Dense Forest (MDF) - sq km
- Open Forest - sq km
- Tree cover (outside forest areas) - sq km
- Total green cover percentage
- District-wise breakdown (if available)

### Step 3.2: Tree Census Data

**Source**: Delhi Government / Municipal Corporations  
**URL**: https://www.data.opencity.in/ (search for Delhi tree census)

**Procedure**:
1. Visit Delhi Open Data portal or Municipal Corporation websites
2. Search for "Tree Census" datasets
3. Download tree inventory data (2019 census if available)

**Data Points**:
- Tree count per ward/district
- Species distribution
- Tree health status
- Tree canopy area
- GPS coordinates (if available)

---

## PHASE 4: CLIMATE & WEATHER DATA COLLECTION

### Step 4.1: IMD Weather Data

**Source**: India Meteorological Department (IMD)  
**URL**: https://mausam.imd.gov.in/

**Procedure**:
1. Visit IMD website
2. Navigate to "Data Supply" section
3. Request historical data for Delhi (Safdarjung, Palam stations)
   - May require registration and payment for detailed data
4. Alternative: Download from data.gov.in

**URL (data.gov.in)**: https://delhi.data.gov.in/keywords/temperature

**Parameters**:
- Daily maximum temperature (°C)
- Daily minimum temperature (°C)
- Mean temperature (°C)
- Relative humidity (%) - morning and evening
- Wind speed (km/h)
- Wind direction
- Rainfall (mm) - daily, monthly totals
- Sunshine hours

**Time Range**: 2013-2024 (for trend analysis)

### Step 4.2: Rainfall Data

**Source**: IMD / data.gov.in  
**URL**: https://www.data.gov.in/catalog/rainfall-india

**Procedure**:
1. Download "Rainfall in India" dataset
2. Filter for Delhi subdivision
3. Extract monthly and seasonal rainfall data

**Data Points**:
- Monthly rainfall (mm)
- Seasonal totals (Monsoon, Winter, Summer)
- Departure from normal (%)
- Rainy days count

### Step 4.3: Kaggle Climate Datasets (Supplementary)

**Source**: Kaggle  
**URL**: https://www.kaggle.com/datasets/yug201/daily-climate-time-series-data-delhi-india

**Procedure**:
1. Create Kaggle account (free)
2. Download "Daily Climate Time Series Data - Delhi, India (2013-2024)"
3. Use for filling gaps in IMD data

---

## PHASE 5: SATELLITE & REMOTE SENSING DATA COLLECTION

### Step 5.1: Sentinel-2 Land Use/Land Cover (LULC)

**Source**: ESA Copernicus / ArcGIS / Google Earth Engine  
**URL**: https://www.arcgis.com/home/item.html?id=352427beedd746ae9c407080b38b85a5

**Procedure**:
1. **Option A: Google Earth Engine**
   - Register for Earth Engine account
   - Use Earth Engine Code Editor
   - Run LULC classification script for Delhi
   - Export results as GeoTIFF

2. **Option B: Sentinel Hub**
   - Create free account (trial)
   - Use EO Browser: https://apps.sentinel-hub.com/eo-browser/
   - Search for Delhi coordinates (28.7041° N, 77.1025° E)
   - Download Sentinel-2 imagery (2020-2024, quarterly)
   - Apply LULC classification using SNAP or Python

**Python Code (Google Earth Engine)**:
```python
import ee
ee.Initialize()

# Define Delhi boundary
delhi = ee.Geometry.Rectangle([76.8, 28.4, 77.4, 28.9])

# Load Sentinel-2 imagery
sentinel = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterBounds(delhi) \
    .filterDate('2023-01-01', '2023-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .median()

# Load ESA WorldCover (10m LULC)
lulc = ee.ImageCollection('ESA/WorldCover/v100').first().clip(delhi)

# Export
task = ee.batch.Export.image.toDrive(
    image=lulc,
    description='Delhi_LULC_2023',
    scale=10,
    region=delhi,
    fileFormat='GeoTIFF'
)
task.start()
```

**LULC Classes**:
- Urban/Built-up
- Agriculture
- Forest/Trees
- Grassland
- Water bodies
- Barren land

### Step 5.2: MODIS NDVI (Vegetation Index)

**Source**: NASA MODIS  
**URL**: https://modis.gsfc.nasa.gov/

**Procedure**:
1. Visit NASA Earthdata: https://earthdata.nasa.gov/
2. Register for free account
3. Use AppEEARS tool: https://appeears.earthdatacloud.nasa.gov/
4. Request MODIS NDVI (MOD13Q1) for Delhi
   - Product: MOD13Q1 (250m, 16-day composite)
   - Time range: 2013-2024
   - Area: Delhi bounding box
5. Download HDF files and convert to GeoTIFF

**Python Code (MODIS Download)**:
```python
from pymodis import downmodis

# Download MODIS NDVI
modis_down = downmodis.downModis(
    destinationFolder='./modis_data',
    tiles='h25v06',  # Tile covering Delhi
    product='MOD13Q1.061',
    today='2024-12-31',
    delta=365*5  # Last 5 years
)
modis_down.connect()
modis_down.downloadsAllDay()
```

**Extract NDVI values**:
```python
import rasterio
import geopandas as gpd

# Load Delhi boundary shapefile
delhi_boundary = gpd.read_file('delhi_boundary.shp')

# Extract NDVI for Delhi
with rasterio.open('MOD13Q1_NDVI_2023.tif') as src:
    ndvi_delhi = src.read(1)  # NDVI band
    # Mask to Delhi boundary
    # Calculate mean NDVI
    mean_ndvi = ndvi_delhi.mean()
    print(f"Mean NDVI for Delhi: {mean_ndvi}")
```

### Step 5.3: Landsat Land Surface Temperature (LST)

**Source**: USGS Landsat  
**URL**: https://earthexplorer.usgs.gov/

**Procedure**:
1. Visit USGS Earth Explorer
2. Define Delhi area (Geocoder or coordinate entry)
3. Select datasets: Landsat 8/9 Level-2, Collection 2
4. Date range: 2019-2024, summer months (April-June for UHI analysis)
5. Download Surface Temperature band (ST_B10)
6. Process to convert DN to temperature (°C)

**Python Code (LST Calculation)**:
```python
import rasterio
import numpy as np

def calculate_lst(landsat_file):
    with rasterio.open(landsat_file) as src:
        st_band = src.read(1)  # Thermal band
        # Convert to Celsius (Landsat 8/9 specific formula)
        lst_celsius = st_band * 0.00341802 + 149.0 - 273.15
        return lst_celsius

# Process multiple dates
lst_2023_summer = calculate_lst('LC08_L2SP_ST_B10_20230515.TIF')
```

### Step 5.4: EOS Data Analytics LandViewer (Alternative)

**Source**: EOS LandViewer  
**URL**: https://eos.com/landviewer/delhi/

**Procedure**:
1. Visit EOS LandViewer
2. Search for Delhi
3. Select satellite (Sentinel-2, Landsat)
4. Apply indices:
   - NDVI (vegetation)
   - NDMI (moisture)
   - NDBI (built-up)
5. Export results as GeoTIFF or PNG

**Free tier**: Limited downloads per month, consider paid plan for extensive data

---

## PHASE 6: BIODIVERSITY DATA COLLECTION

### Step 6.1: Fauna Distribution Data

**Source**: National Biodiversity Database / NMHS  
**URL**: https://delhi.data.gov.in/datasets_webservices/datasets/7466786

**Procedure**:
1. Download Fauna Distribution in IHR dataset
2. Filter for Delhi/NCR region species
3. Extract species lists, occurrence data

### Step 6.2: Bird Diversity Data

**Source**: eBird India  
**URL**: https://ebird.org/india/region/IN-DL

**Procedure**:
1. Visit eBird India portal
2. Request data for Delhi (IN-DL region code)
3. Download Basic Dataset (requires research request)
4. Alternative: Use eBird API for recent sightings

**Python Code (eBird API)**:
```python
import requests

def get_ebird_data(region_code='IN-DL'):
    url = f"https://api.ebird.org/v2/data/obs/{region_code}/recent"
    headers = {'X-eBirdApiToken': 'YOUR_API_KEY'}  # Register for free key
    response = requests.get(url, headers=headers)
    return response.json()

delhi_birds = get_ebird_data()
```

### Step 6.3: Flora Data

**Source**: Botanical Survey of India / Delhi University Herbarium

**Procedure**:
1. Search for published flora surveys of Delhi
2. Contact Delhi University Botany Department for species lists
3. Use India Biodiversity Portal: https://indiabiodiversity.org/
   - Search for Delhi observations
   - Download species occurrence data

---

## PHASE 7: SOCIO-ECONOMIC & AUXILIARY DATA

### Step 7.1: Population Data

**Source**: Census of India  
**URL**: https://censusindia.gov.in/

**Procedure**:
1. Download Delhi district-wise population data (2011 Census)
2. Use projections for 2021-2024 from Economic Survey of Delhi
3. Extract:
   - Total population
   - Population density (per sq km)
   - Urban/rural split
   - Ward-wise data (if available)

### Step 7.2: Vehicle Registration Data

**Source**: Delhi Transport Department / data.gov.in

**Procedure**:
1. Search data.gov.in for "vehicle registration Delhi"
2. Download year-wise vehicle registration data
3. Extract:
   - Total registered vehicles
   - Vehicle type distribution (cars, two-wheelers, commercial)
   - Annual growth rate

### Step 7.3: Industrial Data

**Source**: Delhi Pollution Control Committee (DPCC)

**Procedure**:
1. Visit DPCC website
2. Search for "Consent to Operate" or industrial registrations
3. Extract:
   - Number of industries by category
   - Location (industrial areas)
   - Polluting vs. non-polluting industries

---

## PHASE 8: DATA ORGANIZATION & STORAGE

### Step 8.1: Directory Structure

Create the following folder structure:
```
Ecosystem/
├── data/
│   ├── raw/
│   │   ├── air_quality/
│   │   │   ├── cpcb_realtime/
│   │   │   ├── namp_data/
│   │   │   └── metadata.txt
│   │   ├── water_quality/
│   │   │   ├── yamuna/
│   │   │   ├── groundwater/
│   │   │   └── metadata.txt
│   │   ├── forest_greencover/
│   │   │   ├── fsi_reports/
│   │   │   ├── tree_census/
│   │   │   └── metadata.txt
│   │   ├── climate_weather/
│   │   │   ├── imd_temperature/
│   │   │   ├── imd_rainfall/
│   │   │   └── metadata.txt
│   │   ├── satellite/
│   │   │   ├── sentinel2_lulc/
│   │   │   ├── modis_ndvi/
│   │   │   ├── landsat_lst/
│   │   │   └── metadata.txt
│   │   ├── biodiversity/
│   │   │   ├── fauna/
│   │   │   ├── birds/
│   │   │   ├── flora/
│   │   │   └── metadata.txt
│   │   └── socioeconomic/
│   │       ├── population/
│   │       ├── vehicles/
│   │       ├── industries/
│   │       └── metadata.txt
│   ├── processed/
│   │   ├── air_quality_cleaned.csv
│   │   ├── water_quality_cleaned.csv
│   │   ├── vegetation_index_timeseries.csv
│   │   ├── climate_data_cleaned.csv
│   │   └── master_dataset.parquet
│   └── features/
│       ├── feature_engineered.parquet
│       └── feature_dictionary.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_eda.ipynb
└── docs/
    ├── data_sources.md
    └── data_dictionary.md
```

### Step 8.2: Metadata Documentation

For each dataset, create a metadata file with:
- **Source**: URL or organization
- **Date Downloaded**: YYYY-MM-DD
- **Time Coverage**: Start and end dates
- **Spatial Coverage**: Locations/coordinates
- **Parameters**: List of variables
- **Format**: CSV, Excel, GeoTIFF, etc.
- **Size**: File size in MB/GB
- **License**: Open data license type
- **Contact**: Email for questions
- **Notes**: Any data quality issues, missing values, etc.

Example metadata template:
```yaml
dataset_name: CPCB Real-Time Air Quality Data
source: https://cpcb.nic.in/real-time-data/
date_downloaded: 2024-12-11
time_coverage: 2019-01-01 to 2024-12-10
spatial_coverage: 
  - Anand Vihar
  - RK Puram
  - Punjabi Bagh
  - Dwarka (and 5 other stations)
parameters:
  - PM2.5 (µg/m³)
  - PM10 (µg/m³)
  - NO2 (µg/m³)
  - SO2 (µg/m³)
  - CO (mg/m³)
  - O3 (µg/m³)
  - AQI
format: CSV
size: 450 MB
license: National Data Sharing and Accessibility Policy (NDSAP)
notes: 
  - Missing data for May 2021 due to sensor maintenance
  - Outliers detected in June 2020 (validated as real pollution events)
```

### Step 8.3: Data Quality Checklist

Before moving to preprocessing, verify:
- [ ] All required datasets downloaded
- [ ] File formats compatible (CSV/Excel/GeoTIFF)
- [ ] Time ranges overlap for correlation analysis
- [ ] Spatial coordinates consistent (WGS84 coordinate system)
- [ ] Metadata documented for each dataset
- [ ] No corrupted files
- [ ] Sufficient disk space for processing (~50 GB recommended)

---

## PHASE 9: DATA PREPROCESSING PIPELINE

### Step 9.1: Air Quality Data Cleaning

```python
import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv('data/raw/air_quality/cpcb_realtime/delhi_aqi_2019_2024.csv')

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Handle missing values
# Option 1: Forward fill for short gaps (<6 hours)
df['PM2.5'] = df['PM2.5'].fillna(method='ffill', limit=6)

# Option 2: Interpolate for longer gaps
df['PM10'] = df['PM10'].interpolate(method='time')

# Remove outliers (Z-score method)
from scipy import stats
z_scores = np.abs(stats.zscore(df[['PM2.5', 'PM10', 'NO2']].dropna()))
df_clean = df[(z_scores < 3).all(axis=1)]

# Save cleaned data
df_clean.to_csv('data/processed/air_quality_cleaned.csv', index=False)
```

### Step 9.2: Satellite Data Processing

```python
import rasterio
import geopandas as gpd

# Load Delhi boundary
delhi_boundary = gpd.read_file('data/spatial/delhi_boundary.shp')

# Process NDVI time-series
ndvi_timeseries = []
for file in sorted(glob.glob('data/raw/satellite/modis_ndvi/*.tif')):
    with rasterio.open(file) as src:
        ndvi = src.read(1)
        # Mask to Delhi
        # Calculate statistics
        mean_ndvi = ndvi.mean()
        date = extract_date_from_filename(file)
        ndvi_timeseries.append({'date': date, 'mean_ndvi': mean_ndvi})

ndvi_df = pd.DataFrame(ndvi_timeseries)
ndvi_df.to_csv('data/processed/vegetation_index_timeseries.csv', index=False)
```

---

## PHASE 10: DATA VALIDATION

### Step 10.1: Cross-Validation with External Sources

Compare collected data with:
1. **Air Quality**: Validate against AQI data from aqicn.org
2. **Weather**: Cross-check with Weather Underground historical data
3. **Green Cover**: Validate with Google Earth Engine NDVI estimates

### Step 10.2: Statistical Validation

```python
# Check for consistency
# Example: Correlation between nearby air quality stations
import seaborn as sns
import matplotlib.pyplot as plt

station_corr = df.pivot_table(
    values='PM2.5', 
    index='timestamp', 
    columns='station'
).corr()

sns.heatmap(station_corr, annot=True, cmap='coolwarm')
plt.title('PM2.5 Correlation Between Stations')
plt.show()
```

Expected: High correlation (>0.7) between nearby stations

---

## SUMMARY OF DATA SOURCES

| **Category** | **Dataset** | **Source** | **URL** | **Format** |
|--------------|-------------|------------|---------|------------|
| Air Quality | Real-time AQI | CPCB | https://cpcb.nic.in/real-time-data/ | CSV/API |
| Air Quality | NAMP Data | CPCB | https://cpcb.nic.in/namp-data/ | Excel |
| Air Quality | Category-wise AQI Delhi | data.gov.in | https://www.data.gov.in/resource/category-wise-air-quality-index-major-metropolitan-city-delhi | CSV |
| Water Quality | Yamuna River Quality | data.gov.in | https://www.data.gov.in/keywords/Yamuna | CSV |
| Water Quality | Groundwater | CGWB | https://cgwb.gov.in/ | PDF/Excel |
| Forest Cover | District-wise Forest | data.gov.in | https://www.data.gov.in/resource/district-wise-forest-cover-delhi | CSV |
| Forest Cover | Tree Census | data.opencity.in | https://www.data.opencity.in/ | CSV |
| Climate | Temperature & Humidity | data.gov.in | https://delhi.data.gov.in/keywords/temperature | CSV |
| Climate | Rainfall | data.gov.in | https://www.data.gov.in/catalog/rainfall-india | CSV |
| Climate | Daily Weather (2013-2024) | Kaggle | https://www.kaggle.com/datasets/yug201/daily-climate-time-series-data-delhi-india | CSV |
| Satellite | Sentinel-2 LULC | Google Earth Engine | https://earthengine.google.com/ | GeoTIFF |
| Satellite | MODIS NDVI | NASA | https://modis.gsfc.nasa.gov/ | HDF/GeoTIFF |
| Satellite | Landsat LST | USGS | https://earthexplorer.usgs.gov/ | GeoTIFF |
| Biodiversity | Fauna Distribution | data.gov.in | https://delhi.data.gov.in/datasets_webservices/datasets/7466786 | CSV |
| Biodiversity | Bird Sightings | eBird India | https://ebird.org/india/region/IN-DL | CSV/API |
| Socio-Economic | Population | Census India | https://censusindia.gov.in/ | Excel |
| Socio-Economic | Vehicle Registration | data.gov.in | Search portal | CSV |

---

## NEXT STEPS AFTER DATA COLLECTION

1. **Data Integration**: Merge all datasets into a master temporal-spatial database
2. **Feature Engineering**: Create derived features (indices, trends, lags)
3. **Exploratory Data Analysis**: Visualize patterns, correlations, trends
4. **Model Development**: Train baseline and advanced ML models
5. **Validation**: Test model predictions against held-out data
6. **Deployment**: Integrate into interactive dashboard

---

## ESTIMATED TIME & RESOURCES

**Time Estimate**: 3-4 weeks for complete data collection and preprocessing
**Disk Space**: ~50-100 GB (depending on satellite data resolution)
**Computational Resources**: 
- 16 GB RAM (minimum for satellite processing)
- GPU recommended for deep learning models
- Cloud computing (Google Colab Pro, AWS) if local resources insufficient

**Cost**: Mostly free open data, potential costs:
- IMD detailed data (if required): ₹500-5000
- Sentinel Hub/EOS LandViewer premium (optional): $50-200/month
- Cloud storage (AWS S3, Google Cloud): $10-50/month

---

## CONTACT & SUPPORT

For questions on data access or technical issues:
- **CPCB**: https://cpcb.nic.in/contact/
- **data.gov.in Support**: https://www.data.gov.in/help
- **Google Earth Engine Forum**: https://groups.google.com/g/google-earth-engine-developers
- **Stack Overflow**: Tag questions with `[earth-engine]`, `[geopandas]`, `[remote-sensing]`

---

## LEGAL & ETHICAL COMPLIANCE

- All data from data.gov.in is under **National Data Sharing and Accessibility Policy (NDSAP)**: Free to use with attribution
- Satellite data (Sentinel, Landsat): Open access under Copernicus/USGS terms
- eBird data: Academic/research use allowed with citation
- **Citation Format**: Include dataset source and access date in publications
- **No redistribution** of raw data without permission (processed/aggregated data OK)
- **Privacy**: No personal identifiable information in any dataset used

---

**END OF DATA COLLECTION GUIDE**
