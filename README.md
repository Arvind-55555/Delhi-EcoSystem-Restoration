# Delhi Ecosystem Restoration - ML Platform

**AI-Powered Ecosystem Health Monitoring and Restoration Planning for Delhi, India**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.2-61dafb)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Based on IPCC AR6 framework and utilizing open government data from data.gov.in.

---

## Overview

A comprehensive machine learning platform for ecosystem restoration in Delhi, featuring:
- **Real-time Air Quality Monitoring** - Track AQI, PM2.5, and 6 pollutants
- **Ecosystem Health Scoring** - 0-100 composite health metric
- **AI-Powered Forecasting** - 7-day PM2.5 predictions using LSTM and Prophet
- **Restoration Optimizer** - 100+ AI-optimized strategies using NSGA-II
- **Interactive Dashboard** - Beautiful React UI with real-time data visualization

### Key Achievements
✅ **5 ML Models** trained (XGBoost R² = 0.9975)  
✅ **18 Datasets** collected (16,222 records)  
✅ **89 Features** engineered from 18 base features  
✅ **100 Restoration Scenarios** optimized  
✅ **33% PM2.5 Reduction** potential identified  
✅ **Production-Ready** API and web dashboard  

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm

### Installation & Run

```bash
# Clone repository
git clone https://github.com/Arvind-55555/Delhi-EcoSystem-Restoration.git
cd Delhi-EcoSystem-Restoration

# Install Python dependencies
pip install -r requirements.txt

# Deploy dashboard (automated)
python deploy_dashboard.py --mode dev
```

**Access:**
- Dashboard: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Features

### Dashboard
- Real-time AQI monitoring with color-coded status
- Ecosystem Health Score (0-100)
- 6 pollutant level trackers
- Weather metrics display
- 7-day PM2.5 forecast chart
- Personalized health recommendations

### Restoration Planner
- Interactive budget configuration (₹10M - ₹1,000M)
- Timeline selection (1-15 years)
- 4 optimization priorities
- AI-generated restoration scenarios
- Implementation roadmap

### ML Models
- **XGBoost**: R²=0.9975 (Best)
- **LSTM**: Multi-step forecasting
- **Prophet**: Seasonal forecasting
- **Random Forest**: Feature importance
- **Linear Regression**: Baseline

---

## Project Structure

```
├── api/                   # FastAPI REST API (8 endpoints)
├── dashboard/            # React frontend
├── data/                 # 18 datasets (16,222 records)
├── models/               # 5 trained ML models
├── scripts/              # 9 Python training scripts
├── results/              # 100 optimization scenarios
├── docs/                 # Complete documentation
├── deploy_dashboard.py   # Automated deployment
└── serve_production.py   # Production server
```

---

## ML Pipeline

1. **Data Collection** - 18 datasets from NASA POWER, World Bank, synthetic sources
2. **Feature Engineering** - 89 features from 18 base features
3. **Model Training** - 5 models (XGBoost best: R²=0.9975)
4. **Optimization** - NSGA-II multi-objective (100 scenarios)
5. **Deployment** - FastAPI + React dashboard

---

## Results

- **Model Accuracy**: 99.75% (R²=0.9975, RMSE=2.21 µg/m³)
- **Best Scenario**: 33% PM2.5 reduction (₹1,000M, 11.3 years)
- **Quick Win**: 2% reduction (₹24.4M, 1 year)

---

## Documentation

- [Final Report](FINAL_REPORT.md) - Complete project summary
- [Project Analysis](docs/PROJECT_ANALYSIS.md) - Initial analysis
- [ML Architecture](docs/ML_MODEL_ARCHITECTURE.md) - Model design
- [API Docs](api/README.md) - API endpoints
- [Dashboard Guide](dashboard/README.md) - Frontend setup

---

## Contributing

Contributions welcome! Fork, create feature branch, and submit PR.

---

## Contact

**GitHub**: [@Arvind-55555](https://github.com/Arvind-55555)  
**Issues**: [Create Issue](https://github.com/Arvind-55555/Delhi-EcoSystem-Restoration/issues)

---

## Acknowledgments

Data: NASA POWER, World Bank, CPCB  
Tech: React, FastAPI, XGBoost, TensorFlow, Prophet, Tailwind CSS

---

**Star this repo if you find it helpful!**

Last Updated: December 2025
