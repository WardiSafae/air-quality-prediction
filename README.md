

Readme · MD
# 🌍 Air Quality Prediction in Moroccan Cities
### End-to-End Machine Learning Pipeline · Real-World API Data · Federated Learning
 
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flower-1.30-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/OpenWeatherMap-API-orange?style=for-the-badge"/>
</p>
<p align="center">
  <b>Collected 996 real data points · Trained 4 models · Simulated Federated Learning across 6 cities</b>
</p>
---
 
## 📌 Project Overview
 
This project builds a **complete, production-oriented ML pipeline** to predict air quality (PM2.5 concentration and AQI class) in six major Moroccan cities — using **real data collected over two weeks** via the OpenWeatherMap API.
 
It goes beyond a standard academic exercise by implementing:
- A **fault-tolerant, resumable data collector** with checkpointing
- **Three supervised ML models** benchmarked side by side
- **Two federated learning implementations** (manual FedAvg + Flower framework)
> Developed as the final project for the *Big Data & Intelligent Systems* Master's program (BDSI) — Université Sidi Mohamed Ben Abdellah, Fès.
 
---
 
## 🏙️ Cities Covered
 
| City | Region | Lat | Lon |
|------|--------|-----|-----|
| Casablanca | Atlantic coast | 33.57 | -7.59 |
| Rabat | Atlantic coast | 34.02 | -6.84 |
| Marrakech | Center | 31.63 | -7.98 |
| Fès | North interior | 34.02 | -5.01 |
| Tanger | North | 35.76 | -5.83 |
| Agadir | South Atlantic | 30.43 | -9.60 |
 
---
 
## 🗂️ Project Structure
 
```
projet_air_quality/
│
├── 📁 data/
│   ├── raw/                  # 33 timestamped CSV files from API campaigns
│   ├── processed/            # air_quality_clean.csv · air_quality_scaled.csv
│   └── checkpoints/          # collection_progress.json (auto-resume)
│
├── 📁 src/
│   ├── data/
│   │   ├── collector.py      # Fault-tolerant API collector (retry + checkpoint)
│   │   ├── config.py         # Centralized config (cities, API, intervals)
│   │   └── data_preparation.py
│   │
│   ├── models/
│   │   └── model_building.py # Linear Regression · Random Forest · Decision Tree
│   │
│   └── federated/
│       ├── federated_manual.py       # Custom FedAvg from scratch
│       └── flower/                   # Flower client/server apps (Ray backend)
│           ├── client_app.py
│           ├── server_app.py
│           └── task.py
│
├── 📁 models/
│   ├── centralized/          # .pkl files: LR · RF regressor · RF classifier
│   └── federated/            # federated_model.pkl · encoders
│
└── 📁 visualizations/        # 5 auto-generated plots (PNG)
```
 
---
 
## ⚙️ Pipeline — Step by Step
 
### Step 1 · Data Collection
 
- **Source:** OpenWeatherMap API (Weather endpoint + Air Pollution endpoint)
- **Schedule:** Hourly collection over ~2 weeks (April 25 – May 9, 2026)
- **Variables:** temperature, humidity, pressure, wind speed, clouds, AQI, PM2.5, PM10, NO2, O3, SO2, CO
- **Resilience:** automatic retry (×3 per city), checkpoint-based resume on crash, daily rotating logs
```bash
python src/data/collector.py        # start collection
python src/data/resume_collector.py # resume from last checkpoint
```
 
**Result:** 996 records · 21 columns · 6 cities · 0% missing after preparation
 
---
 
### Step 2 · Data Preparation
 
| Step | Detail |
|------|--------|
| Deduplication | 3 duplicate rows removed |
| Cycle repair | 3 incomplete cycles → linear interpolation (39 values rebuilt) |
| Temporal features | `hour`, `day_of_week`, `day_of_month`, `month` recalculated from timestamp |
| Encoding | `LabelEncoder` on city name → `city_encoded` |
| Normalization | `StandardScaler` on 9 numerical features |
| Feature selection | 12 final features for modelling |
 
**Target variables:**
- `pm2_5` — regression (560 unique values)
- `aqi` — classification (3 classes: Good · Moderate · Degraded)
```bash
python src/data/data_preparation.py
```
 
---
 
### Step 3 · Centralized Modelling
 
#### Regression — PM2.5 Prediction
 
| Model | MAE (µg/m³) | RMSE (µg/m³) | R² |
|-------|------------|-------------|-----|
| Linear Regression | 0.839 | 1.118 | 0.802 |
| Decision Tree | 0.498 | 0.763 | 0.908 |
| **Random Forest** ✅ | **0.328** | **0.463** | **0.966** |
 
#### Classification — AQI Prediction (Random Forest)
 
| Metric | Score |
|--------|-------|
| Accuracy | **98.5%** |
| F1-score (weighted) | **0.983** |
| Classes predicted | Good (1) · Moderate (2) · Degraded (3) |
 
> ⚠️ **Analytical note:** PM10 accounts for ~83% of feature importance in the regression model.
> This high R² reflects the strong structural correlation between PM10 and PM2.5 (PM2.5 ⊂ PM10 physically).
> This is scientifically expected and documented, not a data leakage issue.
 
```bash
python src/models/model_building.py
```
 
---
 
### Step 4 · Federated Learning (Bonus)
 
Each city acts as an **independent federated client** — data never leaves the city, only model parameters are aggregated.
 
#### Implementation 1 — Custom FedAvg (from scratch)
 
- Algorithm: Federated Averaging (weighted by sample count)
- Model: Linear Regression
- Rounds: 5
| Model | R² |
|-------|----|
| Centralized Random Forest | 0.966 |
| Centralized Linear Regression | 0.802 |
| Federated Linear Regression (FedAvg) | 0.412 |
 
#### Implementation 2 — Flower Framework (flwr + Ray)
 
- 6 federated clients (1 per city) · 10 rounds · FedAvg strategy
- Backend: Ray simulation
**Per-city evaluation (global model):**
 
| City | MAE | R² |
|------|-----|-----|
| Agadir | 0.944 | 0.783 |
| Tanger | 0.950 | 0.771 |
| Marrakech | 1.148 | 0.698 |
| Rabat | 2.430 | 0.463 |
| Fès | 1.488 | -0.074 |
| Casablanca | 2.754 | -0.399 |
 
> **Key insight:** The variable per-city R² reveals significant **non-IID heterogeneity** — the global linear model works well for Agadir and Tanger but poorly for Casablanca and Fès, where local pollution dynamics differ. This is a known FedAvg limitation in non-homogeneous settings, and motivates future work with FedProx or personalized FL.
 
```bash
python src/federated/federated_manual.py   # manual implementation
cd src/federated/flower && flwr run . --stream  # Flower simulation
```
 
---
 
## 📊 Visualizations
 
| Plot | Description |
|------|-------------|
| `model_comparison.png` | MAE · RMSE · R² comparison across 3 regression models |
| `confusion_matrix.png` | AQI classification confusion matrix |
| `feature_importance.png` | Top 10 RF feature importances |
| `error_distribution.png` | Prediction error distribution (RF regressor) |
| `federated_comparison.png` | Centralized vs Federated R² · per-client coefficients |
 
---
 
## 🚀 Quick Start
 
```bash
# 1. Clone the repo
git clone https://github.com/your-username/air-quality-morocco.git
cd air-quality-morocco
 
# 2. Create environment
conda create -n air-quality python=3.11
conda activate air-quality
 
# 3. Install dependencies
pip install -r requirements.txt
 
# 4. Add your OpenWeatherMap API key in src/data/config.py
# API_KEY = "your_key_here"
 
# 5. Run the full pipeline
python src/data/collector.py          # Step 1: collect
python src/data/data_preparation.py   # Step 2: prepare
python src/models/model_building.py   # Step 3: train & evaluate
python src/federated/federated_manual.py  # Step 4: federated (manual)
```
 
---
 
## 📦 Requirements
 
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.6.1
matplotlib>=3.7.0
seaborn>=0.12.0
requests>=2.28.0
joblib>=1.3.0
flwr[simulation]>=1.28.0
imbalanced-learn>=0.11.0
```
 
---
 
## 🧠 Key Technical Decisions
 
**Why Random Forest over Linear Regression?**
RF captures non-linear interactions between co-occurring pollutants (e.g. NO2/SO2/CO patterns by city type) that linear models cannot model with a fixed 12-feature set.
 
**Why federated learning on city-partitioned data?**
Real-world air quality monitoring networks are decentralized — sensors belong to municipalities, agencies, or IoT operators. FL enables a shared predictive model without pooling sensitive environmental data, aligning with GDPR-style data sovereignty constraints.
 
**Why two FL implementations?**
The manual FedAvg shows the algorithm's mechanics clearly (educational value); Flower adds production-grade infrastructure (client/server separation, round management, Ray-based simulation) and validates that results are consistent with the ground-up implementation.
 
---
 
## 📈 Results Summary
 
```
╔══════════════════════════════════════════════════════╗
║           BEST MODEL PERFORMANCE SUMMARY             ║
╠══════════════════════════════════════════════════════╣
║  Regression  (PM2.5)  →  RF  R²=0.966  MAE=0.33    ║
║  Classification (AQI) →  RF  Accuracy=98.5%         ║
║  Federated (FedAvg)   →  LR  R²=0.412 (non-IID)    ║
╚══════════════════════════════════════════════════════╝
```
 
---
 
## 👩‍💻 Author
 
**WARDI Safae**
Master BDSI — Faculté des Sciences Dhar El Mahraz, USMBA Fès
 
---
 
## 📄 License
 
This project is open-source under the [MIT License](LICENSE).
 
---

