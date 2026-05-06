# Real-Time State-Wise Sales Forecasting System 📈

An enterprise-grade, end-to-end **multi-state sales forecasting platform** that intelligently benchmarks multiple forecasting models and automatically selects the best-performing model for each region.

Designed for scalability, automation, and production deployment, this system combines classical statistical forecasting, machine learning, and deep learning into a unified forecasting pipeline.

---

# ✨ Key Highlights

* ⚡ Fully automated forecasting workflow
* 🧠 Intelligent per-state model selection
* 📊 Multi-model benchmarking engine
* 🔄 Recursive multi-step forecasting
* 🌍 Holiday-aware feature engineering
* 🚀 Production-ready FastAPI deployment
* 📈 Real-time forecast serving via REST API
* 💾 Model registry & performance tracking
* 🏭 Scalable architecture for enterprise use

---

# 🧩 Tech Stack

| Category                | Technologies            |
| ----------------------- | ----------------------- |
| Backend API             | FastAPI                 |
| Data Processing         | Pandas, NumPy           |
| Statistical Forecasting | ARIMA                   |
| Time-Series ML          | XGBoost                 |
| Deep Learning           | TensorFlow / Keras LSTM |
| Business Forecasting    | Prophet                 |
| Visualization           | Matplotlib              |
| Serialization           | Pickle, JSON            |
| Deployment Ready        | Uvicorn                 |

---

# 🏗️ System Architecture

```text
Raw Excel Data
       │
       ▼
Data Cleaning & Resampling
       │
       ▼
Feature Engineering
(Lags + Rolling Stats + Holidays)
       │
       ▼
Multi-Model Training
 ├── ARIMA
 ├── Prophet
 ├── XGBoost
 └── LSTM
       │
       ▼
Validation Benchmarking
(RMSE / MAE / MAPE)
       │
       ▼
Best Model Selection Per State
       │
       ▼
Model Registry Update
       │
       ▼
Production Forecast Generation
       │
       ▼
FastAPI Forecast Serving
```

---

# 🚀 Features

## ✅ Automated Data Processing

* Weekly sales aggregation
* Missing value handling
* Gap filling using interpolation
* State-wise segmentation
* Consistent time indexing

---

## ✅ Advanced Feature Engineering

The system automatically generates:

### Lag Features

* Lag 1
* Lag 7
* Lag 30

### Rolling Statistics

* Rolling Mean
* Rolling Standard Deviation

### Calendar Intelligence

* Indian holiday indicators
* Seasonal patterns
* Time-aware forecasting signals

---

## ✅ Multi-Model Forecasting Engine

The platform benchmarks four forecasting paradigms:

| Model   | Purpose                                |
| ------- | -------------------------------------- |
| ARIMA   | Statistical trend forecasting          |
| Prophet | Seasonality-aware business forecasting |
| XGBoost | Supervised ML forecasting              |
| LSTM    | Deep learning sequential modeling      |

---

## ✅ Smart Competitive Model Selection

For every individual state:

1. Models are trained independently
2. Predictions are generated on validation data
3. Metrics are calculated:

   * RMSE
   * MAE
   * MAPE
4. The best model is selected automatically
5. Winner is stored in the global registry

This enables:

* Region-specific intelligence
* Higher forecast accuracy
* Adaptive forecasting behavior

---

# 📊 Forecasting Workflow

```text
Input Sales Data
      │
      ▼
Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Train All Models
      │
      ▼
Validation Benchmarking
      │
      ▼
Select Best Model
      │
      ▼
Generate 8-Week Forecast
      │
      ▼
Serve Through API
```

---

# 📂 Project Structure

```text
forecasting_project/
│
├── api/
│   └── main.py
│
├── data/
│   ├── processed/
│   ├── forecasts.json
│   └── sales_data.xlsx
│
├── models/
│   ├── arima/
│   ├── prophet/
│   ├── xgboost/
│   └── lstm/
│
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train_arima.py
│   ├── train_prophet.py
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   ├── evaluate.py
│   └── forecast.py
│
├── model_registry.json
├── requirements.txt
├── run_pipeline.py
└── README.md
```

---

# ⚙️ Installation

## 🔹 Prerequisites

* Python 3.10+
* Git
* Excel dataset with:

  * `date`
  * `state`
  * `sales`

---

## 🔹 Clone Repository

```bash
git clone https://github.com/pvats2006/Multi-State-Sales-Forecasting-System.git
cd Multi-State-Sales-Forecasting-System
```

---

## 🔹 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔹 Add Dataset

Place your dataset here:

```text
data/sales_data.xlsx
```

---

# ▶️ Running the Complete Pipeline

Run the full automation pipeline:

```bash
python run_pipeline.py
```

This automatically performs:

* preprocessing
* feature engineering
* training
* benchmarking
* model selection
* forecasting
* registry updates

---

# 🛠️ Manual Execution

## 1️⃣ Data Preprocessing

```bash
python src/preprocess.py
```

---

## 2️⃣ Model Training & Benchmarking

```bash
python src/evaluate.py
```

---

## 3️⃣ Generate Production Forecasts

```bash
python src/forecast.py
```

---

## 4️⃣ Launch API Server

```bash
uvicorn api.main:app --reload --port 8000
```

API will run at:

```text
http://127.0.0.1:8000
```

---

# 📡 API Endpoints

## 🔹 Get Forecast

```bash
GET /forecast?state=California&weeks=8
```

Example:

```bash
curl "http://127.0.0.1:8000/forecast?state=California&weeks=8"
```

---

## 🔹 View Best Models

```bash
GET /models
```

Example:

```bash
curl "http://127.0.0.1:8000/models"
```

---

## 🔹 List Available States

```bash
GET /states
```

Example:

```bash
curl "http://127.0.0.1:8000/states"
```

---

# 🧠 Model Selection Strategy

The platform follows a **Validation-Based Competitive Benchmarking Framework**.

## Workflow

* Split data into:

  * Training Set
  * Validation Set (last 8 weeks)

* Train all models independently

* Forecast validation horizon

* Evaluate using:

  * RMSE
  * MAE
  * MAPE

* Select model with lowest RMSE

* Store winner in:

  ```text
  model_registry.json
  ```

---

# 🔄 Recursive Forecasting Engine

Supervised models like:

* XGBoost
* LSTM

use a custom recursive forecasting strategy where predicted values are fed back iteratively to generate future forecasts.

This enables:

* multi-step prediction
* dynamic future generation
* long-horizon forecasting

---

# 📈 Example Output

```json
{
  "state": "California",
  "best_model": "XGBoost",
  "forecast": [
    15230,
    15890,
    16120,
    16550,
    17010,
    17240,
    17680,
    18020
  ]
}
```

---

# 🔥 Enterprise Capabilities

✅ Automated forecasting pipeline
✅ Multi-region intelligence
✅ Scalable architecture
✅ Model registry system
✅ Production API serving
✅ Real-time inference support
✅ Extensible forecasting framework

---

# 🧪 Future Improvements

* Docker deployment
* CI/CD integration
* MLflow experiment tracking
* Automated retraining scheduler
* Drift detection
* Streamlit dashboard
* Kafka real-time ingestion
* Cloud deployment (AWS/GCP/Azure)

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Developed by **Priyanshu**
Passionate about:

* AI Engineering
* Forecasting Systems
* MLOps
* Data Engineering
* Production AI Systems

---

# ⭐ Support

If you found this project useful:

⭐ Star the repository
🍴 Fork the project
📢 Share with others
