# 📈 AI-Powered Stock Price Rise Prediction API

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

An intelligent stock market prediction system that combines **technical analysis**, **financial news sentiment**, and **machine learning** to predict significant stock price increases.

Instead of making a simple **Rise/Fall** prediction, the system classifies stocks into multiple gain categories while addressing class imbalance using **SMOTE** and incorporating financial sentiment through **FinBERT**.

---

# ✨ Features

* 📈 Multi-level stock price rise prediction
* 📰 Financial news sentiment analysis
* 🤖 FinBERT-based sentiment scoring
* 🌲 Random Forest classification model
* ⚖️ SMOTE for handling imbalanced datasets
* 📊 Technical indicators

  * Moving Average (5 Days)
  * Moving Average (10 Days)
  * Relative Strength Index (RSI)
  * MACD
* 📡 Live stock analysis using Yahoo Finance
* ⚡ REST API built with FastAPI
* 📉 Prediction confidence scores
* 📋 Feature importance analysis

---

# 🚀 Prediction Categories

| Class | Description               |
| ----- | ------------------------- |
| 0     | No Significant Gain (≤1%) |
| 1     | Moderate Gain (1–5%)      |
| 2     | Good Gain (5–7%)          |
| 3     | Strong Gain (>7%)         |

---

# 🏗️ System Architecture

```
                Historical Stock Data
                         │
                         ▼
              Technical Indicator Generation
         (MA5, MA10, RSI, MACD, Returns)
                         │
                         ▼
                Financial News Headlines
                         │
                         ▼
                  FinBERT Sentiment
                         │
                         ▼
              Feature Engineering
                         │
                         ▼
                Dataset Balancing
                     (SMOTE)
                         │
                         ▼
              Random Forest Training
                         │
                         ▼
               Prediction REST API
                         │
                         ▼
                  Stock Prediction
```

---

# 🛠 Technology Stack

## Backend

* Python
* FastAPI
* Pydantic

## Machine Learning

* Scikit-Learn
* Random Forest Classifier
* SMOTE
* NumPy
* Pandas

## Natural Language Processing

* HuggingFace Transformers
* FinBERT
* TextBlob

## Financial Data

* Yahoo Finance API
* RSS News Feeds

---

# 📂 Project Structure

```
stock-price-rise-prediction/

│
├── app/
│   ├── main.py
│   ├── routes.py
│   ├── training.py
│   ├── prediction.py
│   ├── indicators.py
│   ├── sentiment.py
│   ├── schemas.py
│   └── utils.py
│
├── docs/
│   ├── API_Documentation.md
│   ├── Project_Report.pdf
│   └── Final_Presentation.pdf
│
├── assets/
│   ├── architecture.png
│   └── banner.png
│
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

# 📊 Machine Learning Pipeline

1. Historical stock prices are collected.
2. Financial news headlines are gathered.
3. Technical indicators are computed.
4. News sentiment is extracted using FinBERT.
5. Features are merged into a training dataset.
6. SMOTE balances minority classes.
7. Random Forest model is trained.
8. Predictions are served through FastAPI.

---

# 🌐 API Endpoints

## Health Check

```
GET /
```

Returns application status.

---

## Train Model

```
POST /api/train
```

Uploads stock and news datasets and trains the model.

---

## Predict

```
POST /api/predict
```

Predicts the stock rise category based on technical indicators and sentiment.

---

## Live Stock Analysis

```
GET /api/analyze/{ticker}
```

Downloads live market data, analyzes recent financial news, computes technical indicators, and predicts the stock rise category.

Example:

```
GET /api/analyze/AAPL
```

---

## Model Status

```
GET /api/status
```

Returns training status and feature importance.

---

# 📈 Example Prediction

```json
{
  "prediction": 2,
  "label": "Good Gain (5–7%)",
  "probabilities": {
    "No significant gain (≤1%)": 0.06,
    "Moderate gain (1–5%)": 0.19,
    "Good gain (5–7%)": 0.67,
    "Strong gain (>7%)": 0.08
  }
}
```

---

# 📊 Technical Indicators Used

* Moving Average (5)
* Moving Average (10)
* Relative Strength Index (RSI)
* MACD
* Financial Sentiment Score

---

# 🔮 Future Improvements

* Docker deployment
* PostgreSQL database
* JWT Authentication
* Model persistence
* Continuous training pipeline
* Transformer-based prediction model
* LSTM implementation
* Explainable AI using SHAP
* Streamlit dashboard
* Cloud deployment (AWS/Azure)

---

# ⚠️ Limitations

* Requires sufficient historical stock data.
* Prediction quality depends on news availability.
* FinBERT inference may increase response time during training.
* Designed for educational and research purposes, not financial advice.

---

# 🤝 Contributing

Contributions are welcome.

If you'd like to improve the project, feel free to fork the repository and submit a pull request.

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Authors

Developed as a Machine Learning and Financial Analytics project using FastAPI, FinBERT, technical indicators, and Random Forest for multi-level stock price rise prediction.
