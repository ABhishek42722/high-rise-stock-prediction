# high-rise-stock-prediction
AI-powered stock price rise prediction using financial sentiment analysis, technical indicators, and machine learning.

🚀 Stock Price Rise Prediction using FinBERT & Machine Learning

This project is an end-to-end stock price prediction API built with FastAPI that combines technical indicators, financial news sentiment, and machine learning to predict significant stock price movements.

Unlike traditional stock prediction systems that only predict whether prices rise or fall, this application predicts multiple price increase levels while addressing class imbalance using SMOTE.

Features:

📈 Predicts stock rise categories
📰 Financial news sentiment analysis
🤖 FinBERT integration
🌲 Random Forest classifier
⚖️ SMOTE oversampling
📊 Technical indicators
RSI
Moving Average (5)
Moving Average (10)
MACD
⚡ FastAPI REST API
📡 Live stock analysis using Yahoo Finance
🧠 Prediction confidence scores

Tech Stack:

Python
FastAPI
Scikit-learn
Pandas
NumPy
HuggingFace Transformers
FinBERT
imbalanced-learn
yfinance
TextBlob

Architecture:

Yahoo Finance
        │
        ▼
Historical Prices
        │
        ▼
Technical Indicators
        │
        ▼
Financial News
        │
        ▼
FinBERT Sentiment
        │
        ▼
SMOTE
        │
        ▼
Random Forest
        │
        ▼
Stock Prediction API

API Endpoints:

Method	Endpoint	Description
GET	/	Health Check
POST	/api/train	Train Model
POST	/api/predict	Predict Stock Rise
GET	/api/status	Model Status
GET	/api/analyze/{ticker}	Analyze Live Stock

Project Structure:

stock-price-rise-prediction
│
├── app
│   ├── main.py
│   ├── indicators.py
│   ├── sentiment.py
│   ├── training.py
│   └── schemas.py
│
├── docs
│
├── assets
│
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore

Future Improvements:

Docker support
Authentication
Model persistence
CI/CD
Unit testing
Cloud deployment
Real-time prediction dashboard
