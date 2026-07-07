from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import io
import os

app = FastAPI(title="Stock Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
model_state = {
    "model": None,
    "trained": False,
    "report": None,
    "feature_importance": None,
}

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    MA_5: float
    MA_10: float
    RSI: float
    sentiment: float

class TrainResponse(BaseModel):
    message: str
    report: dict
    feature_importance: dict

class PredictResponse(BaseModel):
    prediction: int
    label: str
    probabilities: dict

# ── Label map ─────────────────────────────────────────────────────────────────

LABELS = {
    0: "No significant gain (≤1%)",
    1: "Moderate gain (1–5%)",
    2: "Good gain (5–7%)",
    3: "Strong gain (>7%)",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_indicators(stock_df: pd.DataFrame) -> pd.DataFrame:
    stock_df['MA_5'] = stock_df['Close'].rolling(window=5).mean()
    stock_df['MA_10'] = stock_df['Close'].rolling(window=10).mean()

    delta = stock_df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    stock_df['RSI'] = 100 - (100 / (1 + rs))
    return stock_df

def classify_return(x):
    if x > 0.07:
        return 3
    elif x > 0.05:
        return 2
    elif x > 0.01:
        return 1
    else:
        return 0

def get_sentiment_score(text: str) -> float:
    """FinBERT sentiment. Falls back to 0.5 if transformers not available."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        bert = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = bert(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        return float(scores.detach().numpy()[0][0])
    except Exception:
        return 0.5  # neutral fallback

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "trained": model_state["trained"]}

@app.post("/api/train", response_model=TrainResponse)
async def train(
    stock_file: UploadFile = File(...),
    news_file: UploadFile = File(...),
):
    try:
        stock_df = pd.read_csv(io.StringIO((await stock_file.read()).decode()))
        news_df = pd.read_csv(io.StringIO((await news_file.read()).decode()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")

    # Preprocess
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')
    stock_df = stock_df.sort_values('Date').dropna()
    news_df = news_df.dropna()

    # Sentiment
    if 'headline' in news_df.columns:
        news_df['sentiment'] = news_df['headline'].apply(get_sentiment_score)
        avg_sentiment = news_df['sentiment'].mean()
    else:
        avg_sentiment = 0.5

    # Technical indicators
    stock_df = compute_indicators(stock_df)

    # Merge & target
    data = stock_df.copy()
    data['sentiment'] = avg_sentiment
    data = data.dropna()
    data['future_return'] = data['Close'].pct_change().shift(-1)
    data['target'] = data['future_return'].apply(classify_return)
    data = data.dropna()

    X = data[['MA_5', 'MA_10', 'RSI', 'sentiment']]
    y = data['target']

    if len(X) < 20:
        raise HTTPException(status_code=400, detail="Not enough data to train (need ≥20 rows after preprocessing).")

    # SMOTE
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
    except Exception:
        X_res, y_res = X, y  # fallback if imbalanced-learn not installed

    # Train
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    model_state["model"] = clf
    model_state["trained"] = True
    model_state["report"] = report
    model_state["feature_importance"] = dict(zip(
        ['MA_5', 'MA_10', 'RSI', 'sentiment'],
        clf.feature_importances_.tolist()
    ))

    return TrainResponse(
        message="Model trained successfully.",
        report=report,
        feature_importance=model_state["feature_importance"],
    )

@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not model_state["trained"]:
        raise HTTPException(status_code=400, detail="Model not trained yet. Upload CSVs first.")

    clf = model_state["model"]
    input_data = [[req.MA_5, req.MA_10, req.RSI, req.sentiment]]
    pred = int(clf.predict(input_data)[0])
    proba = clf.predict_proba(input_data)[0]
    classes = clf.classes_.tolist()

    return PredictResponse(
        prediction=pred,
        label=LABELS.get(pred, "Unknown"),
        probabilities={str(LABELS.get(c, c)): round(float(p), 4) for c, p in zip(classes, proba)},
    )

@app.get("/api/status")
def status():
    return {
        "trained": model_state["trained"],
        "feature_importance": model_state["feature_importance"],
    }

# ── Auto-analyze by ticker ────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    ticker: str
    current_price: float
    MA_5: float
    MA_10: float
    RSI: float
    MACD: float
    sentiment: float
    sentiment_label: str
    news_headlines: list
    prediction: int
    label: str
    probabilities: dict

@app.get("/api/analyze/{ticker}", response_model=AnalyzeResponse)
def analyze(ticker: str):
    if not model_state["trained"]:
        raise HTTPException(status_code=400, detail="Model not trained yet. Upload CSVs and train first.")

    import yfinance as yf
    import feedparser

    ticker = ticker.upper().strip()

    # ── Fetch price history ──
    try:
        hist = yf.download(ticker, period="60d", interval="1d", progress=False, auto_adjust=True)
        if hist.empty or len(hist) < 15:
            raise HTTPException(status_code=404, detail=f"Not enough price data for '{ticker}'. Check the ticker symbol.")
        # Flatten MultiIndex columns if present
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist = hist.reset_index()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch price data: {e}")

    # ── Compute indicators ──
    close = hist['Close'].astype(float)

    def safe_float(val, default=0.0):
        """Convert to float, replacing NaN/inf with a safe default."""
        import math
        try:
            v = float(val)
            return default if (math.isnan(v) or math.isinf(v)) else v
        except Exception:
            return default

    ma5  = safe_float(close.rolling(5).mean().iloc[-1])
    ma10 = safe_float(close.rolling(10).mean().iloc[-1])

    delta    = close.diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    rs       = gain / loss
    rsi      = safe_float((100 - (100 / (1 + rs))).iloc[-1], default=50.0)

    ema12    = close.ewm(span=12, adjust=False).mean()
    ema26    = close.ewm(span=26, adjust=False).mean()
    macd     = safe_float((ema12 - ema26).iloc[-1])

    current_price = safe_float(close.iloc[-1])

    # ── Fetch news & sentiment ──
    headlines = []
    sentiment_score = 0.5
    sentiment_label = "Neutral"

    # Try multiple news sources for the ticker
    rss_urls = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://finance.yahoo.com/rss/headline?s={ticker}",
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}",
    ]

    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            if feed.entries:
                headlines = [e.title for e in feed.entries[:8]]
                break
        except Exception:
            continue

    if headlines:
        try:
            from textblob import TextBlob
            polarities = [TextBlob(h).sentiment.polarity for h in headlines]
            avg_polarity = sum(polarities) / len(polarities)
            # Store raw polarity (-1..+1) directly as sentiment_score for display
            sentiment_score = round(avg_polarity, 3)
        except Exception:
            sentiment_score = 0.0

    import math
    if math.isnan(sentiment_score) or math.isinf(sentiment_score):
        sentiment_score = 0.0

    # Map to label using raw polarity (-1..+1)
    if sentiment_score > 0.05:
        sentiment_label = "Positive"
    elif sentiment_score < -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    # Normalize to 0..1 only for model input
    sentiment_for_model = (sentiment_score + 1) / 2

    # ── Predict ──
    clf = model_state["model"]
    input_data = [[ma5, ma10, rsi, sentiment_for_model]]
    pred  = int(clf.predict(input_data)[0])
    proba = clf.predict_proba(input_data)[0]
    classes = clf.classes_.tolist()

    return AnalyzeResponse(
        ticker=ticker,
        current_price=round(current_price, 2),
        MA_5=round(ma5, 2),
        MA_10=round(ma10, 2),
        RSI=round(rsi, 2),
        MACD=round(macd, 3),
        sentiment=round(sentiment_score, 3),
        sentiment_label=sentiment_label,
        news_headlines=headlines,
        prediction=pred,
        label=LABELS.get(pred, "Unknown"),
        probabilities={str(LABELS.get(c, c)): round(float(p), 4) for c, p in zip(classes, proba)},
    )
