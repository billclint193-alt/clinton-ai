import os
import joblib
import numpy as np
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

app = FastAPI(
    title="Clinton AI",
    version="1.0"
)

model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


FEATURES = [
    "ret_1",
    "ret_3",
    "ema_gap",
    "rsi",
    "macd",
    "macd_signal",
    "atr_pct",
    "volume_z"
]


class TradingViewAlert(BaseModel):
    pair: str
    timeframe: str = "15m"
    close: float = Field(gt=0)

    ret_1: float
    ret_3: float
    ema_gap: float
    rsi: float
    macd: float
    macd_signal: float
    atr_pct: float
    volume_z: float


journal = []


def predict(alert: TradingViewAlert):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="AI model has not been trained yet."
        )

    data = np.array([
        [getattr(alert, feature) for feature in FEATURES]
    ])

    probability_up = float(
        model.predict_proba(data)[0][1]
    )

    if probability_up >= 0.60:

        direction = "BULLISH"
        confidence = probability_up

    elif probability_up <= 0.40:

        direction = "BEARISH"
        confidence = 1 - probability_up

    else:

        direction = "NEUTRAL"
        confidence = 0.50

    return direction, round(confidence * 100, 1)


@app.get("/")
def home():

    return {
        "bot": "CLINTON AI",
        "status": "ONLINE",
        "mode": "PAPER_TRADING",
        "model_loaded": model is not None
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/webhook/tradingview")
def tradingview_webhook(alert: TradingViewAlert):

    direction, confidence = predict(alert)

    signal = {

        "pair": alert.pair.upper(),

        "direction": direction,

        "timeframe": alert.timeframe,

        "status": (
            "WAITING"
            if direction != "NEUTRAL"
            else "CANCELLED"
        ),

        "confidence": confidence,

        "created_at":
            datetime.now(timezone.utc).isoformat(),

        "mode": "PAPER_TRADING"

    }

    journal.append(signal)

    return {

        "bot": "CLINTON AI",

        "signal": signal

    }


@app.get("/signals")
def get_signals():

    return {

        "count": len(journal),

        "signals": journal[-100:]

    }


@app.post("/paper/confirm/{index}")
def confirm_signal(index: int):

    if index < 0 or index >= len(journal):

        raise HTTPException(
            status_code=404,
            detail="Signal not found"
        )

    journal[index]["status"] = "CONFIRMED"

    return journal[index]


@app.post("/paper/close/{index}")
def close_signal(
    index: int,
    result: str = "CLOSED"
):

    if index < 0 or index >= len(journal):

        raise HTTPException(
            status_code=404,
            detail="Signal not found"
        )

    allowed = {
        "IN_PROFIT",
        "CLOSED",
        "CANCELLED",
        "EXPIRED"
    }

    if result not in allowed:

        raise HTTPException(
            status_code=400,
            detail="Invalid status"
        )

    journal[index]["status"] = result

    return journal[index]
