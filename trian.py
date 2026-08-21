import os
import sys
import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

from features import make_features, FEATURES


DATA_URL = os.getenv("DATA_URL")

if not DATA_URL:
    print("DATA_URL environment variable is missing.")
    print("Set DATA_URL to your approved historical OHLCV CSV.")
    raise SystemExit(1)


print("Downloading historical market data...")

df = pd.read_csv(DATA_URL)

print(f"Downloaded {len(df)} candles.")


required = {
    "open",
    "high",
    "low",
    "close",
    "volume"
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Dataset is missing columns: {sorted(missing)}"
    )


print("Creating future-price target...")

df["future_return"] = (
    df["close"].shift(-1) / df["close"] - 1
)

df["target"] = (
    df["future_return"] > 0
).astype(int)


print("Creating AI features...")

df = make_features(df)

df = df.dropna(
    subset=FEATURES + ["target"]
).reset_index(drop=True)


if len(df) < 500:
    raise ValueError(
        "Not enough usable candles. "
        "At least 500 are required; thousands are preferable."
    )


# Time-based split.
# We deliberately DO NOT shuffle financial time-series data.

split = int(len(df) * 0.80)

train_data = df.iloc[:split]
test_data = df.iloc[split:]


print(
    f"Training candles: {len(train_data)}"
)

print(
    f"Testing candles: {len(test_data)}"
)


print("Training Clinton AI...")


model = HistGradientBoostingClassifier(
    max_iter=250,
    learning_rate=0.05,
    max_leaf_nodes=15,
    random_state=42
)


model.fit(
    train_data[FEATURES],
    train_data["target"]
)


print("Testing on unseen data...")


predictions = model.predict(
    test_data[FEATURES]
)


accuracy = accuracy_score(
    test_data["target"],
    predictions
)


print(
    f"Out-of-sample accuracy: {accuracy:.4f}"
)


print("\nClassification report:")

print(
    classification_report(
        test_data["target"],
        predictions,
        digits=3
    )
)


joblib.dump(
    model,
    "model.joblib"
)


print("\n✅ Clinton AI model trained.")
print("Saved as model.joblib")
