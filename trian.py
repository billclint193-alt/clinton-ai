import sys
import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

from features import make_features, FEATURES


if len(sys.argv) != 2:

    print("Usage:")
    print("python train.py historical.csv")

    raise SystemExit(1)


file_path = sys.argv[1]

print("Loading historical market data...")

df = pd.read_csv(file_path)


# Make the target from the NEXT candle.
# 1 = price goes up
# 0 = price does not go up

df["future_return"] = (
    df["close"].shift(-1)
    / df["close"]
    - 1
)

df["target"] = (
    df["future_return"] > 0
).astype(int)


print("Creating market features...")

df = make_features(df)


df = df.dropna(
    subset=FEATURES + ["target"]
).reset_index(drop=True)


if len(df) < 500:

    raise ValueError(
        "Not enough historical candles. "
        "Use at least 500 candles; thousands are better."
    )


# IMPORTANT:
# We split chronologically.
# We NEVER randomly shuffle financial data.

split = int(
    len(df) * 0.80
)


train_data = df.iloc[:split]

test_data = df.iloc[split:]


print(
    f"Training candles: {len(train_data)}"
)

print(
    f"Testing candles: {len(test_data)}"
)


model = HistGradientBoostingClassifier(

    max_iter=250,

    learning_rate=0.05,

    max_leaf_nodes=15,

    random_state=42
)


print("Training Clinton AI...")


model.fit(

    train_data[FEATURES],

    train_data["target"]
)


print("Testing Clinton AI...")


predictions = model.predict(
    test_data[FEATURES]
)


accuracy = accuracy_score(

    test_data["target"],

    predictions
)


print(
    "Out-of-sample accuracy:",
    round(accuracy, 4)
)


print("\nDetailed results:")

print(
    classification_report(
        test_data["target"],
        predictions,
        digits=3
    )
)


# Save the trained AI model.

joblib.dump(
    model,
    "model.joblib"
)


print("\nClinton AI model saved as:")
print("model.joblib")
