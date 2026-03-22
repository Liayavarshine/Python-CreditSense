import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    df = pd.read_csv("data/processed/processed_credit_data.csv")

    df.columns = df.columns.str.strip()
    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop("default", axis=1)
    y = df["default"]

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, "outputs/model.pkl")

    return model