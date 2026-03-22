import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    df = pd.read_csv("data/processed/processed_credit_data.csv")

    # Convert categorical target
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    X = df[[
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term"
    ]]
    
    y = df["Loan_Status"]

    model = LogisticRegression()
    model.fit(X, y)
    
    joblib.dump(model, "outputs/model.pkl")

    return model


def predict(model, data):
    return model.predict([data])[0]