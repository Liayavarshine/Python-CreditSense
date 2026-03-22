import joblib
import pandas as pd

model = joblib.load("outputs/model.pkl")

# Use SAME column names as training
columns = [
    "revenue",
    "profit",
    "balance",
    "transactions",
    "loan_amount",
    "credit_utilization",
    "profit_ratio",
    "transaction_ratio",
    "risk_score"
]

data = pd.DataFrame([[500000, 50000, 20000, 40, 100000, 0.5, 0.1, 0.2, 2]],
                    columns=columns)

risk_score = model.predict(data)[0]

print("Predicted Risk Score:", risk_score)