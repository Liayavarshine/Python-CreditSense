import joblib
import pandas as pd

# Load model
model = joblib.load("outputs/model.pkl")

# Expected columns (same as training)
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

# 🔥 Take file input from user
file_path = input("Enter CSV file path: ")

try:
    # Read CSV file
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Select required columns
    df = df[columns]

    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))

    # Predict for all rows
    predictions = model.predict(df)

    # Add results to dataframe
    df["Predicted_Risk"] = predictions

    # Decision logic
    def decision(score):
        if score < 0.3:
            return "Approve"
        elif score < 0.7:
            return "Conditional"
        else:
            return "Reject"

    df["Decision"] = df["Predicted_Risk"].apply(decision)

    print("\n📊 Results:")
    print(df)

    # Save output file
    output_path = "outputs/predictions.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Predictions saved to {output_path}")

except Exception as e:
    print("❌ Error:", e)