def select_features(df):
    df = df.fillna(df.mean(numeric_only=True))
    
    df["Loan_Status"] = df["Loan_Status"].map({'Y': 1, 'N': 0})

    features = df[[
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History"
    ]]

    target = df["Loan_Status"]

    return features, target