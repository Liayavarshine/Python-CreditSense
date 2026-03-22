import pandas as pd

def make_decision(risk_score):
    if risk_score < 0.4:
        return "Loan Approved"
    elif risk_score < 0.7:
        return "Loan Review Required"
    else:
        return "Loan Rejected"

def run_decision_engine():
    df = pd.read_csv("outputs/predictions.csv")

    df["Decision"] = df["risk_score"].apply(make_decision)

    print(df[["risk_score", "Decision"]])

    df.to_csv("outputs/final_decisions.csv", index=False)

if __name__ == "__main__":
    run_decision_engine()