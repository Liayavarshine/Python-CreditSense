import joblib

model = joblib.load("outputs/model.pkl")

sample = [500000, 50000, 20000, 40, 100000, 0.5, 0.1, 0.2, 2]

risk_score = model.predict([sample])[0]

print("Predicted Risk Score:", risk_score)

if risk_score < 0.3:
    print("Low Risk → Approve")
elif risk_score < 0.7:
    print("Medium Risk → Conditional Approval")
else:
    print("High Risk → Reject")