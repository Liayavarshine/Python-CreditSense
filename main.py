import joblib

model = joblib.load("outputs/model.pkl")

# Example input (same order as dataset)
sample = [500000, 50000, 20000, 40, 100000, 0.5, 0.1, 0.2, 2]

result = model.predict([sample])

if result[0] == 0:
    print("Safe (No Default)")
else:
    print("Risky (Default)")