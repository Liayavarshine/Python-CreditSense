from src.data_processing.data_processing import load_raw_data, clean_data, save_processed
from src.model.ml_models import predict
import joblib

# Step 1: Load & clean data
df = load_raw_data()
df = clean_data(df)
save_processed(df)

# Step 2: Load trained model
model = joblib.load("outputs/model.pkl")

# Step 3: Sample input
sample = [5000, 2000, 150, 360]

result = predict(model, sample)

if result == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")