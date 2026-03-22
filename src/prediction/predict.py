import pandas as pd
import joblib

def generate_predictions():

    # load trained model
    model = joblib.load("outputs/model.pkl")

    # load dataset
    df = pd.read_csv("data/processed/processed_credit_data.csv")

    df.columns = df.columns.str.strip()

    # handle missing values
    df = df.fillna(df.mean(numeric_only=True))

    # remove target column
    X = df.drop("default", axis=1)

    # generate predictions
    predictions = model.predict(X)

    # create output dataframe
    output = pd.DataFrame()
    output["risk_score"] = predictions

    # save predictions
    output.to_csv("outputs/predictions.csv", index=False)

    print("Predictions generated!")

if __name__ == "__main__":
    generate_predictions()