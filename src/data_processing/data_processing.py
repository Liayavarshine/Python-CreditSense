import pandas as pd

def load_raw_data():
    df = pd.read_csv("data/raw/Loan.csv")
    return df


def clean_data(df):
    # Fill missing values
    df = df.fillna(method='ffill')

    # Drop unnecessary columns if needed
    df = df.drop_duplicates()

    return df


def save_processed(df):
    df.to_csv("data/processed/processed_credit_data.csv", index=False)