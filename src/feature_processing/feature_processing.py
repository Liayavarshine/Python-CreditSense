def select_features(df):
    df.columns = df.columns.str.strip()
    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop("default", axis=1)  # all input features
    y = df["default"]              # target

    return X, y