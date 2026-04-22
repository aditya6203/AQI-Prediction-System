import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df = df.dropna(subset=["AQI"])

    features = [
        "PM2.5", "PM10", "NO", "NO2",
        "NOx", "NH3", "CO", "SO2", "O3"
    ]

    df = df[features + ["AQI"]]

    df.fillna(df.mean(), inplace=True)

    return df
