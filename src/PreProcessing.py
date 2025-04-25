import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Check if 'budget' exists before filling
    if 'budget' in df.columns:
        df['budget'].fillna(df['budget'].median(), inplace=True)

    # Check if 'genre' exists before encoding
    if 'genre' in df.columns:
        df['genre'] = df['genre'].astype(str)

    # Convert release_date to datetime if exists
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    return df
