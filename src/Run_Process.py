import os
import pandas as pd
from PreProcessing import preprocess_data

# Ensure processed folder exists
os.makedirs(r'C:\Users\DELL\movie-rating-predictor\data\processed', exist_ok=True)

# Load raw data
df = pd.read_csv(r'C:\Users\DELL\movie-rating-predictor\data\IMDb Movies India.csv', encoding='latin1')

print("🧾 Available columns in dataset:")
print(df.columns)

# Apply preprocessing
df_clean = preprocess_data(df)

# Save the cleaned data
df_clean.to_csv(r'C:\Users\DELL\movie-rating-predictor\data\processed\movies_clean.csv', index=False)

print("✅ Preprocessing complete. Cleaned data saved to: data/processed/movies_clean.csv")