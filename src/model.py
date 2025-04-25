import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def add_features(df):
    # Fill missing values in 'Director' and 'Genre' columns
    df['Director'] = df['Director'].fillna('Unknown')
    df['Genre'] = df['Genre'].fillna('Unknown')

    # Ensure 'Duration' is numeric and handle non-numeric values
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    
    # Check for NaN values in 'Duration'
    if df['Duration'].isnull().sum() > 0:
        print(f"Warning: 'Duration' column contains {df['Duration'].isnull().sum()} NaN values.")

    # If 'Duration' column is entirely NaN, fill it with a default value
    if df['Duration'].isnull().sum() == len(df):
        print("Warning: 'Duration' column contains only NaN values. Filling with default value of 0.")
        df['Duration'] = 0
    else:
        # Replace NaN in 'Duration' with the median value
        df['Duration'] = df['Duration'].fillna(df['Duration'].median())  # Handle NaN in 'Duration'

    # Encode categorical variables
    df['Genre'] = df['Genre'].astype('category').cat.codes
    df['Director'] = df['Director'].astype('category').cat.codes

    # Example feature: Director success = average rating per director
    director_avg = df.groupby('Director')['Rating'].transform('mean')
    df['Director_success'] = director_avg

    # Example feature: Genre success = average rating per genre
    genre_avg = df.groupby('Genre')['Rating'].transform('mean')
    df['Genre_success'] = genre_avg

    return df

def train_model():
    df = pd.read_csv(r'C:\Users\DELL\movie-rating-predictor\data\processed\movies_clean.csv', encoding='latin1')

    # Handle missing values in the target variable 'Rating'
    df['Rating'] = df['Rating'].fillna(df['Rating'].median())  # Fill missing ratings with median value

    # Handle missing values in the features
    df['Director'] = df['Director'].fillna('Unknown')
    df['Genre'] = df['Genre'].fillna('Unknown')

    # Ensure 'Duration' is numeric and handle invalid values
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    
    # Check for NaN values in 'Duration'
    if df['Duration'].isnull().sum() > 0:
        print(f"Warning: 'Duration' column contains {df['Duration'].isnull().sum()} NaN values.")
    
    # Handle the case where 'Duration' column might be entirely NaN
    if df['Duration'].isnull().sum() == len(df):
        print("Warning: 'Duration' column contains only NaN values. Filling with default value of 0.")
        df['Duration'] = 0
    else:
        # Fill any remaining NaN values in 'Duration' with the median value
        df['Duration'] = df['Duration'].fillna(df['Duration'].median())

    # Feature engineering
    df = add_features(df)

    # Selecting the features and target variable
    X = df[['Duration', 'Genre', 'Director', 'Director_success', 'Genre_success']]
    y = df['Rating']

    # Check if there are still NaN values in the features or target
    if X.isnull().any().any() or y.isnull().any():
        print("Warning: Data still contains NaN values after handling.")
        print(X.isnull().sum())  # Show NaN count per feature
        print(y.isnull().sum())  # Show NaN count in target variable
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use GradientBoostingRegressor instead of RandomForestRegressor
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5 
    r2 = r2_score(y_test, y_pred)

    # Print the evaluation metrics
    print(f"MAE (Mean Absolute Error): {mae}")
    print(f"RMSE (Root Mean Squared Error): {rmse}")
    print(f"R2 Score: {r2}")

    # Save model
    joblib.dump(model, r'C:\Users\DELL\movie-rating-predictor\models\movie_rating_model.pkl')
    print("✅ Model saved successfully.")

if __name__ == '__main__':
    train_model()
