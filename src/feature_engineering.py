def add_features(df):
    df = df.copy()

    # Director average rating
    director_avg = df.groupby('director')['rating'].mean()
    df['director_success'] = df['director'].map(director_avg)

    # Genre average rating
    genre_avg = df.groupby('genre')['rating'].mean()
    df['genre_success'] = df['genre'].map(genre_avg)

    # ROI feature
    df['roi'] = df['box_office'] / (df['budget'] + 1)

    return df
   
