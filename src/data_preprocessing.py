import os
import pandas as pd
import numpy as np

def load_and_preprocess_data(filepath=None):
    """
    Loads raw movie dataset, cleans missing values, and performs feature engineering.
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'movies_dataset.csv')
        
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # 1. Handle missing values
    budget_median = df['Budget_USD_Millions'].median()
    df['Budget_USD_Millions'] = df['Budget_USD_Millions'].fillna(budget_median)
    
    metascore_median = df['Metascore'].median()
    df['Metascore'] = df['Metascore'].fillna(metascore_median)
    
    # 2. Feature Engineering
    df['Profit_USD_Millions'] = (df['Revenue_USD_Millions'] - df['Budget_USD_Millions']).round(2)
    df['ROI'] = (df['Revenue_USD_Millions'] / np.maximum(df['Budget_USD_Millions'], 0.1)).round(2)
    
    # Rating Categories
    bins = [0, 5.5, 7.0, 8.2, 10.0]
    labels = ['Low (<5.5)', 'Average (5.5-7.0)', 'High (7.0-8.2)', 'Masterpiece (>8.2)']
    df['Rating_Category'] = pd.cut(df['IMDb_Rating'], bins=bins, labels=labels)
    
    # Release Decade
    df['Decade'] = (df['Release_Year'] // 10 * 10).astype(str) + 's'
    
    # Is High Budget Flag
    df['Is_High_Budget'] = (df['Budget_USD_Millions'] > budget_median).astype(int)
    
    print("[SUCCESS] Data preprocessing & feature engineering completed.")
    return df

def prepare_features_for_ml(df):
    """
    Prepares feature matrix X and target y for rating prediction ML models.
    """
    feature_cols = ['Release_Year', 'Duration_Min', 'Budget_USD_Millions', 'Metascore', 'Votes']
    
    # Encode categorical features: Genre, Director
    encoded_genres = pd.get_dummies(df['Genre'], prefix='Genre', drop_first=True)
    encoded_directors = pd.get_dummies(df['Director'], prefix='Director', drop_first=True)
    
    X = pd.concat([df[feature_cols], encoded_genres, encoded_directors], axis=1)
    y = df['IMDb_Rating']
    
    return X, y

if __name__ == '__main__':
    df = load_and_preprocess_data()
    print("Preprocessed Data Sample:")
    print(df[['Title', 'Genre', 'IMDb_Rating', 'Rating_Category', 'Profit_USD_Millions', 'ROI']].head())
