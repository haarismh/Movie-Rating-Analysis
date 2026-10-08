import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_rating_prediction_models(X, y, output_dir=None):
    """
    Trains machine learning models (Random Forest, Ridge) to predict movie IMDb ratings.
    Calculates evaluation metrics and feature importances.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
    os.makedirs(output_dir, exist_ok=True)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n" + "="*50)
    print("      MACHINE LEARNING RATING PREDICTOR RESULTS     ")
    print("="*50)
    print(f"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")
    
    # 1. Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_r2 = r2_score(y_test, y_pred_rf)
    
    # 2. Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    
    ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    ridge_r2 = r2_score(y_test, y_pred_ridge)
    
    print("\n--- Model Performance Metrics ---")
    metrics_df = pd.DataFrame({
        'Model': ['Random Forest Regressor', 'Ridge Regression'],
        'MAE': [rf_mae, ridge_mae],
        'RMSE': [rf_rmse, ridge_rmse],
        'R2 Score': [rf_r2, ridge_r2]
    })
    print(metrics_df.to_string(index=False))
    
    # 3. Feature Importance (Random Forest)
    importances = rf_model.feature_importances_
    feature_names = X.columns
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, hue=feat_imp.index, palette='viridis', legend=False)
    plt.title('Top 10 Feature Importances for Movie Rating Prediction', fontsize=14, fontweight='bold')
    plt.xlabel('Relative Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    chart_path = os.path.join(output_dir, '06_feature_importance.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[SAVED] Feature Importance Chart saved to '{chart_path}'")
    
    return rf_model, X.columns

def predict_single_movie(rf_model, feature_columns, duration_min, budget_mb, metascore, votes, genre, director, release_year=2025):
    """
    Predicts IMDb rating for a custom user-defined movie.
    """
    # Create single row dataframe matching feature columns
    input_data = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    
    input_data['Release_Year'] = release_year
    input_data['Duration_Min'] = duration_min
    input_data['Budget_USD_Millions'] = budget_mb
    input_data['Metascore'] = metascore
    input_data['Votes'] = votes
    
    genre_col = f"Genre_{genre}"
    if genre_col in input_data.columns:
        input_data[genre_col] = 1.0
        
    director_col = f"Director_{director}"
    if director_col in input_data.columns:
        input_data[director_col] = 1.0
        
    predicted_rating = rf_model.predict(input_data)[0]
    return round(predicted_rating, 2)

if __name__ == '__main__':
    from data_preprocessing import load_and_preprocess_data, prepare_features_for_ml
    df = load_and_preprocess_data()
    X, y = prepare_features_for_ml(df)
    model, feature_cols = train_rating_prediction_models(X, y)
    
    # Test sample prediction
    pred = predict_single_movie(model, feature_cols, duration_min=150, budget_mb=180, metascore=85, votes=500000, genre='Sci-Fi', director='Christopher Nolan')
    print(f"\n[DEMO PREDICTION] Predicted IMDb Rating for Nolan's Sci-Fi Epic: {pred} / 10.0")
