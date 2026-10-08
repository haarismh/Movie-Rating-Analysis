import os
import sys

# Ensure src module is discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# pyrefly: ignore [missing-import]
from dataset_generator import generate_movie_dataset
# pyrefly: ignore [missing-import]
from data_preprocessing import load_and_preprocess_data, prepare_features_for_ml
# pyrefly: ignore [missing-import]
from eda import generate_eda_reports_and_charts
# pyrefly: ignore [missing-import]
from ml_model import train_rating_prediction_models, predict_single_movie

def run_pipeline():
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'movies_dataset.csv')
    
    # Step 1: Dataset Verification/Generation
    if not os.path.exists(dataset_path):
        print("[STEP 1/4] Dataset missing. Generating synthetic movie dataset...")
        generate_movie_dataset(num_samples=1000)
    else:
        print(f"[STEP 1/4] Found dataset at '{dataset_path}'")
        
    # Step 2: Data Preprocessing
    print("\n[STEP 2/4] Preprocessing data and engineering features...")
    df = load_and_preprocess_data(dataset_path)
    
    # Step 3: Exploratory Data Analysis & Visualizations
    print("\n[STEP 3/4] Running Exploratory Data Analysis (EDA)...")
    generate_eda_reports_and_charts(df)
    
    # Step 4: Machine Learning Model Training
    print("\n[STEP 4/4] Training Machine Learning Rating Predictors...")
    X, y = prepare_features_for_ml(df)
    model, feature_cols = train_rating_prediction_models(X, y)
    
    print("\n" + "="*60)
    print("      PROJECT EXECUTION & MODEL TRAINING COMPLETE!      ")
    print("="*60)
    
    # Quick Interactive Custom Prediction Demo
    print("\n--- [DEMO] Sample Custom Rating Prediction ---")
    custom_rating = predict_single_movie(
        rf_model=model,
        feature_columns=feature_cols,
        duration_min=148,
        budget_mb=160,
        metascore=88,
        votes=450000,
        genre='Sci-Fi',
        director='Christopher Nolan'
    )
    print(f"Movie Details: Christopher Nolan | Sci-Fi | Budget: $160M | Metascore: 88")
    print(f"Predicted IMDb Rating: {custom_rating} / 10.0")
    print("="*60)
    print("All charts exported to 'charts/' directory.")

if __name__ == '__main__':
    run_pipeline()
