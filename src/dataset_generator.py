import os
import numpy as np
import pandas as pd

def generate_movie_dataset(num_samples=1000, seed=42):
    """
    Generates a realistic movie dataset for rating analysis and machine learning tasks.
    """
    np.random.seed(seed)
    
    genres_list = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Romance', 'Thriller', 'Animation', 'Adventure', 'Crime']
    directors_list = ['Christopher Nolan', 'Steven Spielberg', 'Martin Scorsese', 'Quentin Tarantino', 
                      'Greta Gerwig', 'Denis Villeneuve', 'James Cameron', 'Guillermo del Toro', 
                      'Ridley Scott', 'Ava DuVernay', 'Indie Director']
    
    movie_adjectives = ['The Last', 'Eternal', 'Shadow of', 'Dark', 'Silent', 'Golden', 'Lost', 'Future', 'Cosmic', 'Rising']
    movie_nouns = ['Horizon', 'Kingdom', 'Dream', 'Protocol', 'Legacy', 'Echo', 'Journey', 'Vengeance', 'City', 'Signal']
    
    titles = [f"{np.random.choice(movie_adjectives)} {np.random.choice(movie_nouns)}" for _ in range(num_samples)]
    # Ensure unique titles
    titles = list(dict.fromkeys(titles))
    while len(titles) < num_samples:
        titles.append(f"Project {len(titles) + 1}")

    release_years = np.random.randint(1990, 2025, size=num_samples)
    durations = np.random.normal(loc=115, scale=20, size=num_samples).clip(75, 210).round().astype(int)
    
    primary_genres = np.random.choice(genres_list, size=num_samples)
    secondary_genres = np.random.choice(genres_list, size=num_samples)
    
    directors = np.random.choice(directors_list, size=num_samples)
    
    # Generate realistic budget based on genre
    genre_budget_multiplier = {
        'Sci-Fi': 120, 'Action': 110, 'Animation': 100, 'Adventure': 105,
        'Thriller': 45, 'Drama': 30, 'Crime': 40, 'Horror': 15, 'Comedy': 35, 'Romance': 25
    }
    
    budgets = []
    for g in primary_genres:
        base = genre_budget_multiplier.get(g, 50)
        budget = np.random.exponential(scale=base) + 5
        budgets.append(round(min(budget, 350), 2))
    budgets = np.array(budgets)
    
    # Generate IMDb rating based on Director, Budget, Duration, and Genre with random noise
    director_quality = {
        'Christopher Nolan': 8.4, 'Steven Spielberg': 8.0, 'Martin Scorsese': 8.1,
        'Quentin Tarantino': 8.2, 'Denis Villeneuve': 8.0, 'Greta Gerwig': 7.7,
        'James Cameron': 7.8, 'Guillermo del Toro': 7.6, 'Ridley Scott': 7.3,
        'Ava DuVernay': 7.2, 'Indie Director': 6.5
    }
    
    ratings = []
    revenues = []
    votes_list = []
    metascores = []
    
    for i in range(num_samples):
        dir_score = director_quality.get(directors[i], 6.5)
        dur_bonus = 0.3 if durations[i] > 130 else (-0.2 if durations[i] < 90 else 0)
        budget_bonus = np.log1p(budgets[i]) * 0.15
        
        # Rating formula with Gaussian noise
        base_rating = dir_score + dur_bonus + budget_bonus + np.random.normal(0, 0.65)
        rating = round(float(np.clip(base_rating, 2.5, 9.8)), 1)
        ratings.append(rating)
        
        # Revenue correlates with Budget and Rating
        multiplier = np.random.uniform(0.5, 4.5) if rating >= 7.0 else np.random.uniform(0.1, 1.8)
        rev = round(float(budgets[i] * multiplier + np.random.normal(10, 20)), 2)
        revenues.append(max(0.05, rev))
        
        # Votes correlate with popularity/rating
        votes = int(np.exp(rating * 1.1) * np.random.uniform(50, 300))
        votes_list.append(votes)
        
        # Metascore correlates with rating
        meta = int(np.clip(rating * 10 + np.random.normal(0, 8), 15, 99))
        metascores.append(meta)
        
    df = pd.DataFrame({
        'Movie_ID': [f"MOV{i+1001}" for i in range(num_samples)],
        'Title': titles,
        'Release_Year': release_years,
        'Genre': primary_genres,
        'Secondary_Genre': secondary_genres,
        'Duration_Min': durations,
        'Director': directors,
        'Budget_USD_Millions': budgets,
        'Revenue_USD_Millions': revenues,
        'IMDb_Rating': ratings,
        'Metascore': metascores,
        'Votes': votes_list
    })
    
    # Introduce a small percentage of missing values to simulate real-world data cleaning needs
    df.loc[df.sample(frac=0.03, random_state=42).index, 'Budget_USD_Millions'] = np.nan
    df.loc[df.sample(frac=0.02, random_state=24).index, 'Metascore'] = np.nan
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'movies_dataset.csv')
    df.to_csv(filepath, index=False)
    print(f"[SUCCESS] Generated dataset with {len(df)} movies at '{filepath}'")
    return df

if __name__ == '__main__':
    generate_movie_dataset()
