import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
sns.set_theme(style="darkgrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def generate_eda_reports_and_charts(df, output_dir=None):
    """
    Performs Exploratory Data Analysis (EDA) on movie rating dataset and exports publication-ready charts.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("        EXPLORATORY DATA ANALYSIS REPORT        ")
    print("="*50)
    
    # Summary Statistics
    print("\n--- Dataset Overview ---")
    print(f"Total Movies Analyzed: {len(df)}")
    print(f"Average IMDb Rating: {df['IMDb_Rating'].mean():.2f} (Std: {df['IMDb_Rating'].std():.2f})")
    print(f"Median Budget: ${df['Budget_USD_Millions'].median():.2f} Million")
    print(f"Median Revenue: ${df['Revenue_USD_Millions'].median():.2f} Million")
    
    # Top 5 Highest Rated Movies
    print("\n--- Top 5 Highest-Rated Movies ---")
    top_5 = df.sort_values(by='IMDb_Rating', ascending=False)[['Title', 'Genre', 'Release_Year', 'Director', 'IMDb_Rating']].head(5)
    print(top_5.to_string(index=False))
    
    # ----------------------------------------------------
    # Chart 1: IMDb Rating Distribution & Categories
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(df['IMDb_Rating'], kde=True, ax=axes[0], color='#4C72B0', bins=25)
    axes[0].axvline(df['IMDb_Rating'].mean(), color='red', linestyle='--', label=f"Mean: {df['IMDb_Rating'].mean():.2f}")
    axes[0].axvline(df['IMDb_Rating'].median(), color='green', linestyle=':', label=f"Median: {df['IMDb_Rating'].median():.2f}")
    axes[0].set_title('IMDb Rating Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('IMDb Rating')
    axes[0].set_ylabel('Number of Movies')
    axes[0].legend()
    
    # Pie chart of rating categories
    category_counts = df['Rating_Category'].value_counts()
    axes[1].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=sns.color_palette("mako", len(category_counts)), startangle=140)
    axes[1].set_title('Movie Quality Breakdown', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, '01_rating_distribution.png')
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"[SAVED] Chart 1 saved to '{chart1_path}'")
    
    # ----------------------------------------------------
    # Chart 2: Average Rating by Genre
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    genre_avg = df.groupby('Genre')['IMDb_Rating'].agg(['mean', 'count']).reset_index().sort_values(by='mean', ascending=False)
    
    barplot = sns.barplot(x='mean', y='Genre', data=genre_avg, hue='Genre', palette='viridis', legend=False)
    plt.title('Average IMDb Rating by Primary Genre', fontsize=14, fontweight='bold')
    plt.xlabel('Average IMDb Rating')
    plt.ylabel('Primary Genre')
    
    for index, row in genre_avg.reset_index().iterrows():
        barplot.text(row['mean'] + 0.05, index, f"{row['mean']:.2f} ({row['count']} movies)", va='center', fontsize=10)
        
    plt.xlim(0, 10)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, '02_top_genres_by_rating.png')
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"[SAVED] Chart 2 saved to '{chart2_path}'")

    # ----------------------------------------------------
    # Chart 3: Correlation Matrix Heatmap
    # ----------------------------------------------------
    plt.figure(figsize=(9, 7))
    num_cols = ['IMDb_Rating', 'Metascore', 'Votes', 'Budget_USD_Millions', 'Revenue_USD_Millions', 'Duration_Min', 'Profit_USD_Millions', 'ROI']
    corr = df[num_cols].corr()
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Heatmap of Movie Attributes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, '03_correlation_heatmap.png')
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"[SAVED] Chart 3 saved to '{chart3_path}'")

    # ----------------------------------------------------
    # Chart 4: Budget vs IMDb Rating
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Budget_USD_Millions', y='IMDb_Rating', hue='Genre', size='Revenue_USD_Millions', sizes=(30, 300), data=df, alpha=0.75)
    plt.axhline(7.0, color='gray', linestyle='--', alpha=0.7, label='Benchmark Rating (7.0)')
    plt.title('Production Budget vs. IMDb Rating', fontsize=14, fontweight='bold')
    plt.xlabel('Budget (USD Millions)')
    plt.ylabel('IMDb Rating')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, '04_budget_vs_rating.png')
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"[SAVED] Chart 4 saved to '{chart4_path}'")

    # ----------------------------------------------------
    # Chart 5: Director Performance Analysis
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    dir_stats = df.groupby('Director').agg({'IMDb_Rating': 'mean', 'Revenue_USD_Millions': 'sum', 'Title': 'count'}).reset_index().sort_values(by='IMDb_Rating', ascending=False)
    
    sns.barplot(x='IMDb_Rating', y='Director', data=dir_stats, hue='Director', palette='magma', legend=False)
    plt.title('Director Performance: Average IMDb Rating', fontsize=14, fontweight='bold')
    plt.xlabel('Average IMDb Rating')
    plt.ylabel('Director')
    plt.xlim(0, 10)
    
    for idx, row in dir_stats.reset_index().iterrows():
        plt.text(row['IMDb_Rating'] + 0.05, idx, f"{row['IMDb_Rating']:.2f} rating | ${row['Revenue_USD_Millions']:.0f}M rev", va='center', fontsize=9)
        
    plt.tight_layout()
    chart5_path = os.path.join(output_dir, '05_director_performance.png')
    plt.savefig(chart5_path, dpi=300)
    plt.close()
    print(f"[SAVED] Chart 5 saved to '{chart5_path}'")
    
    print("\n[SUCCESS] EDA complete. All 5 visualization charts exported.")

if __name__ == '__main__':
    from data_preprocessing import load_and_preprocess_data
    df = load_and_preprocess_data()
    generate_eda_reports_and_charts(df)
