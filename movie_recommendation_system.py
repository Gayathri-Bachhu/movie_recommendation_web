# -*- coding: utf-8 -*-
"""
Improved Movie Recommendation System
Fixes dtype warnings and adds error handling
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load movie data with robust error handling for corrupted rows"""
    try:
        print("Loading movie dataset...")
        
        # First, try to read with minimal constraints
        df = pd.read_csv(filepath, low_memory=False, dtype=str)
        print(f"Initial load: {len(df)} rows")
        
        # Convert numeric columns safely
        numeric_columns = {
            'popularity': 'float64',
            'vote_average': 'float64', 
            'vote_count': 'int64'
        }
        
        rows_before = len(df)
        
        # Clean and convert numeric columns
        for col, dtype in numeric_columns.items():
            if col in df.columns:
                if dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
        
        # Remove rows where critical columns are NaN
        df = df.dropna(subset=['vote_average', 'vote_count'])
        df = df[df['vote_count'] > 0]  # Remove movies with 0 votes
        
        rows_after = len(df)
        if rows_before != rows_after:
            print(f"Removed {rows_before - rows_after} rows with invalid data")
        
        print(f"Successfully loaded {len(df)} valid movies")
        return df
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{filepath}'")
        print("Please ensure the movies_metadata.csv file is in the same directory")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Trying alternative loading method...")
        
        # Alternative: Read line by line and skip problematic rows
        try:
            df = pd.read_csv(filepath, low_memory=False, dtype=str, on_bad_lines='skip')
            print(f"Alternative method loaded {len(df)} rows")
            
            # Convert numeric columns
            for col in ['popularity', 'vote_average', 'vote_count']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Clean data
            df = df.dropna(subset=['vote_average', 'vote_count'])
            df = df[df['vote_count'] > 0]
            
            print(f"Final dataset: {len(df)} valid movies")
            return df
            
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            return None

def clean_data(df):
    """Clean and prepare the data"""
    print("Cleaning data...")
    
    # Ensure required columns exist
    required_cols = ['title', 'overview', 'vote_average', 'vote_count']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None
    
    # Fill NaN values in overview
    df['overview'] = df['overview'].fillna('')
    
    # Fill NaN values in title
    df['title'] = df['title'].fillna('Unknown Title')
    
    # Additional data quality checks
    initial_count = len(df)
    
    # Remove rows with empty titles or overviews
    df = df[df['title'].str.strip() != '']
    df = df[df['title'] != 'Unknown Title']
    
    # Ensure vote_average is between 0 and 10
    df = df[(df['vote_average'] >= 0) & (df['vote_average'] <= 10)]
    
    # Remove duplicates based on title (keep first occurrence)
    df = df.drop_duplicates(subset=['title'], keep='first')
    
    final_count = len(df)
    if initial_count != final_count:
        print(f"Data cleaning removed {initial_count - final_count} invalid rows")
    
    print(f"Final clean dataset: {final_count} movies")
    return df

def simple_recommender(df, top_n=20):
    """Simple recommendation system based on weighted rating"""
    print("\n=== SIMPLE RECOMMENDATION SYSTEM ===")
    
    # Check if we have enough data for visualization
    if len(df) < 10:
        print("Not enough data for meaningful recommendations")
        return df
    
    try:
        # Visualize vote count distribution
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='vote_count')
        plt.title('Distribution of Vote Counts')
        plt.show()
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    # Calculate statistics
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(0.90)
    
    print(f"Average vote across all movies: {C:.2f}")
    print(f"90th percentile vote count: {m:.0f}")
    
    # Filter qualified movies
    q_movies = df[df['vote_count'] >= m].copy()
    
    if len(q_movies) == 0:
        print("No movies meet the vote count threshold. Lowering threshold...")
        m = df['vote_count'].quantile(0.70)  # Lower threshold
        q_movies = df[df['vote_count'] >= m].copy()
    
    print(f"Number of qualified movies: {len(q_movies)} out of {len(df)} ({len(q_movies)/len(df)*100:.1f}%)")
    
    if len(q_movies) == 0:
        print("Error: No movies qualify for recommendations")
        return df
    
    # Calculate weighted rating
    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)
    
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
    q_movies = q_movies.sort_values('score', ascending=False)
    
    print(f"\nTop {top_n} Movies by Weighted Score:")
    print("="*60)
    top_movies = q_movies[['title', 'vote_count', 'vote_average', 'score']].head(top_n)
    for i, (_, row) in enumerate(top_movies.iterrows(), 1):
        title = str(row['title'])[:40]
        print(f"{i:2d}. {title:<40} | Score: {row['score']:.2f} | Votes: {int(row['vote_count']):,} | Avg: {row['vote_average']:.1f}")
    
    return q_movies

def content_based_recommender(df):
    """Content-based recommendation system using TF-IDF and cosine similarity"""
    print("\n=== CONTENT-BASED RECOMMENDATION SYSTEM ===")
    
    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    
    # Fit and transform the overview data
    print("Computing TF-IDF matrix...")
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Calculate cosine similarity
    print("Computing cosine similarity matrix...")
    cosine_sim = cosine_similarity(tfidf_matrix)
    print(f"Cosine similarity matrix shape: {cosine_sim.shape}")
    
    # Create indices mapping
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    def get_recommendations(title, cosine_sim=cosine_sim, top_n=10):
        """Get movie recommendations based on content similarity"""
        try:
            # Get the index of the movie
            idx = indices[title]
            
            # Get pairwise similarity scores
            sim_scores = list(enumerate(cosine_sim[idx]))
            
            # Sort movies based on similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get the indices of the most similar movies (excluding the movie itself)
            sim_scores = sim_scores[1:top_n+1]
            movie_indices = [i[0] for i in sim_scores]
            
            # Return the top similar movies with their similarity scores
            recommendations = df.iloc[movie_indices][['title', 'vote_average', 'vote_count']]
            scores = [score[1] for score in sim_scores]
            recommendations['similarity_score'] = scores
            
            return recommendations
            
        except KeyError:
            print(f"Movie '{title}' not found in the dataset.")
            print("Available movies containing your search term:")
            matches = df[df['title'].str.contains(title, case=False, na=False)]['title'].head(10)
            for match in matches:
                print(f"  - {match}")
            return None
    
    return get_recommendations, indices

def demonstrate_recommendations(df, get_recommendations):
    """Demonstrate the recommendation system with examples"""
    print("\n=== DEMONSTRATION ===")
    
    # Test movies
    test_movies = [
        'The Dark Knight Rises',
        'Batman: Mask of the Phantasm',
        'Toy Story',
        'The Shawshank Redemption'
    ]
    
    for movie in test_movies:
        print(f"\nRecommendations for '{movie}':")
        print("-" * 50)
        recommendations = get_recommendations(movie)
        
        if recommendations is not None:
            for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                print(f"{i:2d}. {row['title'][:35]:<35} | Similarity: {row['similarity_score']:.3f} | Rating: {row['vote_average']:.1f} | Votes: {row['vote_count']:,}")

def main():
    """Main function to run the movie recommendation system"""
    print("Movie Recommendation System")
    print("=" * 50)
    
    # Load data
    df = load_data("movies_metadata.csv")
    if df is None:
        return
    
    # Clean data
    df = clean_data(df)
    print(f"Dataset shape after cleaning: {df.shape}")
    
    # Simple recommender
    q_movies = simple_recommender(df)
    
    # Content-based recommender
    get_recommendations, indices = content_based_recommender(df)
    
    # Demonstrate recommendations
    demonstrate_recommendations(df, get_recommendations)
    
    # Interactive mode
    print("\n=== INTERACTIVE MODE ===")
    print("Enter movie titles to get recommendations (or 'quit' to exit):")
    
    while True:
        user_input = input("\nEnter movie title: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input:
            recommendations = get_recommendations(user_input)
            if recommendations is not None:
                print(f"\nTop 10 recommendations for '{user_input}':")
                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    print(f"{i:2d}. {row['title']}")

if __name__ == "__main__":
    main()