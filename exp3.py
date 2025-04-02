import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame({
    'MovieID': [0, 1, 2, 3, 4],
    'Title': ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"],
    'Genre': [[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1]]  # Example genres (Action, Comedy, Drama)
})

user_ratings_matrix = np.array([
    [5, 3, 0, 1, 4],  
    [4, 0, 5, 1, 0],  
    [0, 2, 4, 0, 5]
])

genre_matrix = np.array(movies['Genre'].tolist())
user_profiles = np.dot(user_ratings_matrix, genre_matrix) / np.maximum(user_ratings_matrix.sum(axis=1, keepdims=True), 1e-9)

for i, profile in enumerate(user_profiles):
    print(f"User {i+1} Profile (Weighted Genre Preferences): {profile}")

for user_id, user_profile in enumerate(user_profiles):
    movie_scores = cosine_similarity([user_profile], genre_matrix)[0]
    recommendations = sorted([(movies.iloc[i]['Title'], movie_scores[i]) for i in range(len(movies)) if user_ratings_matrix[user_id, i] == 0], key=lambda x: -x[1])
    
    print(f"\nRecommended Movies for User {user_id+1}:")
    for movie, score in recommendations:
        print(f"{movie}: {score:.2f}")