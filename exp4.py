import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame({
    'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'title': [
        "Toy Story", "Jumanji", "Grumpier Old Men", "Waiting to Exhale", "Father of the Bride",
        "The Lion King", "Pulp Fiction", "Forrest Gump", "The Matrix", "Titanic"
    ],
    'genres': [
        "Adventure|Animation|Children|Comedy|Fantasy", "Adventure|Children|Fantasy",
        "Comedy|Romance", "Comedy|Drama|Romance", "Comedy",
        "Animation|Adventure|Drama", "Crime|Drama|Thriller", "Drama|Romance|Comedy", "Action|Sci-Fi", "Drama|Romance"
    ]
})

ratings_data = {
    'userId': [1, 1, 1, 2, 2, 3, 4, 5, 3, 4, 5, 1, 2, 3, 4],
    'movieId': [1, 2, 3, 1, 4, 5, 4, 1, 6, 7, 8, 9, 10, 6, 7],
    'rating': [5, 4, 3, 5, 4, 2, 3, 4, 5, 4, 3, 5, 4, 5, 4],
}
ratings_df = pd.DataFrame(ratings_data)

movies['genres'] = movies['genres'].str.replace('|', ' ')

vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(movies['genres'])

similarity_matrix = cosine_similarity(genre_matrix)

def recommend_movies(movie_title, num_recommendations=3):
    if movie_title not in movies['title'].values:
        return "Movie not found in the dataset!"

    idx = movies[movies['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_titles = [movies.iloc[i[0]]['title'] for i in sorted_movies[1:num_recommendations+1]]
    return recommended_titles

recommended_movies = recommend_movies("The Lion King")
print("Movies similar to 'The Lion King':", recommended_movies)
