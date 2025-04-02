import pandas as pd

def load_movie_data():
    """Loads a sample movie dataset with genre, rating, year, and director."""
    return pd.DataFrame([
        {"id": 1, "title": "Movie A", "genre": "Action", "rating": 7.8, "year": 2020, "director": "Director X"},
        {"id": 2, "title": "Movie B", "genre": "Comedy", "rating": 8.2, "year": 2018, "director": "Director Y"},
        {"id": 3, "title": "Movie C", "genre": "Drama", "rating": 6.9, "year": 2019, "director": "Director X"},
        {"id": 4, "title": "Movie D", "genre": "Sci-Fi", "rating": 9.0, "year": 2022, "director": "Director Z"},
        {"id": 5, "title": "Movie E", "genre": "Action", "rating": 7.5, "year": 2017, "director": "Director Y"},
        {"id": 6, "title": "Movie F", "genre": "Comedy", "rating": 6.8, "year": 2021, "director": "Director Z"},
        {"id": 7, "title": "Movie G", "genre": "Sci-Fi", "rating": 8.7, "year": 2015, "director": "Director X"},
    ])

def constraint_based_movie_recommendation(movies, genre=None, min_rating=None, max_year=None, director=None):
    """
    Filters movies based on user constraints.

    Parameters:
    - movies (DataFrame): The movie dataset.
    - genre (str): Filter by movie genre.
    - min_rating (float): Minimum IMDb rating.
    - max_year (int): Maximum release year.
    - director (str): Filter by director.

    Returns:
    - DataFrame of recommended movies.
    """
    filtered_movies = movies.copy()

    if genre:
        filtered_movies = filtered_movies[filtered_movies['genre'] == genre]
    if min_rating:
        filtered_movies = filtered_movies[filtered_movies['rating'] >= min_rating]
    if max_year:
        filtered_movies = filtered_movies[filtered_movies['year'] <= max_year]
    if director:
        filtered_movies = filtered_movies[filtered_movies['director'] == director]

    return filtered_movies


movies = load_movie_data()

scenarios = [
    {"name": "High-Rated Action Movies", "constraints": {"genre": "Action", "min_rating": 7.5}},
    {"name": "Comedy Movies Before 2020", "constraints": {"genre": "Comedy", "max_year": 2020}},
    {"name": "Sci-Fi Movies with IMDb 8+", "constraints": {"genre": "Sci-Fi", "min_rating": 8.0}},
    {"name": "Movies Directed by 'Director X'", "constraints": {"director": "Director X"}},
    {"name": "Best Movies from the Last 5 Years", "constraints": {"min_rating": 8.0, "max_year": 2023}}
]

for scenario in scenarios:
    print(f"\n=== {scenario['name']} ===")
    recommended_movies = constraint_based_movie_recommendation(movies, **scenario["constraints"])
    print(recommended_movies if not recommended_movies.empty else "No movies found.")
