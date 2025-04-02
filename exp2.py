import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ratings_matrix = np.array([
    [5, 4, 0, 3, 2],
    [4, 0, 5, 3, 1],
    [0, 5, 4, 0, 3],
    [3, 3, 0, 5, 4],
    [2, 1, 3, 4, 0]
])

ratings_df = pd.DataFrame(ratings_matrix, columns=["Item1", "Item2", "Item3", "Item4", "Item5"])
print(ratings_df)
ratings_filled = ratings_df.replace(0, ratings_df.mean())
scaler = StandardScaler()
ratings_scaled = scaler.fit_transform(ratings_filled)
pca = PCA(n_components=2)
ratings_pca = pca.fit_transform(ratings_scaled)
ratings_reconstructed = pca.inverse_transform(ratings_pca)
recommendation_df = pd.DataFrame(ratings_reconstructed, columns=ratings_df.columns)
print(recommendation_df)
top_recommendations = recommendation_df.idxmax(axis=1)
print(top_recommendations)
