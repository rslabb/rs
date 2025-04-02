import numpy as np
import pandas as pd
from scipy.linalg import svd

ratings_matrix = np.array([
    [5, 4, 0, 3, 2],
    [4, 0, 5, 3, 1],
    [0, 5, 4, 0, 3],
    [3, 3, 0, 5, 4],
    [2, 1, 3, 4, 0]
])
ratings_df = pd.DataFrame(ratings_matrix, columns=["Item1", "Item2", "Item3", "Item4", "Item5"])
ratings_filled = ratings_df.replace(0, ratings_df.mean())
U, sigma, Vt = svd(ratings_filled)
k = 2
sigma_k = np.diag(sigma[:k])
U_k = U[:, :k]
Vt_k = Vt[:k, :]
ratings_reconstructed = np.dot(U_k, np.dot(sigma_k, Vt_k))
recommendation_df = pd.DataFrame(ratings_reconstructed, columns=ratings_df.columns)
print(recommendation_df)
top_recommendations = recommendation_df.idxmax(axis=1)
print(top_recommendations)
