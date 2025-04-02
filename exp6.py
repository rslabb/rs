import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

np.random.seed(42)
num_users = 12
num_items = 10
ratings = np.random.randint(1, 6, (num_users, num_items))

ratings_df = pd.DataFrame(ratings, columns=[f'Item_{i+1}' for i in range(num_items)])
ratings_df.index = [f'User_{i+1}' for i in range(num_users)]

def recommend_items(rating_matrix, k=3):
    rating_matrix = rating_matrix.astype(float)
    U, sigma, Vt = svds(rating_matrix, k=k)
    sigma = np.diag(sigma)
    reconstructed_matrix = np.dot(np.dot(U, sigma), Vt)
    return reconstructed_matrix

original_recommendations = recommend_items(ratings)

fake_user = np.random.randint(1, 3, (1, num_items))
fake_user[0, 4] = 5  # Attacker targets "Item_5"

ratings_with_attack = np.vstack([ratings, fake_user])

attacked_recommendations = recommend_items(ratings_with_attack)

original_ranking = np.argsort(-original_recommendations.mean(axis=0))
attacked_ranking = np.argsort(-attacked_recommendations.mean(axis=0))

ranking_df = pd.DataFrame({
    'Item': [f'Item_{i+1}' for i in range(num_items)],
    'Rank Before Attack': np.argsort(original_ranking) + 1,
    'Rank After Attack': np.argsort(attacked_ranking) + 1
})

print("Item Rankings Before and After Attack:")
print(ranking_df)

plt.figure(figsize=(10, 5))
plt.hist(ratings.flatten(), bins=np.arange(1, 7)-0.5, alpha=0.7, label='Before Attack', color='blue')
plt.hist(ratings_with_attack.flatten(), bins=np.arange(1, 7)-0.5, alpha=0.7, label='After Attack', color='red')
plt.xlabel("Rating Value")
plt.ylabel("Frequency")
plt.title("Histogram of Ratings Before and After Attack")
plt.legend()
plt.show()