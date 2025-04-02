import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5],
    'item_id': ['Laptop', 'Phone', 'Laptop', 'Tablet', 'Phone', 'Headphones', 'Phone', 'Smartwatch', 'Tablet'],
    'rating': [5, 3, 2, 5, 4, 5, 4, 5, 3]  # Ratings based on user preferences
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)
print("\nUser-Item Matrix:")
print(user_item_matrix)

user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
print("\nUser Similarity Matrix:")
print(user_similarity_df)

user_id = 2
similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:4]
recommended_items = set()
user_1_rated_items = set(df[df['user_id'] == user_id]['item_id'])

for similar_user in similar_users.index:
    similar_user_ratings = df[df['user_id'] == similar_user]
    for _, row in similar_user_ratings.iterrows():
        if row['item_id'] not in user_1_rated_items:
            recommended_items.add(row['item_id'])

print("\nRecommended Items for User 2:")
print(recommended_items)
