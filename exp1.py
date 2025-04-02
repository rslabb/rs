
from scipy.spatial import distance
P1 = (19, 16, 35)
P2 = (11,24, 19)
print(distance.euclidean(P1,P2))


import numpy as np
from scipy.spatial.distance import cityblock
point1 = np.array([1, 2, 3])
point2 = np.array([4, 5, 6])
manhattan_distance = cityblock(point1, point2)
print(f"Manhattan Distance: {manhattan_distance}")


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
items_data = {'Item1': [3, 4, 5], 'Item2': [1, 2, 3], 'Item3': [4, 5, 6], 'Item4': [2, 3, 4]}
items_matrix = np.array(list(items_data.values()))
print(items_matrix)
similarity_matrix = cosine_similarity(items_matrix)
print("Similarity Matrix:")
print(similarity_matrix)


import numpy as np
from sklearn.metrics import jaccard_score
point1_binary = np.array([1, 1, 0, 0, 1])
point2_binary = np.array([1, 0, 1, 1, 0])
jaccard_similarity = jaccard_score(point1_binary, point2_binary)
print(f"Jaccard Similarity: {jaccard_similarity}")


import numpy as np
from scipy.spatial.distance import minkowski
point1 = np.array([1, 2, 3])
point2 = np.array([4, 5, 6])
p = 3
minkowski_distance = minkowski(point1, point2, p)
print(f"Minkowski Distance (p={p}): {minkowski_distance}")