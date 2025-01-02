import torch
import os
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

VECTOR_DIR = "optimized_vectors"
NUM_CLUSTERS = 4
concept_vectors = []
concept_names = []

# Load concept vectors
for file_name in os.listdir(VECTOR_DIR):
    if file_name.endswith(".pt"):
        file_path = os.path.join(VECTOR_DIR, file_name)
        vector = torch.load(file_path).cpu().numpy()
        concept_vectors.append(vector)
        concept_names.append(file_name.replace(".pt", ""))

# Convert the list of NumPy arrays to a single NumPy array
concept_vectors = np.array(concept_vectors)

# Perform clustering
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
kmeans.fit(concept_vectors)
cluster_assignments = kmeans.labels_
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, concept_vectors)

# Print results
print("Cluster Assignments:")
for idx, concept in enumerate(concept_names):
    print(f"{concept}: Cluster {cluster_assignments[idx]}")

print("\nClosest Concepts to Cluster Centers:")
for cluster_id, vector_idx in enumerate(closest):
    print(f"Cluster {cluster_id}: {concept_names[vector_idx]}")
