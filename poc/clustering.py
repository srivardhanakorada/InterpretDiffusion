import torch
import os
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

VECTOR_DIR = "path/to/your/pt/files"
NUM_CLUSTERS = 4
concept_vectors = []
concept_names = []
for file_name in os.listdir(VECTOR_DIR):
    if file_name.endswith(".pt"):
        file_path = os.path.join(VECTOR_DIR, file_name)
        vector = torch.load(file_path).numpy()
        concept_vectors.append(vector)
        concept_names.append(file_name.replace(".pt", ""))
concept_vectors = torch.tensor(concept_vectors).numpy()
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
kmeans.fit(concept_vectors)
cluster_assignments = kmeans.labels_
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, concept_vectors)
print("Cluster Assignments:")
for idx, concept in enumerate(concept_names):
    print(f"{concept}: Cluster {cluster_assignments[idx]}")
print("\nClosest Concepts to Cluster Centers:")
for cluster_id, vector_idx in enumerate(closest):
    print(f"Cluster {cluster_id}: {concept_names[vector_idx]}")