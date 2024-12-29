import torch
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

VECTOR_DIR = "path/to/your/pt/files"
concept_vectors = []
concept_names = []
for file_name in os.listdir(VECTOR_DIR):
    if file_name.endswith(".pt"):
        file_path = os.path.join(VECTOR_DIR, file_name)
        vector = torch.load(file_path).numpy()
        concept_vectors.append(vector)
        concept_names.append(file_name.replace(".pt", ""))
concept_vectors = np.array(concept_vectors)

distance_matrix = squareform(pdist(concept_vectors, metric="cosine"))
linkage_matrix = linkage(pdist(concept_vectors, metric="cosine"), method="ward")
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=concept_names, leaf_rotation=90, leaf_font_size=10)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Concepts")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

embedded = TSNE(n_components=2, metric="cosine", random_state=42).fit_transform(concept_vectors)
plt.figure(figsize=(8, 6))
plt.scatter(embedded[:, 0], embedded[:, 1], c="blue", alpha=0.7)
for i, concept in enumerate(concept_names):
    plt.annotate(concept, (embedded[i, 0], embedded[i, 1]), fontsize=9)
plt.title("t-SNE Visualization of Concept Vectors")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.tight_layout()
plt.show()