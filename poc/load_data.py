import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import numpy as np

VECTOR_DIR = "optimized_vectors"
concept_vectors = []
concept_names = []

# Load concept vectors
for file_name in os.listdir(VECTOR_DIR):
    if file_name.endswith(".pt"):
        file_path = os.path.join(VECTOR_DIR, file_name)
        vector = torch.load(file_path).cpu().numpy()
        concept_vectors.append(vector)
        concept_names.append(file_name.replace(".pt", ""))

# Filter and order concept vectors by categories
categories = ["female", "car", "table"]
filtered_vectors, filtered_names = [],[]
temp_one,temp_two = [],[]

for category in categories:
    for name, vector in zip(concept_names, concept_vectors):
        if category in name.lower():
            filtered_names.append(name)
            filtered_vectors.append(vector)
        elif "male" in name.lower() and "female" not in name.lower() and len(temp_one) < 3:
            temp_one.append(name)
            temp_two.append(vector)
for i in range(len(temp_one)):
    filtered_names.append(temp_one[i])
    filtered_vectors.append(temp_two[i])

# Normalize the filtered concept vectors
filtered_vectors = np.array(filtered_vectors)
norms = np.linalg.norm(filtered_vectors, axis=1, keepdims=True)
normalized_vectors = filtered_vectors / norms

# Calculate pairwise distances
distances = pairwise_distances(normalized_vectors, metric="cosine")
confusion_matrix = np.exp(distances)

# Plot the confusion matrix-like distance matrix
plt.figure(figsize=(10, 8))
plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar(label='Cosine Distance')
plt.xticks(ticks=np.arange(len(filtered_names)), labels=filtered_names, rotation=90)
plt.yticks(ticks=np.arange(len(filtered_names)), labels=filtered_names)
plt.title("Concept Vector Distance Matrix by Category")
plt.xlabel("Concepts")
plt.ylabel("Concepts")
plt.tight_layout()
plt.savefig("poc/distance_matrix.png")
plt.show()