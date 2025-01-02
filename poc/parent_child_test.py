import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load concept vectors
VECTOR_DIR = "optimized_vectors"
concept_vectors = []
concept_names = []

for file_name in os.listdir(VECTOR_DIR):
    if file_name.endswith(".pt"):
        file_path = os.path.join(VECTOR_DIR, file_name)
        vector = torch.load(file_path).cpu().numpy()
        concept_vectors.append(vector)
        concept_names.append(file_name.replace(".pt", ""))

# Normalize vectors
concept_vectors = np.array(concept_vectors)
norms = np.linalg.norm(concept_vectors, axis=1, keepdims=True)
normalized_vectors = concept_vectors / norms

# Parent-child relationship test
parent_child_pairs = [
    ("male", "young_male"),
    ("male", "old_male"),
    ("female", "young_female"),
    ("female", "old_female"),
]

# Compute deltas
deltas = {}
for parent, child in parent_child_pairs:
    deltas[child] = normalized_vectors[concept_names.index(child)] - normalized_vectors[concept_names.index(parent)]

# Compare deltas between siblings
sibling_pairs = [
    ("young_male", "old_male"),
    ("young_female", "old_female"),
]

print("Sibling Delta Similarities:")
for sibling1, sibling2 in sibling_pairs:
    delta1 = deltas[sibling1]
    delta2 = deltas[sibling2]
    similarity = cosine_similarity(delta1.reshape(1, -1), delta2.reshape(1, -1))[0, 0]
    print(f"Cosine Similarity between {sibling1} and {sibling2}: {similarity:.4f}")

# Cross-category projections
cross_category_tests = [
    ("female", "young_female", "young_male"),
    ("female", "old_female", "old_male"),
]

print("\nCross-Category Projections:")
for new_parent, expected_child, source_delta in cross_category_tests:
    new_parent_idx = concept_names.index(new_parent)
    expected_child_idx = concept_names.index(expected_child)
    source_delta_vec = deltas[source_delta]

    # Predict the child vector
    predicted_child = normalized_vectors[new_parent_idx] + source_delta_vec
    predicted_child = predicted_child / np.linalg.norm(predicted_child)

    # Compare to actual child
    actual_child = normalized_vectors[expected_child_idx]
    similarity = cosine_similarity(predicted_child.reshape(1, -1), actual_child.reshape(1, -1))[0, 0]
    print(f"Similarity between predicted {expected_child} and actual {expected_child}: {similarity:.4f}")
