import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import torch
import csv

# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def encode_images(image_paths, processor, model):
    embeddings = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_embedding = model.get_image_features(**inputs)
        embeddings.append(image_embedding)
    return torch.vstack(embeddings)

def encode_texts(concept_descriptions, processor, model):
    text_inputs = processor(text=concept_descriptions, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs)
    return text_embeddings

def compute_clip_similarity(image_embeddings, text_embedding):
    # Normalize embeddings
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    text_embedding = F.normalize(text_embedding, p=2, dim=1)
    # Cosine similarity
    return torch.matmul(image_embeddings, text_embedding.T).squeeze()

def compute_baseline_scores(image_embeddings, text_embedding):
    similarities = compute_clip_similarity(image_embeddings, text_embedding)
    mean_similarity = similarities.mean().item()
    variance = similarities.var().item()
    return mean_similarity, variance

# Path to generated images
CONCEPTS = {
    "Car": "poc/car/concept_0/images",
    "Sports-Car": "poc/car/concept_1/images",
    "Luxury-Car": "poc/car/concept_2/images",
    "Female": "poc/female/concept_0/images",
    "Young-Female": "poc/female/concept_1/images",
    "Old-Female": "poc/female/concept_2/images",
    "Male": "poc/male/concept_0/images",
    "Young-Male": "poc/male/concept_1/images",
    "Old-Male": "poc/male/concept_2/images",
    "Table": "poc/table/concept_0/images",
    "Wooden-Table": "poc/table/concept_1/images",
    "Round-Table":  "poc/table/concept_2/images",
}

TEXT_DESCRIPTIONS = {
    "Car": "A car",
    "Sports-Car": "A sports car",
    "Luxury-Car": "A luxury car",
    "Female": "A female CEO",
    "Young-Female": "A young female CEO",
    "Old-Female": "An old female CEO",
    "Male": "A male CEO",
    "Young-Male": "A young male CEO",
    "Old-Male": "An old male CEO",
    "Table": "A table",
    "Wooden-Table": "A wooden table",
    "Round-Table": "A round table",
}

# Compute metrics for each concept
results = {}
for concept, image_dir in CONCEPTS.items():
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
    image_embeddings = encode_images(image_paths, processor, model)
    text_embedding = encode_texts([TEXT_DESCRIPTIONS[concept]], processor, model)
    mean_sim, variance = compute_baseline_scores(image_embeddings, text_embedding)
    results[concept] = {"Mean Similarity": mean_sim, "Variance": variance}

output_csv_path = "clip_similarity_results.csv"

with open(output_csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Concept", "Mean Similarity", "Variance"])
    for concept, metrics in results.items():
        writer.writerow([concept, metrics["Mean Similarity"], metrics["Variance"]])
print(f"Results have been saved to {output_csv_path}")
