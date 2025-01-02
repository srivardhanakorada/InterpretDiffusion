import torch
from tqdm import tqdm

def compute_similarity(vector1, vector2):
    norm1 = torch.norm(vector1)
    norm2 = torch.norm(vector2)
    if norm1 == 0 or norm2 == 0: return 0
    return torch.dot(vector1, vector2) / (norm1 * norm2)

def contrastive_loss(h_parent, deltas, child_vectors, temperature=0.1):
    N = len(deltas)
    loss = 0
    for i in range(N):
        delta_i = deltas[i]
        target_similarity = compute_similarity(h_parent + delta_i, child_vectors[i]) / temperature
        negative_similarities = []
        for j in range(N):
            if i != j:
                negative_similarities.append(compute_similarity(h_parent + delta_i, child_vectors[j]) / temperature)
        negative_similarities = torch.stack(negative_similarities)
        neg_log_sum = torch.logsumexp(negative_similarities, dim=0)
        loss += -target_similarity + neg_log_sum
    return loss / N

def optimize_hierarchical_vectors(h_parent, child_vectors, learning_rate=0.01, num_iterations=100, temperature=0.1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    h_parent = h_parent.to(device)
    child_vectors = [v.to(device) for v in child_vectors]
    deltas = torch.stack([child - h_parent for child in child_vectors], dim=0).to(device)
    h_parent = h_parent.clone().detach().requires_grad_(True)
    deltas.requires_grad_(True)
    optimizer = torch.optim.Adam([h_parent, deltas], lr=learning_rate)
    for _ in tqdm(range(num_iterations)):
        optimizer.zero_grad()
        loss = contrastive_loss(h_parent, deltas, child_vectors, temperature)
        loss.backward()
        optimizer.step()
    final_vectors = [h_parent + delta for delta in deltas]
    return h_parent.detach(), torch.stack(final_vectors).detach()

def save_vectors(path, h_parent, child_names, final_vectors, parent_name):
    torch.save(h_parent, f"{path}/{parent_name}.pt")
    for name, vector in zip(child_names, final_vectors):
        torch.save(vector, f"{path}/{name}.pt")

if __name__ == "__main__":
    h_parent = torch.load("concept_vectors/table.pt")
    child_vectors = [
        torch.load("concept_vectors/wooden_table.pt"),
        torch.load("concept_vectors/round_table.pt")
    ]
    child_names = ["wooden_table", "round_table"]
    parent_name = "table"
    optimized_parent, final_vectors = optimize_hierarchical_vectors(h_parent, child_vectors, learning_rate=0.01, num_iterations=200, temperature=0.1)
    save_vectors("optimized_vectors", optimized_parent, child_names, final_vectors, parent_name)
    print("Optimized Parent Vector and Child Vectors have been saved.")