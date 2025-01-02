import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=1280, output_ch=1280, resolution=1, nonlinearity="relu"):
        super(MLP, self).__init__()
        output_dim = output_ch * resolution * resolution
        self.resolution = resolution
        self.output_ch = output_ch
        self.fc1 = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x, x_ts):
        x = x.to(self.fc1.weight.dtype)
        x = self.fc1(x)  # Shape is b.s * 81920
        return x.view(x.shape[0], self.output_ch, self.resolution, self.resolution)  # b.s * 1280 * 8 * 8

class ModifiedMLP(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=1280, output_ch=1280, resolution=1, nonlinearity="relu", fixed_vectors=None):
        super(ModifiedMLP, self).__init__()
        output_dim = output_ch * resolution * resolution
        self.resolution = resolution
        self.output_ch = output_ch
        self.fixed_vectors = fixed_vectors
        self.fc1 = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x, x_ts):
        if self.fixed_vectors is not None:
            # Return fixed vectors during inference
            batch_size = x.shape[0]
            fixed_vector_reshaped = self.fixed_vectors.view(batch_size, self.output_ch, self.resolution, self.resolution)
            return fixed_vector_reshaped
        else:
            x = x.to(self.fc1.weight.dtype)
            x = self.fc1(x)
            return x.view(x.shape[0], self.output_ch, self.resolution, self.resolution)

model_types = {"MLP": MLP, "ModifiedMLP": ModifiedMLP}
