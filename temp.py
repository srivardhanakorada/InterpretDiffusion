import os
import torch

def inspect_pt_files(directory):
    # List all .pt files in the directory
    pt_files = [f for f in os.listdir(directory) if f.endswith('.pt')]

    for file in pt_files:
        file_path = os.path.join(directory, file)
        try:
            # Load the .pt file
            data = torch.load(file_path)

            # Check the type of data
            if isinstance(data, torch.Tensor):
                print(f"File: {file}")
                print(f"Shape: {data.shape}")
                print(f"Content: {data}\n")
            elif isinstance(data, dict):
                print(f"File: {file}")
                print(f"Dictionary keys: {list(data.keys())}")
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        print(f"Key: {key}, Shape: {value.shape}")
                        print(f"Content: {value}\n")
            elif isinstance(data, list):
                print(f"File: {file}")
                print(f"List length: {len(data)}")
                if isinstance(data[0], torch.Tensor):
                    print(f"Shape of first element: {data[0].shape}")
                print(f"Content: {data}\n")
            else:
                print(f"File: {file} contains unsupported data type: {type(data)}\n")

        except Exception as e:
            print(f"Failed to load {file}: {e}")

# Usage example
directory = "results/exp_female/concept_0/vectors"  # Replace with the directory containing your .pt files
inspect_pt_files(directory)