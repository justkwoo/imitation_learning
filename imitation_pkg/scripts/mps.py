import torch

# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


