import torch

def classical_kernel(X_batch):
    return torch.mm(X_batch, X_batch.T) # simple matrix multiplication for kernel