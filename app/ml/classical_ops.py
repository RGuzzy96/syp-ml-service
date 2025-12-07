import torch

def classical_kernel(X1: torch.Tensor, X2: torch.Tensor = None):
    """
    Computes a simple linear kernel.
    If X2 is None → compute K(X1, X1) for training.
    Else → compute K(X1, X2) for prediction.
    """
    if X2 is None:
        X2 = X1
    return torch.mm(X1, X2.T)
