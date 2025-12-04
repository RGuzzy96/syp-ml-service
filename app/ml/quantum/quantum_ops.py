import time
import torch

def quantum_kernel(X_batch):
    # simulate latency + noise
    time.sleep(0.1)

    # mock: randomly distort the classical kernel
    K = torch.mm(X_batch, X_batch.T)
    noise = torch.randn_like(K) * 0.05

    return K + noise