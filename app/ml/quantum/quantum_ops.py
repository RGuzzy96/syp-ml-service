import time
import torch
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.kernels import FidelityQuantumKernel

algorithm_globals.random_seed = 123

# using simulator backend for now instead of AWS braket
backend = AerSimulator(method="statevector")

# cache kernel objects so we avoid rebuilding feature maps repeatedly
_kernel_cache = {}

def _get_or_create_kernel(num_features: int):
    if num_features not in _kernel_cache:
        feature_map = ZZFeatureMap(num_features, reps=1)
        _kernel_cache[num_features] = FidelityQuantumKernel(
            feature_map=feature_map,
            # backend=backend
        )
    return _kernel_cache[num_features]


def quantum_kernel(X1: torch.Tensor, X2: torch.Tensor = None) -> torch.Tensor:
    """
    Computes a quantum kernel matrix between X1 and X2.
    If X2 is None → compute K(X1, X1).
    """
        
    X1_np = X1.detach().cpu().numpy()
    if X2 is None:
        X2_np = X1_np
    else:
        X2_np = X2.detach().cpu().numpy()

    num_features = X1_np.shape[1]
    qkernel = _get_or_create_kernel(num_features)

    # evaluate the kernel matrix
    K = qkernel.evaluate(X1_np, X2_np)

    return torch.tensor(K, dtype=torch.float32)