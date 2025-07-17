import torch
import numpy as np
from src.convex import SDP
from src.nonconvex import sgd
from src.utils import augment_mask, augment_Z, spectral_initialization

def check(mask: np.ndarray, Z: np.ndarray, p, r, mu=None):
    # Check Convex
    _, _, X_star, convex_E = SDP(Z, mask)
    print(X_star)
    print(convex_E)
    # Check Nonconvex
    L, R = spectral_initialization(p, r, Z, mask)
    L.requires_grad_(True)
    R.requires_grad_(True)
    losses, nonconvex_E, grad_norm, min_eigen = sgd(L, R, mask, Z, mu, projected=False, optim='SGD', iters=5000, lr=1e-3)
    print(L.data)
    print(R.data)
    print((L @ R.T).data)
    print(nonconvex_E)
    print(grad_norm)
    print(min_eigen)


    
