import torch 
import numpy as np
from tqdm import tqdm
from torch.autograd import grad
from torch.autograd.functional import hessian
from src.utils import create_Z, sample_mask, spectral_initialization

def loss_fn(L, R, mask, Z):
    return torch.square(mask * (L @ R.T - Z)).sum() / 4

@torch.no_grad()
def project(X, coherence):
    n, r = X.size()
    # incoherence projection of X
    row_norm_X = torch.linalg.vector_norm(X, ord=2, dim=1)
    frob_norm_X = torch.linalg.matrix_norm(X, ord='fro')
    row_norm_X.div_(frob_norm_X * np.sqrt(coherence / n))
    scale_X = torch.maximum(row_norm_X, torch.ones_like(row_norm_X))
    X.div_(scale_X.unsqueeze(-1))

def sgd(L, R, mask, Z, mu=None, projected=False, optim='SGD', iters=100, lr=0.001):
    # Initialization
    criterion = loss_fn
    if optim == 'SGD':
        optimizer = torch.optim.SGD([L, R], lr=lr)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam([L, R], lr=lr, weight_decay=1e-4)
    else:
        raise NotImplementedError
    # SGD
    losses = []
    for i in tqdm(range(iters), leave=False):
        optimizer.zero_grad()
        loss = criterion(L, R, mask, Z)
        losses.append((loss.item()))
        loss.backward()
        optimizer.step()
        new_lr = 0.5 * (lr + 2e-5) + 0.5 * (lr - 2e-5) * np.cos(np.pi * i / iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        # Project to feasible set
        if projected:
            project(L, mu)
            project(R, mu)
    
    w_L, h_L = L.size() 
    grad_norm_L = torch.linalg.matrix_norm(grad(criterion(L, R, mask, Z), L)[0])
    min_eigenvalue_L = torch.linalg.eigvalsh(hessian(criterion, (L, R, mask, Z))[0][0].reshape(w_L,h_L,-1).reshape(w_L*h_L,-1))[0]

    return losses, torch.sum(((L @ R.T) - Z)**2).item(), grad_norm_L.item(), min_eigenvalue_L.item()