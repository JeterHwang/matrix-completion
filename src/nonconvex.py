import torch 
import numpy as np
from typing import List
from tqdm import tqdm
from torch.autograd import grad
from torch.autograd.functional import hessian
from src.utils import incoherence_proj

def create_loss_fn(loss_type, **kwargs):
    def MC_loss_fn(X_L: torch.Tensor, X_R: torch.Tensor, e: torch.Tensor = kwargs['e'], Z_L: torch.Tensor = kwargs['ZL'], Z_R: torch.Tensor = kwargs['ZR']):
        return torch.square(kwargs['mask'] * (X_L @ X_R.T - Z_L @ Z_R.T + e)).sum() / 4
    def MC_loss_fn_PSD(X: torch.Tensor, e: torch.Tensor = kwargs['e'], Z: torch.Tensor = kwargs['Z']):
        return torch.square(kwargs['mask'] * (X @ X.T - Z @ Z.T + e)).sum() / 4
    def PSSE_loss_fn(x: torch.Tensor, e: torch.Tensor = kwargs['e'], z: torch.Tensor = kwargs['Z']):
        pred = (x.T.unsqueeze(0) @ kwargs['P'] @ x.unsqueeze(0))
        gt = (z.T.unsqueeze(0) @ kwargs['P'] @ z.unsqueeze(0))
        return (pred - gt + e).square().sum() / 2
    
    if loss_type == 'MC':
        return MC_loss_fn
    elif loss_type == 'MC_PSD':
        return MC_loss_fn_PSD
    elif loss_type == 'PSSE':
        return PSSE_loss_fn
    else:
        raise NotImplementedError

def create_proj_fn(task, **kwargs):
    def MC_project(X):
        with torch.no_grad():
            incoherence_proj(X, kwargs['mu'])
    def PSSE_project(x):
        with torch.no_grad():
            x[len(x) // 2] = 0
    if task == 'MC':
        return MC_project
    elif task == 'PSSE':
        return PSSE_project
    else:
        raise NotImplementedError

def sgd(
    parameters: List[torch.Tensor], 
    criterion,
    project_fn,
    optim: str      = 'SGD', 
    iters: int      = 100, 
    lr: float       = 0.001,
    min_lr: float   = 1e-6,
    lr_sched: str   = 'static',
    ret_eig: bool   = False,
):
    assert len(parameters) in [1, 2]
    # Initialization
    if optim == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=lr)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=lr)
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(parameters, lr=lr)
    else:
        raise NotImplementedError
    # SGD
    losses = []
    with tqdm(total=iters, leave=False) as pbar:
        for i in range(iters):
            optimizer.zero_grad()
            loss = criterion(*parameters)
            losses.append((loss.item()))
            loss.backward()
            optimizer.step()
            if lr_sched == 'static':
                new_lr = lr
            elif lr_sched == 'cosine':
                new_lr = 0.5 * (lr + min_lr) + 0.5 * (lr - min_lr) * np.cos(np.pi * i / iters)
            elif lr_sched == 'step':
                new_lr = lr * ((0.1) ** (i * 5 // iters))
            else:
                raise NotImplementedError
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            # Project to feasible set
            if project_fn is not None:
                for param in parameters:
                    project_fn(param)
            pbar.update(1)
            pbar.set_description(f"Iter: {i}, lr={new_lr:.6f}, loss={losses[-1]:.9f}")
    
    w_L, h_L = parameters[0].size() 
    grad_norm_L = torch.linalg.matrix_norm(grad(criterion(*parameters), parameters[0])[0]).item()
    if ret_eig:
        min_eigenvalue_L = torch.linalg.eigvalsh(hessian(criterion, tuple(parameters))[0][0].reshape(w_L,h_L,-1).reshape(w_L*h_L,-1))[0].item()
    else:
        min_eigenvalue_L = 0

    return losses, grad_norm_L, min_eigenvalue_L