import torch 
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from torch.autograd import grad
from torch.autograd.functional import hessian
from src.utils import incoherence_proj, mag_ang_proj, create_mask

def create_loss_fn(loss_type, **kwargs):
    _e = kwargs['e'] if 'e' in kwargs else 0
    _P = kwargs['P'] if 'P' in kwargs else 0
    _ZL = kwargs['ZL'] if 'ZL' in kwargs else 0
    _ZR = kwargs['ZR'] if 'ZR' in kwargs else 0
    _Z = kwargs['Z'] if 'Z' in kwargs else 0
    _A = kwargs['A'] if 'A' in kwargs else 0
    _k = kwargs['top_k'] if 'top_k' in kwargs else 100000000

    def MC_loss_fn(
        XL: torch.Tensor, 
        XR: torch.Tensor, 
        e: torch.Tensor = _e, 
        ZL: torch.Tensor = _ZL, 
        ZR: torch.Tensor = _ZR, 
        P: torch.Tensor = _P,
        k: int = _k             # Not optimized
    ):
        mask = create_mask(P, k)
        return torch.square(((mask - P).detach() + P) * (X_L @ X_R.T - Z_L @ Z_R.T + e)).sum() / 4
    
    def MC_loss_fn_PSD(
        X: torch.Tensor, 
        e: torch.Tensor = _e, 
        Z: torch.Tensor = _Z,
        P: torch.Tensor = _P,
        k: int = _k             # Not optimized
    ):
        mask = create_mask(P, k)
        return torch.square(((mask - P).detach() + P) * (X @ X.T - Z @ Z.T + e)).sum() / 4
    
    def PSSE_loss_fn(
        X: torch.Tensor, 
        e: torch.Tensor = _e, 
        Z: torch.Tensor = _Z,
        P: torch.Tensor = _P,
        k: int = _k,            # Not optimized
        A: torch.Tensor = _A,   # Not optimized
    ):
        mask = create_mask(P, k)
        pred = (X.T.unsqueeze(0) @ A @ X.unsqueeze(0)).squeeze()
        gt = (Z.T.unsqueeze(0) @ A @ Z.unsqueeze(0)).squeeze()
        masked = ((mask - P).detach() + P) * (pred - gt + e)
        # print(pred-gt+e)
        # print(masked)
        # print(masked)
        return masked.square().sum() / 2
    
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
    def PSSE_project(X):
        with torch.no_grad():
            if len(X) % 2 == 0:
                X[len(X) // 2] = 0
            mag_ang_proj(X, 1.2, 0.8, 90, -90)
    if task == 'MC':
        return MC_project
    elif task == 'PSSE':
        return PSSE_project
    else:
        raise NotImplementedError

def sgd(
    parameters: Dict[str, torch.Tensor], 
    criterion,
    project_fn,
    optim: str      = 'SGD', 
    iters: int      = 100, 
    lr: float       = 0.001,
    min_lr: float   = 1e-6,
    lr_sched: str   = 'static',
    ret_eig: bool   = False,
):
    # Initialization
    if optim == 'SGD':
        optimizer = torch.optim.SGD(list(parameters.values()), lr=lr)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(list(parameters.values()), lr=lr)
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(list(parameters.values()), lr=lr)
    else:
        raise NotImplementedError
    # SGD
    losses = []
    with tqdm(total=iters, leave=False) as pbar:
        for i in range(iters):
            optimizer.zero_grad()
            loss = criterion(**parameters)
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
                project_fn(**parameters)
            pbar.update(1)
            pbar.set_description(f"lr={new_lr:.6f}, loss={losses[-1]:.9f}")
    
    w_L, h_L = parameters['X'].size() 
    grad_norm_L = torch.linalg.matrix_norm(grad(criterion(**parameters), parameters['X'])[0]).item()
    if ret_eig:
        min_eigenvalue_L = torch.linalg.eigvalsh(hessian(criterion, tuple(parameters.values()))[0][0].reshape(w_L,h_L,-1).reshape(w_L*h_L,-1))[0].item()
    else:
        min_eigenvalue_L = 0

    return losses, grad_norm_L, min_eigenvalue_L