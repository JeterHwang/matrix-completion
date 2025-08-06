import torch
import numpy as np
from typing import List
from src.convex import SDP
from src.nonconvex import sgd, create_loss_fn, create_proj_fn
from src.utils import spectral_initialization

def check_MC(
    counter_LR: List[np.ndarray],
    mask: np.ndarray, 
    LR: List[np.ndarray], 
    p: float, 
    r: float, 
    mu: float = None, 
    PSD: bool = False,
    verbose = False,
):
    Z = (LR[0] @ LR[-1].T)
    counter_Z = (counter_LR[0] @ counter_LR[-1].T)
    ########### Check Convex ###########
    _, _, X_star_c, dist_2_truth_c = SDP(Z, mask)
    dist_2_counter_c = np.linalg.norm(counter_Z - X_star_c)
    if verbose:
        print(X_star_c)
        print(f"Distance to Ground Truth (Convex) = {dist_2_truth_c}")
        print(f"Distance to Counter Example (Convex) = {dist_2_counter_c}")
    # convex_success = (dist_2_truth_c < 1e-6 and dist_2_truth_c < dist_2_counter_c)
    ########### Check Nonconvex ###########
    ZL = torch.tensor(LR[0])
    ZR = torch.tensor(LR[-1])
    mask = torch.tensor(mask)
    # parameters = spectral_initialization(p, r, Z, mask, PSD)
    parameters = [torch.tensor(U).requires_grad_(True) for U in counter_LR]
    if PSD:
        criterion = create_loss_fn('MC_PSD', Z=ZL, mask=mask, ZL=0, ZR=0, e=0)
    else:
        criterion = create_loss_fn('MC_PSD', Z=0, mask=mask, ZL=ZL, ZR=ZR, e=0)
    proj_fn = create_proj_fn("MC", mu=mu)
    _, grad_norm, min_eigen = sgd(
        parameters, 
        criterion,
        proj_fn,
        optim   = 'SGD', 
        iters   = 20000, 
        lr      = 1e-2,
        min_lr  = 1e-6,
        lr_sched= 'cosine',
        ret_eig = True
    )
    counter_Z = torch.tensor(counter_Z)
    Z = torch.tensor(Z)
    X_star_nc = parameters[0] @ parameters[-1].T
    dist_2_truth_nc = torch.linalg.norm(X_star_nc - Z, ord='fro').item()
    dist_2_counter_nc = torch.linalg.norm(X_star_nc - counter_Z, ord='fro').item()
    if verbose:
        print(parameters)
        print(X_star_nc.data)
        print(f"Distance to Ground Truth (Nonconvex) = {dist_2_truth_nc}")
        print(f"Distance to Counter Example (Nonconvex) = {dist_2_counter_nc}")
        print(f"Gradient norm of X^*= {grad_norm}")
        print(f"Minimum eigen value of X^* = {min_eigen}")
    # nonconvex_success = (dist_2_truth_nc < 1e-6 and dist_2_truth_nc < dist_2_counter_nc)

    if dist_2_truth_c > 1e-6:
        print(f"(Convex) Distance to ground truth = {dist_2_truth_c:.2e} > 1e-6, thus FAILED :(")
        convex_success = False
    else:
        print(f"(Convex) Distance to ground truth = {dist_2_truth_c:.2e} <= 1e-6, thus SUCCESS :)")
        convex_success = True
    
    if dist_2_truth_nc > 1e-6:
        print(f"(Non-convex) Distance to ground truth = {dist_2_truth_nc:.2e} > 1e-6, thus FAILED :(")
        nonconvex_success = False
    else:
        print(f"(Non-convex) Distance to ground truth = {dist_2_truth_nc:.2e} <= 1e-6, thus SUCCESS :)")
        nonconvex_success = True
    
    return (convex_success and (not nonconvex_success))

def check_PSSE(
    counter_X: torch.Tensor,
    P: torch.Tensor,
    z: torch.Tensor
):
    criterion = create_loss_fn('PSSE', P=P, Z=z, ZL=0, ZR=0, e=0)
    proj_fn = create_proj_fn('PSSE')
    parameters = [counter_X]
    _, grad_norm, _ = sgd(
        parameters, 
        criterion,
        proj_fn,
        optim='SGD', 
        iters=30000, 
        lr=0.0002,
        lr_sched='cosine',
    )
    error_norm = torch.linalg.norm(parameters[0].detach() - z, ord='fro')
    if error_norm > 1e-9:
        return True
    else:
        print(f"Error Norm = {error_norm} too small :(")
        return False

    