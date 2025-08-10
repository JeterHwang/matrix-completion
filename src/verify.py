import torch
import numpy as np
from typing import List, Dict
from src.convex import SDP
from src.nonconvex import sgd, create_loss_fn, create_proj_fn
from src.utils import spectral_initialization, create_mask

def check_MC(
    parameters: Dict[str, torch.Tensor],
    top_k: int, 
    mu: float = None, 
    PSD: bool = False,
    verbose: bool = False,
    P : torch.Tensor = None,
):
    if PSD:
        Z = parameters['Z']
        X = parameters['X']
        GT = Z.detach() @ Z.detach().T
        counter =  X.detach() @ X.detach().T
    else:
        ZL = parameters['ZL']
        ZR = parameters['ZR']
        XL = parameters['XL']
        XR = parameters['XR']
        GT = ZL.detach() @ ZR.detach().T
        counter =  XL.detach() @ XR.detach().T
    P = parameters['P'] if 'P' in parameters else P
    e = parameters['e']

   ########### Check Convex ###########
    mask = create_mask(P, top_k).to(torch.long).detach().numpy()
    _, _, X_star_c, dist_2_truth_c = SDP(GT.numpy(), mask)
    dist_2_counter_c = np.linalg.norm(counter.numpy() - X_star_c)
    if verbose:
        print(X_star_c)
        print(f"Distance to Ground Truth (Convex) = {dist_2_truth_c}")
        print(f"Distance to Counter Example (Convex) = {dist_2_counter_c}")
    ########### Check Nonconvex ###########
    if PSD:
        sgd_parameters = {'X': X}
        criterion = create_loss_fn(
            'MC_PSD', 
            P = P.detach(),
            Z = Z.detach(),
            e = e.detach(),
            top_k = top_k
        )
    else:
        sgd_parameters = {'XL': XL, 'XR': XR} 
        criterion = create_loss_fn(
            'MC',
            P = P.detach(),
            ZL = ZL.detach(),
            ZR = ZR.detach(),
            e = e.detach(),
            top_k = top_k
        )
    proj_fn = create_proj_fn(
        "MC", 
        mu = mu
    )
    _, grad_norm, min_eigen = sgd(
        sgd_parameters, 
        criterion,
        proj_fn,
        optim   = 'SGD', 
        iters   = 20000, 
        lr      = 1e-2,
        min_lr  = 1e-6,
        lr_sched= 'cosine',
        ret_eig = True
    )
    if PSD:
        X_star_nc = sgd_parameters['X'] @ sgd_parameters['X'].T
    else:
        X_star_nc = sgd_parameters['XL'] @ sgd_parameters['XR'].T
    dist_2_truth_nc = torch.linalg.norm(X_star_nc - GT, ord='fro').item()
    dist_2_counter_nc = torch.linalg.norm(X_star_nc - counter, ord='fro').item()
    if verbose:
        print(sgd_parameters)
        print(X_star_nc.data)
        print(f"Distance to Ground Truth (Nonconvex) = {dist_2_truth_nc}")
        print(f"Distance to Counter Example (Nonconvex) = {dist_2_counter_nc}")
        print(f"Gradient norm of X^*= {grad_norm}")
        print(f"Minimum eigen value of X^* = {min_eigen}")
    
    ###### Print check results ######
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
    A: torch.Tensor,
    Z: torch.Tensor,
    P: torch.Tensor,
    top_k: int
):
    criterion = create_loss_fn(
        'PSSE', 
        P = P.detach(), 
        Z = Z.detach(), 
        top_k = top_k,
        A = A,
    )
    proj_fn = create_proj_fn('PSSE')
    parameters = {'X': counter_X}
    _, grad_norm, _ = sgd(
        parameters, 
        criterion,
        proj_fn,
        optim='SGD', 
        iters=30000, 
        lr=0.0002,
        lr_sched='cosine',
    )
    error_norm = torch.linalg.norm(parameters['X'].detach() - Z, ord='fro')
    if error_norm > 1e-9:
        return True
    else:
        print(f"Error Norm = {error_norm} too small :(")
        return False

    