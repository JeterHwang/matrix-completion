import torch
import numpy as np
import logging
import copy
from pathlib import Path
from torch.autograd import grad
from torch.autograd.functional import hessian
from typing import List, Dict
from src.eigen import lanczos
from src.convex import SDP
from src.nonconvex import sgd, create_loss_fn, create_proj_fn
from src.utils import spectral_initialization, logits2prob, create_mask, emb2rect, symbasis

def check_MC(
    parameters: Dict[str, torch.Tensor],
    top_k       : int, 
    mu          : float         = None, 
    PSD         : bool          = False,
    verbose     : bool          = False,
    M           : torch.Tensor  = None,
    M_type      : str           = 'STE',
    save_path   : Path          = None
):
    if PSD:
        Z = parameters['Z']
        X = parameters['X']
        GT = Z.detach() @ Z.detach().T
        # counter =  X.detach() @ X.detach().T
    else:
        ZL = parameters['ZL']
        ZR = parameters['ZR']
        XL = parameters['XL']
        XR = parameters['XR']
        GT = ZL.detach() @ ZR.detach().T
        # counter =  XL.detach() @ XR.detach().T
    e = parameters['e']
    P = parameters['P'] if 'P' in parameters else torch.zeros_like(e).reshape(-1)

   ########### Check Convex ###########
    prob_mat = logits2prob(P, M_type)
    mask_coo = create_mask(prob_mat, top_k, M_type, M)
    zeros = torch.zeros_like(e).reshape(-1)
    zeros[mask_coo.indices()] = mask_coo.values()
    mask = zeros.reshape(e.size()).to(torch.long).detach().numpy()
    # _, _, X_star_c, dist_2_truth_c = SDP(GT.numpy(), mask)
    # dist_2_counter_c = np.linalg.norm(counter.numpy() - X_star_c)
    # if verbose:
        # print(X_star_c)
        # logging.info(f"Distance to Ground Truth (Convex) = {dist_2_truth_c}")
        # print(f"Distance to Counter Example (Convex) = {dist_2_counter_c}")
    ########### Check Nonconvex ###########
    
    if PSD:
        sgd_parameters = {'X': X}
        criterion = create_loss_fn(
            'MC_PSD', 
            P       = P.detach(),
            Z       = Z.detach(),
            e       = e.detach(),
            top_k   = top_k,
            M       = mask_coo.detach(),
            M_type  = M_type,
            sym     = symbasis(e.size(0)),
            diff    = False
        )
    else: # TODO
        sgd_parameters = {'XL': XL, 'XR': XR} 
        criterion = create_loss_fn(
            'MC',
            P       = P.detach(),
            ZL      = ZL.detach(),
            ZR      = ZR.detach(),
            e       = e.detach(),
            top_k   = top_k,
            M       = mask_coo.detach(),
            M_type  = M_type,
        )
    proj_fn = create_proj_fn(
        "MC", 
        mu = mu
    )
    loss, sq_dist_2_truth_nc, grad_norm, min_eigen = sgd(
        sgd_parameters, 
        criterion,
        proj_fn,
        optim   = 'L-BFGS', 
        iters   = 5000, 
        lr      = 1e-2,
        min_lr  = 1e-6,
        lr_sched= 'cosine',
        ret_eig = True
    )
    dist_2_truth_nc = torch.linalg.norm(sgd_parameters['X'] - Z)
    # if PSD:
    #     X_star_nc = sgd_parameters['X'] @ sgd_parameters['X'].T
    # else:
    #     X_star_nc = sgd_parameters['XL'] @ sgd_parameters['XR'].T
    if verbose:
        # logging.info(sgd_parameters)
        # logging.info(X_star_nc.data)
        logging.info("\n")
        logging.info(f"Distance to Ground Truth (Nonconvex) = {dist_2_truth_nc}")
        logging.info(f"Square distance to Ground Truth (Nonconvex) = {sq_dist_2_truth_nc}")
        logging.info(f"Loss value of X^* = {loss}")
        logging.info(f"Gradient norm of X^* = {grad_norm}")
        logging.info(f"Minimum eigen value of X^* = {min_eigen}")
        logging.info("\n")
    
    ###### Print check results ######
    # if dist_2_truth_c > 1e-4:
    #     logging.info(f"(Convex) Distance to ground truth = {dist_2_truth_c:.2e} > 1e-4, thus FAILED :(")
    #     convex_success = False
    # else:
    #     logging.info(f"(Convex) Distance to ground truth = {dist_2_truth_c:.2e} <= 1e-4, thus SUCCESS :)")
    #     convex_success = True
    convex_success = True
    
    if sq_dist_2_truth_nc > 1e-2:
        logging.info(f"(Non-convex) Square distance to ground truth = {sq_dist_2_truth_nc:.2e} > 1e-2, thus FAILED :(")
        nonconvex_success = False
    else:
        logging.info(f"(Non-convex) Square distance to ground truth = {sq_dist_2_truth_nc:.2e} <= 1e-2, thus SUCCESS :)")
        nonconvex_success = True
    
    if convex_success and (not nonconvex_success):
        np.save(
            save_path,
            {
                'counter': parameters['X'].detach().numpy(),
                'Z': parameters['Z'].detach().numpy(),
                'M': mask
            }
        )
        categories = check_random_MC(
            50,
            parameters['X'],
            parameters['Z'],
            criterion,
            proj_fn,
        )
        return True, categories
    else:
        return False, {}


def check_random_MC(
    iters: int,
    X: torch.Tensor,
    Z: torch.Tensor,
    loss_fn,
    proj_fn,
):
    categories = {'X': 0, 'Z': 0, 'Others': 0}
    for _ in range(iters):
        random_X = torch.normal(0, torch.ones_like(X)).requires_grad_()
        parameters = {'X': random_X}
        proj_fn(**parameters)
        sgd(
            parameters, 
            loss_fn,
            proj_fn,
            optim       = 'L-BFGS', 
            iters       = 2000, 
            lr          = 1e-2,
            lr_sched    = 'cosine',
            ret_eig     = False
        )
        with torch.no_grad():
            Xt = parameters['X']
            dist_2_GT = torch.linalg.norm(Xt - Z, ord='fro')
            dist_2_counter = torch.linalg.norm(Xt - X, ord='fro')
            # print(dist_2_GT.item(), dist_2_counter.item())
        if dist_2_counter < 1e-2:
            categories['X'] += 1
        elif dist_2_GT < 1e-2:
            categories['Z'] += 1
        else:
            categories['Others'] += 1
    
    return categories

def check_PSSE(
    parameters: Dict[str, torch.Tensor],
    A           : torch.Tensor,
    top_k       : int, 
    verbose     : bool          = False,
    M           : torch.Tensor  = None,
    M_type      : str           = 'STE',
    constraints : tuple         = (0.3, 1, -torch.pi/2, torch.pi/2),
    save_path   : Path          = None
):
    X = parameters['X']
    Z = parameters['Z']
    e = parameters['e']
    P = parameters['P'] if 'P' in parameters else torch.zeros_like(e)
    prob_mat = logits2prob(P, M_type)
    mask = create_mask(prob_mat, top_k, M_type, M)

    sgd_parameters = {'X': X}
    loss_fn = create_loss_fn(
        'PSSE', 
        P       = P.detach(),
        Z       = Z.detach(), 
        top_k   = top_k,
        A       = A,
        is_rect = False,
        Vmin    = constraints[0],
        Vmax    = constraints[1],
        thmin   = constraints[2],
        thmax   = constraints[3],
        M       = mask.detach(),
        M_type  = M_type
    )
    proj_fn = create_proj_fn('PSSE')
    losses = sgd(
        sgd_parameters, 
        loss_fn,
        proj_fn,
        optim       = 'L-BFGS', 
        iters       = 2000, 
        lr          = 1e-2,
        min_lr      = 1e-6,
        lr_sched    = 'cosine',
        ret_eig     = False
    )
    counter_X = sgd_parameters['X']
    counter_X_rect = emb2rect(counter_X, *constraints)
    Z_rect = emb2rect(Z, *constraints)
    orig = copy.deepcopy(loss_fn.__defaults__)
    loss_fn.__defaults__ = orig[:1] + (Z_rect,) + orig[2:5] + (True,) + orig[6:]
    
    grad_norm = torch.linalg.matrix_norm(grad(loss_fn(X=counter_X_rect), counter_X_rect)[0]).item()
    min_eig = lanczos(loss_fn, counter_X_rect, e, Z_rect, P).item()
    error_norm = torch.linalg.norm(counter_X_rect - Z_rect, ord='fro')
    
    loss_fn.__defaults__ = orig
    if verbose:
        logging.info(f"Gradient Norm of X_t = {grad_norm}")
        logging.info(f"|X_t - Z|_F = {error_norm}")
        logging.info(f"f(X_t) = {losses[-1]}")
        logging.info(f"\lamb_min(hess(X_t)) = {min_eig}")
    if error_norm > 1e-2 and grad_norm < 1e-2:
        logging.info("SUCCEED :)")
        np.save(
            save_path,
            {
                'counter': counter_X_rect.detach().numpy(),
                'Z': Z_rect.detach().numpy(),
                'M': mask.detach().numpy()
            }
        )
        categories = check_random_PSSE(
            50,
            counter_X,
            Z,
            loss_fn,
            proj_fn,
            constraints
        )
        return True, categories
    else:
        logging.info("FAILED :(")
        return False, {}

def check_random_PSSE(
    iters: int,
    X: torch.Tensor,
    Z: torch.Tensor,
    loss_fn,
    proj_fn,
    constraints
):
    categories = {'X': 0, 'Z': 0, 'Others': 0}
    for _ in range(iters):
        random_X = torch.normal(0, 3 * torch.ones_like(X)).requires_grad_()
        parameters = {'X': random_X}
        proj_fn(**parameters)
        sgd(
            parameters, 
            loss_fn,
            proj_fn,
            optim       = 'L-BFGS', 
            iters       = 2000, 
            lr          = 1e-2,
            lr_sched    = 'cosine',
        )
        with torch.no_grad():
            Xt_rect = emb2rect(parameters['X'], *constraints)
            X_rect = emb2rect(X, *constraints)
            Z_rect = emb2rect(Z, *constraints)
            dist_2_GT = torch.linalg.norm(Xt_rect - Z_rect, ord='fro')
            dist_2_counter = torch.linalg.norm(Xt_rect - X_rect, ord='fro')
            # print(dist_2_GT.item(), dist_2_counter.item())
        if dist_2_counter < 5e-2:
            categories['X'] += 1
        elif dist_2_GT < 5e-2:
            categories['Z'] += 1
        else:
            categories['Others'] += 1
    
    return categories

