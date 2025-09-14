import torch
import time
import numpy as np
import logging
from typing import List, Dict
from tqdm import tqdm
from torch.autograd import grad
from torch.func import jacrev
from torch.autograd.functional import hessian
from torch.linalg import eigh, eigvalsh
from src.utils import (
    incoherence_proj, 
    emb2rect, 
    symbasis, 
    create_mask, 
    same_seed, 
    logits2prob
)
from src.eigen import lanczos, power_method, eigen_decomp
from src.nonconvex import create_loss_fn
from src.verify import check_MC
from src.plot import plot_loss

def create_search_proj_fn(proj_type, **kwargs):
    @torch.no_grad()
    def MC_project(X, e=None, Z=None, **args):
        # incoherence projection of X
        incoherence_proj(X, kwargs['mu'])
        # incoherence projection of Z
        if Z is not None:
            incoherence_proj(Z, kwargs['mu'])
        # norm projection
        if e is not None:
            norm_e = torch.linalg.matrix_norm(e, ord='fro')
            if norm_e > kwargs['max_norm']:
                e.data = e.data / norm_e * kwargs['max_norm']
    @torch.no_grad()
    def PSSE_project(X, e=None, Z=None, **args):
        assert len(X) % 2 != 0
        # mag_ang_proj(X, 1.2, 0.8, 90, -90)
        # if Z is not None:
        #     mag_ang_proj(Z, 1.2, 0.8, 90, -90)
        if e is not None:
            norm_e = torch.linalg.vector_norm(e, ord=2)
            if norm_e > kwargs['max_norm']:
                e.data = e.data / norm_e * kwargs['max_norm']
    if proj_type == 'MC':
        return MC_project
    elif proj_type == 'PSSE':
        return PSSE_project
    else:
        raise NotImplementedError


def create_search_loss_fn(f, loss_type='MC_sum', **kwargs): # Use MC-PSD-2
    _e = kwargs['e'] if 'e' in kwargs else 0
    _P = kwargs['P'] if 'P' in kwargs else 0
    _Z = kwargs['Z'] if 'Z' in kwargs else 0
    _Vmin = kwargs['Vmin'] if 'Vmin' in kwargs else 0.7
    _Vmax = kwargs['Vmax'] if 'Vmax' in kwargs else 1.3
    _thmin = kwargs['thmin'] if 'thmin' in kwargs else -torch.pi / 2
    _thmax = kwargs['thmax'] if 'thmax' in kwargs else torch.pi / 2

    def MC_loss_sum(X, Z=_Z, e=_e, P=_P, alpha=1, beta=1):
        assert X.size() == Z.size()
        w, h = X.size()
        f_Z = f(Z, e, Z, P)
        f_X = f(X, e, Z, P)
        diff_loss = f_Z - f_X 
        dist_loss = -torch.linalg.norm(X - Z)
        first_order_loss_X = torch.linalg.matrix_norm(grad(f(X, e, Z, P), X, create_graph=True)[0])
        second_order_loss_X = -torch.linalg.eigvalsh(hessian(f, (X, e, Z, P), create_graph=True)[0][0].reshape(w,h,-1).reshape(w*h,-1))[0]
        transform_loss = -torch.linalg.matrix_norm(X @ X.T - Z @ Z.T, ord='fro')
        sum_loss = first_order_loss_X + second_order_loss_X
        return (
            alpha * transform_loss + beta * sum_loss, 
            diff_loss, 
            first_order_loss_X, 
            second_order_loss_X, 
            f_X, 
            f_Z, 
            dist_loss,
            transform_loss,
            sum_loss
        )
    def MC_loss_max(
        X, 
        Z: torch.Tensor = _Z, 
        e: torch.Tensor = _e, 
        P: torch.Tensor = _P, 
        alpha           = 1, 
        beta            = 1,
        tau             = 1,
    ): 
        assert X.size() == Z.size()
        w, h = X.size()
        time1 = time.time()
        f_Z, _ = f(Z, e, Z, P, tau=tau, diff=False)
        f_X, sq_loss, grad_X, hess_X = f(X, e, Z, P, tau=tau, diff=True)
        # print(grad_X)
        # print(hess_X)
        # f_Z = f(Z, e, Z, P, tau=tau, diff=False)
        # f_X = f(X, e, Z, P, tau=tau, diff=False)
        transform_loss = -sq_loss
        time2 = time.time()
        # print("gradient closed form: ", grad_X)
        # print(f_X, sq_loss, grad_X, hess_X)
        diff_loss = f_Z - f_X 
        # dist_loss = -torch.linalg.norm(X - Z)
        # gradients = grad(f(X, e, Z, P), X, create_graph=True)
        # hessians = hessian(f, (X, e, Z, P), create_graph=True)
        # hessian_X = hessians[0][0].reshape(w,h,-1).reshape(w*h,-1)
        # hessian_X = (hessian_X + hessian_X.T) / 2
        
        first_order_loss_X = torch.linalg.norm(grad_X)
        # grad_X = grad(f(X, e, Z, P), X, create_graph=True)
        # print(grad_X)
        # first_order_loss_X = torch.linalg.norm(grad_X[0])
        
        time3 = time.time()
        eigvals = eigvalsh(hess_X)
        second_order_loss_X = -eigvals[0]
        # second_order_loss_X = -lanczos(f, X, e, Z, P)
        time4 = time.time()
        # eigen_decomp(f, X, e, Z, P) # torch.tensor([0.0], dtype=torch.float64) #
        # print(second_order_loss_X)
        
        # transform_loss = -torch.linalg.matrix_norm(X @ X.T - Z @ Z.T, ord='fro')
        
        # print(f"Compute loss, grad, hess: {time2 - time1} (s)")
        # print(f"Compute norm: {time3 - time2} (s)")
        # print(f"Compute eigen decomposition: {time4 - time3}")
        # print(first_order_loss_X, second_order_loss_X)
        max_loss = torch.maximum(first_order_loss_X, second_order_loss_X)
        return (
            max_loss, 
            diff_loss, 
            first_order_loss_X, 
            second_order_loss_X, 
            f_X, 
            f_Z, 
            # dist_loss,
            transform_loss,
            max_loss
        )
    
    def PSSE_loss_sum(X, e=_e, Z=_Z, P=_P, alpha=1, beta=1): # Use PSSE-1
        w, h = X.size()
        f_X = f(X, e, Z, P)
        f_Z = f(Z, e, Z, P)
        diff_loss = f_Z - f_X
        transform_loss = -torch.linalg.matrix_norm(X @ X.T - Z @ Z.T, ord='fro')
        dist_loss = -torch.linalg.norm(X - Z)
        first_order_loss = torch.linalg.norm(grad(f(X, e, Z, P), X, create_graph=True)[0])
        second_order_loss = -torch.linalg.eigvalsh(hessian(f, (X, e, Z, P), create_graph=True)[0][0].reshape(w,h,-1).reshape(w*h,-1))[0]
        sum_loss = first_order_loss + second_order_loss
        return (
            alpha * transform_loss + beta * sum_loss, 
            diff_loss, 
            first_order_loss, 
            second_order_loss,
            f_X, 
            f_Z, 
            dist_loss,
            transform_loss,
            sum_loss
        )
    def PSSE_loss_max(
            X       : torch.Tensor, 
            e       : torch.Tensor  = _e, 
            Z       : torch.Tensor  = _Z, 
            P       : torch.Tensor  = _P, 
            alpha   : float         = 1,        # Not optimized
            beta    : float         = 1,        # Not optimized
            tau     : float         = 1,
            Vmin    : float         = _Vmin,    # Not optimized
            Vmax    : float         = _Vmax,    # Not optimized
            thmin   : float         = _thmin,   # Not optimized
            thmax   : float         = _thmax,   # Not optimized
        ): # Use PSSE-1
        #################
        f_X, sq_loss, grad_X, hess_X = f(X, e, Z, P, tau=tau, Vmin=Vmin, Vmax=Vmax, thmin=thmin, thmax=thmax, diff=True)
        f_Z, _ = f(Z, e, Z, P, tau=tau, Vmin=Vmin, Vmax=Vmax, thmin=thmin, thmax=thmax, diff=False)
        diff_loss = f_Z - f_X
        transform_loss = -sq_loss
        first_order_loss_X = torch.linalg.norm(grad_X)
        eigvals = eigvalsh(hess_X)
        second_order_loss_X = -eigvals[0]
        max_loss = torch.maximum(first_order_loss_X, second_order_loss_X)
        return (
            alpha * transform_loss + beta * max_loss, 
            diff_loss, 
            first_order_loss_X, 
            second_order_loss_X,
            f_X, 
            f_Z, 
            # dist_loss,
            transform_loss,
            max_loss
        )
    
    if loss_type == 'MC_sum':
        return MC_loss_sum
    elif loss_type == 'MC_max':
        return MC_loss_max
    elif loss_type == 'PSSE_sum':
        return PSSE_loss_sum
    elif loss_type == 'PSSE_max':
        return PSSE_loss_max
    else:
        raise NotImplementedError
    
def search(
    parameters: Dict[str, torch.Tensor], 
    # mask: torch.Tensor, 
    # Z: torch.Tensor, 
    # coeff: torch.Tensor,
    criterion,
    project_fn,
    optim: str      = 'SGD', 
    iters: int      = 100, 
    lr: float       = 0.001,
    min_lr: float   = 5e-5,
    lr_sched: str   = 'static',
    T: float        = 1, # Softmax temperature
    trans_bound:tuple = (-1, -100)
    # clip_grad: bool = False,
    # loss_scale: str = 'linear',
):
    parameter_groups = []
    for key, val in parameters.items():
        parameter_groups.append({
            'name': key,
            'params': [val],
            'lr': lr
        })
    if optim == 'Adam':
        optimizer_W = torch.optim.Adam(parameter_groups, lr=lr)
        # optimizer_coeff = torch.optim.Adam([coeff], lr=0.025)
    elif optim == 'AdamW':
        optimizer_W = torch.optim.AdamW(parameter_groups, lr=lr)
        # optimizer_coeff = torch.optim.Adam([coeff], lr=0.025)
    elif optim == 'SGD':
        optimizer_W = torch.optim.SGD(parameter_groups, lr=lr, momentum=0.9, weight_decay=1e-4)
        # optimizer_coeff = torch.optim.Adam([coeff], lr=0.025)
    else:
        raise NotImplementedError
    losses = []
    # Pre-test
    loss_0, diff_0, grad_X_0, hess_X_0, f_X_0, f_Z_0, trans_0, max_loss_0 = criterion(**parameters)
    losses.append((loss_0.item(), diff_0.item(), grad_X_0.item(), hess_X_0.item(), f_X_0.item(), f_Z_0.item(), trans_0.item()))
    # T = torch.sum(coeff.detach())
    alpha, beta, tau = 1, 1, 1
    # SGD
    with tqdm(total=iters, leave=False) as pbar:
        for i in range(iters):
            # alpha, beta = calc_loss_coeff(alpha_0, beta_0, coeff[0], coeff[1], i, iters, loss_scale)
            # alpha = coeff[0] 
            # beta = abs(trans / max(grad_X, hess_X)) * beta
            # alpha = coeff.detach()[0]
            # beta = coeff.detach()[1]
            time1 = time.time()
            loss, diff, grad_X, hess_X, f_X, f_Z, trans, max_loss = criterion(**parameters, alpha=alpha, beta=beta, tau=tau)
            # print(f"Max Loss = {loss}")
            # print(parameters, f_X.item(), f_Z.item(), trans.item(), grad_X.item(), hess_X.item(), max_loss.item())
            time2 = time.time()
            inputs = [parameters['X'], parameters['Z']] if 'Z' in parameters else [parameters['X']]
            gradients_1 = grad(outputs=trans, inputs=inputs, retain_graph=True, allow_unused=True)
            gradients_2 = grad(outputs=max_loss, inputs=inputs, retain_graph=True, allow_unused=True)
            optimizer_W.zero_grad()
            time3 = time.time()
            loss.backward()
            # print(parameter_groups[0]['params'][0].grad)
            # print(parameter_groups[1]['params'][0].grad)
            time4 = time.time()
            # print(f"forward pass: {time2 - time1} (s)")
            # print(f"Compute gradients: {time3 - time2} (s)")
            # print(f"backward pass: {time4 - time3} (s)")
            # if clip_grad:
            #     torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, norm_type=2)
            
            grad_norm_1 = torch.sqrt(sum([0 if GG is None else torch.linalg.norm(GG).square() for GG in gradients_1]))
            grad_norm_2 = torch.sqrt(sum([0 if GG is None else torch.linalg.norm(GG).square() for GG in gradients_2]))
            
            # if i % 1000 == 0:
            #     print(grad_norm_1, grad_norm_2)
            # g2 = coeff[1] * grad_norm_2.detach()  
            # g_bar = 0.5 * (g1 + g2)
            # print(grad_norm_1.item(), grad_norm_2.item())
            # g1 = grad_norm_2.item() / (grad_norm_1.item() + grad_norm_2.item())
            # g2 = grad_norm_1.item() / (grad_norm_1.item() + grad_norm_2.item())
            
            # boost = 0.5 * (1 + 3) + 0.5 * (3 - 1) * np.cos(np.pi * i / iters)
            r1 = (trans.item() - trans_0.item()) / (trans_bound[0] - trans_0.item())
            r2 = (max_loss_0.item() - max_loss.item()) / max_loss_0.item() # * boost
            alpha = grad_norm_2.item() / grad_norm_1.item() * (np.exp(r1 * T) / (np.exp(r1 * T) + np.exp(r2 * T)))
            beta =  (np.exp(r2 * T) / (np.exp(r1 * T) + np.exp(r2 * T)))
            tau = 0.5 * (0.5 + 0.05) + 0.5 * (0.5 - 0.05) * np.cos(np.pi * i / iters)
            # alpha = 0.01
            # beta = 100
            # r1 = trans.item() / trans_0          # f1_init cached at step 0
            # r2 = max_loss.item() / max_loss_0
            # r_bar = 0.5 * (r1 + r2)
            # t1 = g_bar * (r1 / r_bar) ** 0.5
            # t2 = g_bar * (r2 / r_bar) ** 0.5
            # G = (torch.abs(g1 - t1.detach()) + torch.abs(g2 - t2.detach()))
            # optimizer_coeff.zero_grad()
            # G.backward()

            optimizer_W.step()
            # optimizer_coeff.step()
            # with torch.no_grad():
            #     coeff.clamp_(min=1e-3)
            #     coeff.mul_(T / coeff.sum())
                
            # Adjust learning rate
            if lr_sched == 'cosine':
                new_lr = 0.5 * (lr + min_lr) + 0.5 * (lr - min_lr) * np.cos(np.pi * i / iters)
            elif lr_sched == 'static':
                new_lr = lr
            elif lr_sched == 'step':
                new_lr = lr * ((0.25) ** (i * 3 // iters))
            else:
                raise NotImplementedError
            for param_group in optimizer_W.param_groups:
                if param_group['name'] == 'X':
                    param_group['lr'] = new_lr
                elif param_group['name'] == 'P':
                    param_group['lr'] = new_lr
                else:
                    param_group['lr'] = new_lr * 0.3
            # Project to feasible set
            project_fn(**parameters)
            # Evaluating
            losses.append((loss.item(), diff.item(), grad_X.item(), hess_X.item(), f_X.item(), f_Z.item(), trans.item()))
            # Early Stop
            if trans > trans_bound[0]:
                logging.warning("transform loss too small !!")
                break
            elif trans < trans_bound[1]:
                logging.warning("transform loss explodes !!")
                break
            pbar.set_description(f"lr={new_lr:.6f}, diff={diff.item():.2f}, trans={trans.item():.2f}, grad={grad_X.item():.2f}, hess={hess_X.item():.2f}, coeff=({alpha:.2f},{beta:.2f})")
            pbar.update(1)
    return losses

def compute_X_Z_e(
    variables, 
    top_k, 
    n, 
    r, 
    mu, 
    e_norm, 
    loss_type   = 'MC_max', 
    optim       = 'Adam', 
    iters       = 100, 
    lr          = 0.001, 
    min_lr      = 5e-5, 
    lr_sched    = 'cosine', 
    T           = 1, 
    trans_bound = (-1e-2, -1e2),
    M_type      = 'STE',
    device      = torch.device('cpu'),
):
    parameters = {}
    for name, val  in variables.items():
        if name == 'X':
            X = torch.normal(mean=0.0, std=torch.ones((n, r))).to(device).requires_grad_()
            parameters['X'] = X
        elif name == 'Z':
            if val is None:
                Z = torch.normal(mean=0.0, std=torch.ones((n, r))).to(device).requires_grad_()
                parameters['Z'] = Z
            else:
                Z = val.to(device)
        elif name == 'e':
            if val is None:
                e = torch.zeros((n, n), requires_grad=True, device=device)
                parameters['e'] = e
            else:
                e = val.to(device)
        elif name == 'M':
            Ps = symbasis(n)
            M = val
            Ps_AT = torch.sparse.mm(Ps, M.T)
            Ps, M, Ps_AT = Ps.to(device), M.to(device), Ps_AT.to(device)
    # check whether we need to train the mask
    P = torch.rand(n*n, device=device)
    if top_k > 0:
        P.requires_grad_()
        parameters['P'] = P
    
    loss_fn = create_loss_fn(
        'MC_PSD', 
        P       = P.detach(),
        Z       = Z.detach(),
        e       = e.detach(),
        top_k   = top_k,
        M_type  = M_type,
        M       = M,
        sym     = Ps,
        Ps_AT   = Ps_AT,
    )
    criterion = create_search_loss_fn(
        loss_fn, 
        loss_type, 
        P = P.detach(),
        Z = Z.detach(),
        e = e.detach(),
    )
    proj_fn = create_search_proj_fn(
        'MC', 
        mu = mu, 
        max_norm = e_norm
    )
    proj_fn(**parameters)
    losses = search(
        parameters,
        criterion,
        proj_fn,
        optim,
        iters,
        lr,
        min_lr,
        lr_sched,
        T,
        trans_bound,
    )
    
    return parameters, losses

def search_counter(
    variables, 
    top_k, 
    n,
    e_norm, 
    A,                          # COO format
    loss_type   = 'PSSE_max', 
    optim       = 'Adam', 
    iters       = 100, 
    lr          = 0.001, 
    min_lr      = 5e-5, 
    lr_sched    = 'cosine', 
    T           = 1, 
    trans_bound = (-1, -1e-2), 
    constraints = (0.3, 1, -torch.pi/2, torch.pi/2),
    M_type      = 'STE'
):
    # x_0 = PSSE_initialization(z, 1.0)
    # x = torch.tensor(np.concatenate([x_0.real, x_0.imag]), requires_grad=True, dtype=torch.float64)
    d = A.size(0)
    parameters = {}
    for name, val  in variables.items():
        if name == 'X':
            X = torch.normal(0.0, 3.0 * torch.ones((n, 1))).requires_grad_()
            parameters['X'] = X
        elif name == 'Z':
            if val is None:
                Z = torch.normal(0.0, 3.0 * torch.ones((n, 1))).requires_grad_()
                parameters['Z'] = Z
            else:
                Z = val
        elif name == 'e':
            if val is None:
                e = torch.zeros(len(A), requires_grad=True)
                parameters['e'] = e
            else:
                e = val
        elif name == 'M':
            M = val
        
    # check whether we need to train the mask
    P = torch.rand(d, dtype=torch.float64)
    if top_k > 0:
        P.requires_grad_()
        parameters['P'] = P
    
    loss_fn = create_loss_fn(
        'PSSE', 
        P       = P.detach(), 
        Z       = Z.detach(), 
        e       = e.detach(),
        top_k   = top_k,
        A       = A,
        M_type  = M_type,
        M       = M.detach(),
        sym     = symbasis(n),
        diff    = True
    )
    criterion = create_search_loss_fn(
        loss_fn, 
        loss_type, 
        P = P.detach(),
        Z = Z.detach(), 
        e = e.detach(),
        Vmin = constraints[0],
        Vmax = constraints[1],
        thmin = constraints[2],
        thmax = constraints[3],
    )
    proj_fn = create_search_proj_fn('PSSE', max_norm=e_norm)
    proj_fn(**parameters)
    losses = search(
        parameters,
        criterion,
        proj_fn,
        optim,
        iters,
        lr,
        min_lr,
        lr_sched,
        T,
        trans_bound
    )
    return parameters, losses

def DSE_MC(
    top_k, 
    search_loops, 
    device,
    n, 
    r, 
    mu, 
    e_norm,  
    loss_type   = 'A', 
    optim       = 'Adam', 
    iters       = 100, 
    lr          = 0.001, 
    min_lr      = 5e-5, 
    lr_sched    = 'cosine', 
    T           = 1, 
    trans_bound = (-1e-1, -1e2),
    M_type      = 'STE',
    M           = None,         # COO
    save_path   = None,
):
    npy_path = save_path / "npys"
    fig_path = save_path / "figs"
    npy_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(search_loops):
        logging.info(f"========== Search Loop {i} ==========")
        variables = {
            "X": None,
            "Z": None,
            "e": None,
            "M": M,
        }
        _top_k = top_k - len(M.values())
        parameters, losses = compute_X_Z_e(
            variables,
            _top_k,
            n, 
            r,  
            mu,
            e_norm, 
            loss_type, 
            optim, 
            iters, 
            lr, 
            min_lr,
            lr_sched,
            T,
            trans_bound,
            M_type,
            device
        ) 
        plot_loss(fig_path / f"iter-{i}.png", losses)
        if len(losses) < iters + 1: # Early Stop
            continue
        # print_counter_MC(f"Search Loop {i}", _top_k, parameters, losses, M=mask, M_type=M_type)
        flag, categories = check_MC(
            parameters,
            _top_k,
            mu, 
            PSD         = True,
            M           = M,
            M_type      = M_type,
            verbose     = True,
            save_path   = npy_path / f"iter-{i}.npy"
        )
        if flag:
            logging.info(categories)

def DSE_PSSE(
    top_k, 
    search_loops, 
    e_norm, 
    A,
    loss_type       = 'PSSE_max', 
    optim           = 'Adam', 
    iters           = 100, 
    lr              = 0.001, 
    min_lr          = 5e-5, 
    lr_sched        = 'cosine', 
    T               = 1, 
    trans_bound     = (-1, -1e2), 
    constraints     = (0.3, 1, -torch.pi/2, torch.pi/2),
    M_type          = 'STE',
    M               = None,
    save_path   = None,
):
    Pareto = []
    for i in range(search_loops):
        logging.info(f"================== Search Iteration : {i} ==================")
        # Search for (X, Z, e)
        variables = {
            "X": None,
            "Z": None,
            "e": None,
            "M": M,
        }
        _top_k = top_k - len(M.values())
        parameters, losses = search_counter(
            variables,
            _top_k, 
            e_norm, 
            A,
            loss_type, 
            optim, 
            iters, 
            lr, 
            min_lr,
            lr_sched, 
            T,
            trans_bound,
            constraints,
            M_type,
        )
        if len(losses) < iters + 1: # Early Stop
            continue
        loss_fn = create_loss_fn(
            'PSSE', 
            P       = parameters['P'].detach() if 'P' in parameters else M, 
            Z       = parameters['Z'].detach(), 
            top_k   = _top_k,
            A       = A,
            is_rect = False,
            Vmin    = constraints[0],
            Vmax    = constraints[1],
            thmin   = constraints[2],
            thmax   = constraints[3],
            M       = M,
            M_type  = M_type
        )
        proj_fn = create_proj_fn('PSSE')
        plot_contour(f"search-{i} (before)", parameters['X'].detach(), parameters['Z'].detach(), loss_fn, constraints)
        flag = check_PSSE(  # fist check
            parameters['X'], 
            parameters['Z'],
            loss_fn,
            proj_fn,
            constraints
        )
        if flag:  
            categories = check_random( # double check
                50,
                parameters['X'],
                parameters['Z'],
                loss_fn,
                proj_fn,
                constraints,
            )
            logging.info(categories)
            if categories['X'] != 0:
                logging.info(f"{'=> Total loss':<27} {'=':^1} {losses[-1][0]:>10.2e}")
                logging.info(f"{'=> f(Z) - f(X)':<27} {'=':^1} {losses[-1][1]:>10.2e}")
                logging.info(f"{'=> |gradient(X)|':<27} {'=':^1} {losses[-1][2]:>10.2e}")
                logging.info(f"{'=> -lambda_min(hessian(X))':<27} {'=':^1} {losses[-1][3]:>10.2e}")
                logging.info(f"{'=> |XX^T-ZZ^T|_F':<27} {'=':^1} {losses[-1][6]:>10.2e}")
                logging.info(f"{'=> |X-Z|_F':<27} {'=':^1} {torch.linalg.norm(emb2rect(parameters['X'], *constraints) - emb2rect(parameters['Z'], *constraints), ord='fro'):>10.2e}")
                Pareto.append((categories, copy.deepcopy(parameters)))
    return Pareto
    