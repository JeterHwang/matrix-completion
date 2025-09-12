import torch 
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from torch.autograd import grad
from torch.autograd.functional import hessian
from torch.func import jacrev
from torch_sparse import spspmm, spmm, transpose
from src.utils import incoherence_proj, mag_ang_proj, create_mask, emb2rect, logits2prob, kron_X_I, kron_I_X
import time

def create_loss_fn(loss_type, **kwargs):
    _e      = kwargs['e'] if 'e' in kwargs else 0
    _P      = kwargs['P'] if 'P' in kwargs else 0
    _ZL     = kwargs['ZL'] if 'ZL' in kwargs else 0
    _ZR     = kwargs['ZR'] if 'ZR' in kwargs else 0
    _Z      = kwargs['Z'] if 'Z' in kwargs else 0
    _A      = kwargs['A'] if 'A' in kwargs else None
    _top_k  = kwargs['top_k'] if 'top_k' in kwargs else 100000000
    _is_rect= kwargs['is_rect'] if 'is_rect' in kwargs else False
    _Vmin   = kwargs['Vmin'] if 'Vmin' in kwargs else 0.7
    _Vmax   = kwargs['Vmax'] if 'Vmax' in kwargs else 1.3
    _thmin  = kwargs['thmin'] if 'thmin' in kwargs else -torch.pi / 2
    _thmax  = kwargs['thmax'] if 'thmax' in kwargs else torch.pi / 2
    _M_type = kwargs['M_type'] if 'M_type' in kwargs else 'STE'
    _M      = kwargs['M'] if 'M' in kwargs else 0
    _sym    = kwargs['sym'] if 'sym' in kwargs else 0
    _diff   = kwargs['diff'] if 'diff' in kwargs else False
    
    if _A is not None: # PSSE
        MA = torch.sparse.mm(_M, _A)
        Ps_AT = torch.sparse.mm(_sym, MA.T)
    else:
        Ps_AT = torch.sparse.mm(_sym, _M.T)
    
    # _top_k = _k - int(torch.sum(_M == 1).item())

    def MC_loss_fn(
        XL      : torch.Tensor, 
        XR      : torch.Tensor, 
        e       : torch.Tensor  = _e, 
        ZL      : torch.Tensor  = _ZL, 
        ZR      : torch.Tensor  = _ZR, 
        P       : torch.Tensor  = _P,
        top_k   : int           = _top_k,   # Not optimized
        M_type  : str           = _M_type,  # Not optimized
        M       : torch.Tensor  = _M,       # default mask (Not optimized)
        tau     : float         = 1,
    ):
        # P_train = P.requires_grad
        # prob_mat = logits2prob(P, top_k, M_type, M, tau)
        # mask = create_mask(prob_mat, top_k, P_train, M_type, M)
        return torch.square(P * (XL @ XR.T - ZL @ ZR.T + e)).sum() / 4
        # return torch.square(((mask - prob_mat).detach() + prob_mat) * (XL @ XR.T - ZL @ ZR.T + e)).sum() / 4
    
    def MC_loss_fn_PSD(
        X       : torch.Tensor, 
        e       : torch.Tensor  = _e, 
        Z       : torch.Tensor  = _Z,
        P       : torch.Tensor  = _P,
        top_k   : int           = _top_k,   # Not optimized
        M_type  : str           = _M_type,  # Not optimized
        M       : torch.Tensor  = _M,       # default mask (COO format)
        Ps      : torch.Tensor  = _sym,     # phi_n @ phi_n.T (COO format)
        tau     : float         = 1,
        diff    : bool          = _diff,
    ):
        n, r = X.size()
        # X = X.contiguous()
        # Z = Z.contiguous()
        # print(X)
        # M_dense = M.to_dense()
        # def testing(a, b, c, d):
        #     rr = (a @ a.T - c @ c.T + b).reshape((n*n, 1))
        #     masked = M_dense @ rr
        #     return masked.square().sum() / 4
        # print("Loss by PyTorch:", testing(X, e, Z, P))
        # hessian_X = jacrev(jacrev(testing, argnums=0), argnums=0)(X, e, Z, P).reshape(n,r,-1).reshape(n*r,-1)
        # grad_X = jacrev(testing, argnums=0)(X, e, Z, P)
        # print("grad by Pytorch: ", grad_X)
        # print("hessian by Pytorch: ", hessian_X)
        P_train = P.requires_grad
        if P_train:
            prob_mat = logits2prob(P, M_type, tau)          # Sparse COO
            mask = create_mask(prob_mat, top_k, M_type, _M)  # Sparse COO
            mask = (mask - prob_mat).detach() + prob_mat
            M_idx, M_val = mask.indices(), mask.values()
            print(M_idx.device, M_val.device)
            MT_idx, MT_val = transpose(M_idx, M_val, n*n, n*n)
            Ps_AT_idx, Ps_AT_val = spspmm(Ps.indices(), Ps.values(), MT_idx, MT_val, n*n, n*n, n*n)
        else:
            M_idx, M_val = M.indices(), M.values()
            Ps_AT_idx, Ps_AT_val = Ps_AT.indices(), Ps_AT.values()
        time1 = time.time()
        sq = X @ X.T - Z @ Z.T
        res = sq + e
        time2 = time.time()
        
        # Ar = torch.sparse.mm(M, res.reshape((n*n, 1)))  # Sparse @ Dense = Dense
        Ar = spmm(M_idx, M_val, n*n, n*n, res.reshape((n*n, 1)))
        # Ps_AT = torch.sparse.mm(Ps, M.T)                # Sparse
        # JxT_AT = 2 * torch.sparse.mm(XT_I, Ps_AT) 
        # JxT_AT = torch.sparse_coo_tensor(JxT_AT_idx, JxT_AT_val * 2, (n*r, n*n)).coalesce()
        # print(time2 - time1, time3 - time2, time4 - time3, time5 - time4, time6 - time5)    
        # JxT = torch.sparse.mm(XT_I, Ps)
        ################## TODO ##################
        ### 1. Coalesce
        ### 2. Optimize Kronecker
        ### 3. Switch spspmm to torch_sparse
        ##########################################
        loss = Ar.square().sum() / 4
        sq_loss = torch.linalg.matrix_norm(sq, ord='fro')

        if not diff:
            return loss, sq_loss
        else:
            XT_I_idx, XT_I_val = kron_X_I(X.T, n)  
            JxT_AT_idx, JxT_AT_val = spspmm(XT_I_idx, XT_I_val, Ps_AT_idx, Ps_AT_val, n*r, n*n, n*n)
            JxT_AT_val = JxT_AT_val * 2 
            # print(JxT_AT.to_dense(), Ar.to_dense())
            time1 = time.time()
            gradient = 0.5 * spmm(JxT_AT_idx, JxT_AT_val, n*r, n*n, Ar)# Dense
            # print(gradient)
            # sym_Ar = torch.sparse.mm(Ps_AT, Ar).reshape(n, n)# Dense
            time2 = time.time()
            sym_Ar = spmm(Ps_AT_idx, Ps_AT_val, n*n, n*n, Ar).reshape(n, n)
            # print(JxT_AT)
            time3 = time.time()
            curv_idx, curv_val = kron_I_X(sym_Ar, r)                  # Sparse
            curv = torch.sparse_coo_tensor(curv_idx, curv_val, (n*r, n*r))
            time4 = time.time()
            A_Jx_idx, A_Jx_val = transpose(JxT_AT_idx, JxT_AT_val, n*r, n*n)
            GN_idx, GN_val = spspmm(JxT_AT_idx, JxT_AT_val, A_Jx_idx, A_Jx_val, n*r, n*n, n*r)
            GN = torch.sparse_coo_tensor(GN_idx, GN_val * 0.5, (n*r, n*r))
            # GN = 0.5 * torch.sparse.mm(JxT_AT, JxT_AT.T) # Sparse
            time5 = time.time()
            hess = (GN + curv).coalesce()
            time6 = time.time()
            hess_dense = hess.to_dense()
            time7 = time.time()
            # print("second:")
            # print(time2 - time1, time3 - time2, time4 - time3, time5 - time4, time6 - time5, time7 - time6)
            
            return loss, sq_loss, gradient, hess_dense
        # return torch.square(((mask - prob_mat).detach() + prob_mat) * (X @ X.T - Z @ Z.T + e)).sum() / 4
    
    def PSSE_loss_fn(
        X_emb   : torch.Tensor, 
        e       : torch.Tensor  = _e, 
        Z_emb   : torch.Tensor  = _Z,
        P       : torch.Tensor  = _P,
        A       : torch.Tensor  = _A,       # measurement matrix (COO format)
        Ps      : torch.Tensor  = _sym,     # phi_n @ phi_n.T (COO format)
        top_k   : int           = _top_k,   # Not optimized
        M_type  : str           = _M_type,  # Not optimized
        M       : torch.Tensor  = _M,       # default mask (COO format)
        Vmin    : float         = _Vmin,    # Not optimized
        Vmax    : float         = _Vmax,    # Not optimized
        thmin   : float         = _thmin,   # Not optimized
        thmax   : float         = _thmax,   # Not optimized 
        tau     : float         = 1,
        diff    : bool          = _diff,
    ):
        n, r = X.size()
        d = A.size(0)
        X = emb2rect(X_emb, Vmin, Vmax, thmin, thmax)
        Z = emb2rect(Z_emb, Vmin, Vmax, thmin, thmax)
        
        P_train = P.requires_grad
        if P_train:
            prob_mat = logits2prob(P, M_type, tau)          # Sparse COO
            mask = create_mask(prob_mat, top_k, M_type, _M)  # Sparse COO
            mask = (mask - prob_mat).detach() + prob_mat
            M_idx, M_val = mask.indices(), mask.values()
            A_idx, A_val = spspmm(M_idx, M_val, A.indices(), A.values(), d, d, n*n)
            AT_idx, AT_val = transpose(A_idx, A_val, d, n*n)
            Ps_AT_idx, Ps_AT_val = spspmm(Ps.indices(), Ps.values(), AT_idx, AT_val, n*n, n*n, d)
        else:
            A_idx, A_val = MA.indices(), MA.values()
            Ps_AT_idx, Ps_AT_val = Ps_AT.indices(), Ps_AT.values()
        
        sq = X @ X.T - Z @ Z.T
        res = sq + e
        Ar = spmm(A_idx, A_val, d, n*n, res.reshape((n*n, 1)))
        loss = Ar.square().sum() / 4
        sq_loss = torch.linalg.matrix_norm(sq, ord='fro')

        if not diff:
            return loss, sq_loss
        else:
            XT_I_idx, XT_I_val = kron_X_I(X.T, n)  
            JxT_AT_idx, JxT_AT_val = spspmm(XT_I_idx, XT_I_val, Ps_AT_idx, Ps_AT_val, n*r, n*n, d)
            JxT_AT_val = JxT_AT_val * 2 
            gradient = 0.5 * spmm(JxT_AT_idx, JxT_AT_val, n*r, d, Ar)# Dense
            sym_Ar = spmm(Ps_AT_idx, Ps_AT_val, n*n, d, Ar).reshape(n, n)
            curv_idx, curv_val = kron_I_X(sym_Ar, r)                  # Sparse
            curv = torch.sparse_coo_tensor(curv_idx, curv_val, (n*r, n*r))
            
            A_Jx_idx, A_Jx_val = transpose(JxT_AT_idx, JxT_AT_val, n*r, d)
            GN_idx, GN_val = spspmm(JxT_AT_idx, JxT_AT_val, A_Jx_idx, A_Jx_val, n*r, d, n*r)
            GN = torch.sparse_coo_tensor(GN_idx, GN_val * 0.5, (n*r, n*r))
            hess = (GN + curv).coalesce()
            hess_dense = hess.to_dense()
            return loss, sq_loss, gradient, hess_dense

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
            # mag_ang_proj(X, 1.2, 0.8, 90, -90)
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
    elif optim == 'L-BFGS':
        optimizer = torch.optim.LBFGS(list(parameters.values()), lr=lr, line_search_fn='strong_wolfe')
    else:
        raise NotImplementedError
    # SGD
    losses = []
    with tqdm(total=iters, leave=False) as pbar:
        for i in range(iters):
            if optim == 'L-BFGS':
                def closure():
                    optimizer.zero_grad()
                    loss, _ = criterion(**parameters)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
                losses.append((loss.item()))
            else:
                optimizer.zero_grad()
                loss, _ = criterion(**parameters)
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
            if optim != 'L-BFGS': # L-BFGS use line search to decide step size
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
            # Project to feasible set
            if project_fn is not None:
                project_fn(**parameters)
            pbar.update(1)
            pbar.set_description(f"lr={new_lr:.6f}, loss={losses[-1]:.9f}")
    
    if ret_eig:
        with torch.no_grad():
            loss, sq_loss, grad, hess = criterion(diff=True, **parameters)
            grad_norm = torch.linalg.norm(grad).item()
            min_eig = torch.linalg.eigvalsh(hess)[0].item()
            return loss, sq_loss, grad_norm, min_eig
    else:
        return losses