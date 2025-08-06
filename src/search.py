import torch
import numpy as np
from typing import List
from tqdm import tqdm
from torch.autograd import grad
from torch.autograd.functional import hessian
from src.utils import incoherence_proj, calc_loss_coeff

def create_search_proj_fn(proj_type, **kwargs):
    @torch.no_grad()
    def MC_project(X, E=None, Z=None):
        # incoherence projection of X
        incoherence_proj(X, kwargs['mu'])
        # incoherence projection of Z
        if Z is not None:
            incoherence_proj(Z, kwargs['mu'])
        # norm projection
        if E is not None:
            norm_E = torch.linalg.matrix_norm(E * kwargs['mask'], ord='fro')
            if norm_E > kwargs['max_norm']:
                E.data = E.data / norm_E * kwargs['max_norm']
    @torch.no_grad()
    def PSSE_project(x, e=None, z=None):
        x[len(x) // 2] = 0
        if z is not None:
            z[len(z) // 2] = 0
        if e is not None:
            norm_E = torch.linalg.vector_norm(e, ord=2)
            if norm_E > kwargs['max_norm']:
                e.data = e.data / norm_E * kwargs['max_norm']
    if proj_type == 'MC':
        return MC_project
    elif proj_type == 'PSSE':
        return PSSE_project
    else:
        raise NotImplementedError

def create_search_loss_fn(f, loss_type='MC_sum', **kwargs): # Use MC-PSD-2
    def MC_loss_sum(X, e=kwargs['e'], Z=kwargs['Z'], alpha=1, beta=1):
        assert X.size() == Z.size()
        w, h = X.size()
        f_Z = f(Z, e, Z)
        f_X = f(X, e, Z)
        diff_loss = f_Z - f_X 
        dist_loss = -torch.linalg.norm(X - Z)
        first_order_loss_X = torch.linalg.matrix_norm(grad(f(X, e, Z), X, create_graph=True)[0])
        second_order_loss_X = -torch.linalg.eigvalsh(hessian(f, (X, e, Z), create_graph=True)[0][0].reshape(w,h,-1).reshape(w*h,-1))[0]
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
    def MC_loss_max(X, e=kwargs['e'], Z=kwargs['Z'], alpha=1, beta=1): # Use MC-PSD-2
        assert X.size() == Z.size()
        w, h = X.size()
        f_Z = f(Z, e, Z)
        f_X = f(X, e, Z)
        diff_loss = f_Z - f_X 
        dist_loss = -torch.linalg.norm(X - Z)
        first_order_loss_X = torch.linalg.matrix_norm(grad(f(X, e, Z), X, create_graph=True)[0])
        second_order_loss_X = -torch.linalg.eigvalsh(hessian(f, (X, e, Z), create_graph=True)[0][0].reshape(w,h,-1).reshape(w*h,-1))[0]
        transform_loss = -torch.linalg.matrix_norm(X @ X.T - Z @ Z.T, ord='fro')
        max_loss = torch.maximum(first_order_loss_X, second_order_loss_X)
        return (
            alpha * transform_loss + beta * max_loss, 
            diff_loss, 
            first_order_loss_X, 
            second_order_loss_X, 
            f_X, 
            f_Z, 
            dist_loss,
            transform_loss,
            max_loss
        )
    def PSSE_loss_sum(x, e=kwargs['e'], z=kwargs['Z'], alpha=1, beta=1): # Use PSSE-1
        w, h = x.size()
        f_X = f(x, e, z)
        f_Z = f(z, e, z)
        diff_loss = f_Z - f_X
        transform_loss = -torch.linalg.matrix_norm(x @ x.T - z @ z.T, ord='fro')
        dist_loss = -torch.linalg.norm(x - z)
        first_order_loss = torch.linalg.norm(grad(f(x, e, z), x, create_graph=True)[0])
        second_order_loss = -torch.linalg.eigvalsh(hessian(f, (x, e, z), create_graph=True)[0][0].reshape(w,h,-1).reshape(w*h,-1))[0]
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
    def PSSE_loss_max(x, e=kwargs['e'], z=kwargs['Z'], alpha=1, beta=1): # Use PSSE-1
        w, h = x.size()
        f_X = f(x, e, z)
        f_Z = f(z, e, z)
        diff_loss = f_Z - f_X
        transform_loss = -torch.linalg.matrix_norm(x @ x.T - z @ z.T, ord='fro')
        dist_loss = -torch.linalg.norm(x - z)
        first_order_loss = torch.linalg.norm(grad(f(x, e, z), x, create_graph=True)[0])
        second_order_loss = -torch.linalg.eigvalsh(hessian(f, (x, e, z), create_graph=True)[0][0].reshape(w,h,-1).reshape(w*h,-1))[0]
        max_loss = torch.maximum(first_order_loss, second_order_loss)
        return (
            alpha * transform_loss + beta * max_loss, 
            diff_loss, 
            first_order_loss, 
            second_order_loss,
            f_X, 
            f_Z, 
            dist_loss,
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
    parameters: List[torch.Tensor], 
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
    if optim == 'Adam':
        optimizer_W = torch.optim.Adam(parameters, lr=lr)
        # optimizer_coeff = torch.optim.Adam([coeff], lr=0.025)
    elif optim == 'AdamW':
        optimizer_W = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
        # optimizer_coeff = torch.optim.Adam([coeff], lr=0.025)
    elif optim == 'SGD':
        optimizer_W = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=1e-4)
        # optimizer_coeff = torch.optim.Adam([coeff], lr=0.025)
    else:
        raise NotImplementedError
    
    losses = []
    # Pre-test
    loss_0, diff_0, grad_X_0, hess_X_0, f_X_0, f_Z_0, dist_0, trans_0, max_loss_0 = criterion(*parameters)
    losses.append((loss_0.item(), diff_0.item(), grad_X_0.item(), hess_X_0.item(), f_X_0.item(), f_Z_0.item(), trans_0.item()))
    # T = torch.sum(coeff.detach())
    alpha, beta = 1, 1
    # SGD
    with tqdm(total=iters, leave=False) as pbar:
        for i in range(iters):
            # alpha, beta = calc_loss_coeff(alpha_0, beta_0, coeff[0], coeff[1], i, iters, loss_scale)
            # alpha = coeff[0] 
            # beta = abs(trans / max(grad_X, hess_X)) * beta
            # alpha = coeff.detach()[0]
            # beta = coeff.detach()[1]
            loss, diff, grad_X, hess_X, f_X, f_Z, dist, trans, max_loss = criterion(*parameters, alpha=alpha, beta=beta)
            gradients_1 = grad(outputs=trans, inputs=parameters, retain_graph=True, allow_unused=True)
            gradients_2 = grad(outputs=max_loss, inputs=parameters, retain_graph=True, allow_unused=True)
            optimizer_W.zero_grad()
            loss.backward()
            # if clip_grad:
            #     torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, norm_type=2)
            
            grad_norm_1 = torch.sqrt(sum([0 if GG is None else torch.linalg.norm(GG).square() for GG in gradients_1]))
            # g1 = coeff[0] * grad_norm_1.detach()
            grad_norm_2 = torch.sqrt(sum([0 if GG is None else torch.linalg.norm(GG).square() for GG in gradients_2]))
            # g2 = coeff[1] * grad_norm_2.detach()  
            # g_bar = 0.5 * (g1 + g2)
            # print(grad_norm_1.item(), grad_norm_2.item())
            # g1 = grad_norm_2.item() / (grad_norm_1.item() + grad_norm_2.item())
            # g2 = grad_norm_1.item() / (grad_norm_1.item() + grad_norm_2.item())
            r1 = (trans.item() - trans_0.item()) / (trans_bound[0] - trans_0.item())
            r2 = (max_loss_0.item() - max_loss.item()) / max_loss_0.item()
            alpha = grad_norm_2.item() * (np.exp(r1 * T) / (np.exp(r1 * T) + np.exp(r2 * T)))
            beta = grad_norm_1.item() * (np.exp(r2 * T) / (np.exp(r1 * T) + np.exp(r2 * T)))
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
                param_group['lr'] = new_lr
            # Project to feasible set
            project_fn(*parameters)
            # Evaluating
            losses.append((loss.item(), diff.item(), grad_X.item(), hess_X.item(), f_X.item(), f_Z.item(), trans.item()))
            # Early Stop
            if trans > trans_bound[0]:
                print("transform loss too small !!")
                break
            elif trans < trans_bound[1]:
                print("transform loss explodes !!")
                break
            pbar.set_description(f"Iter: {i}, lr={new_lr:.6f}, diff={diff.item():.4f}, dist={dist.item():.4f}, trans={trans.item():.4f}, max={max(grad_X.item(), hess_X.item()):.4f}, coeff=({alpha:.4f},{beta:.4f})")
            pbar.update(1)
    return losses