import torch
import time
from scipy.sparse.linalg import eigsh, LinearOperator
from torch.func import jacrev
from torch.autograd.functional import hessian
from torch.linalg import eigh

def build_hvp_operator(f, X, e, Z, P):
    """
    Build an HVP callable for the current x.
    f: () -> scalar loss computed from current x (x.requires_grad must be True)
    x: tensor
    Returns: Hv(v, retain=True|False)
    """
    # one-time gradient with graph so we can differentiate it w.r.t. x
    (g,) = torch.autograd.grad(f(X, e, Z, P), X, create_graph=True, retain_graph=True)
    g = g.view(-1)

    def Hv(v, create_graph=False):
        # one backward through g's graph; no need to recreate g or re-run f
        # print("dfdfdfdfd")
        (Hv_,) = torch.autograd.grad(g, X, grad_outputs=v, create_graph=create_graph, retain_graph=create_graph)
        # print("qwqwqwqw")
        return Hv_.view(-1)
    return Hv


def power_method(f, X, e, Z, P, tol=1e-4, iters=50, margin=1.05):
    device = X.device
    hvp = build_hvp_operator(f, X, e, Z, P)
    # Estimate alpha
    v = torch.randn_like(X).view(-1).to(device)
    v = v / (torch.linalg.norm(v) + 1e-18)
    for i in range(8):
        w = hvp(v, True)
        v = w / (torch.linalg.norm(w) + 1e-18)
    lamb_max = torch.dot(v, hvp(v, True))
    alpha = margin * lamb_max
    # Compute lambda min
    v = torch.randn_like(X).view(-1).to(device)
    v = v / (torch.linalg.norm(v) + 1e-18)
    for i in range(iters):
        Hv = hvp(v, True)
        shift_Hv = alpha * v - Hv
        norm_Hv = torch.linalg.norm(shift_Hv)
        if norm_Hv.item() == 0:
            break
        v = shift_Hv / (norm_Hv + 1e-18)
    lamb_min = torch.dot(v.detach(), hvp(v.detach(), True))
    return lamb_min

def lanczos(f, X, e, Z, P, tol=1e-4):
    device = X.device
    hvp = build_hvp_operator(f, X, e, Z, P)
    def _scipy_apply(x):
        x = torch.from_numpy(x).to(device)
        # print("Start")
        out = hvp(x, True)
        # print("end")
        return out.detach().cpu().numpy()
    n = X.numel()
    scipy_op = LinearOperator((n, n), _scipy_apply)
    eigenvals, eigenvecs = eigsh(
        A       = scipy_op,
        k       = 1,
        which   = 'SA',
        # maxiter = n,
        tol     = tol,
        ncv     = 40,
        return_eigenvectors = True,
    )
    v = torch.from_numpy(eigenvecs[:, 0]).to(device)
    v = v / (torch.linalg.norm(v) + 1e-18)
    lamb_min = torch.dot(v.detach(), hvp(v.detach(), create_graph=True))
    return lamb_min

def lobpcg():
    pass

def eigen_decomp(f, X, e, Z, P):
    w, h = X.size()
    time1 = time.time()
    # hessian_X = hessian(f, (X, e, Z, P), create_graph=False)[0][0].reshape(w,h,-1).reshape(w*h,-1)
    hessian_X = jacrev(jacrev(f, argnums=0), argnums=0)(X, e, Z, P).reshape(w,h,-1).reshape(w*h,-1)
    print(hessian_X)
    time2 = time.time()
    hvp = build_hvp_operator(f, X, e, Z, P)
    time3 = time.time()
    eigenvalues, eigenvectors = eigh((hessian_X + hessian_X.T) / 2)
    time4 = time.time()
    v = eigenvectors[0] / (torch.linalg.norm(eigenvectors[0]) + 1e-18)
    lamb_min = torch.dot(v.detach(), hvp(v.detach(), create_graph=True))
    time5 = time.time()
    print(lamb_min, eigenvalues[0])
    # print(f"Compute hessian : {time2 - time1} (s)")
    # print(f"build hvp operator: {time3 - time2} (s)")
    # print(f"Eigen Decomposition: {time4 - time3} (s)")
    # print(f"Compute lambda min: {time5 - time4} (s)")
    return eigenvalues[0]
