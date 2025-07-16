import torch
import numpy as np
from torch.linalg import qr, svd

def same_seed(seed):
    # set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def incoherent_factor(n, r, mu):
    # Create a random orthonormal matrix
    Q, _ = qr(np.random.standard_normal((n, r)))
    row_norm2 = np.sum(Q**2, axis=1)

    # choose the row we will 'spike'
    i_star = np.random.randint(n)

    # 3. target norm^2 for the spike and for the rest
    target_spike = mu * r / n
    target_rest  = (r - target_spike) / (n - 1)

    # 4. rescale rows (then re-orthonormalise columns)
    scale = np.sqrt(
        np.where(
            np.arange(n)==i_star,
            target_spike/row_norm2,
            target_rest/row_norm2
        )
    )
    U = (scale[:,None] * Q)

    # 5. final re-orthonormalisation (tiny tweak)
    U, _ = qr(U)         # keeps rows norms very close to target
    
    # Calculate the acutal coherence
    actual_mu = np.max(np.sum(U**2, axis=1) / (r / n))

    return U, actual_mu

def create_Z(n, m, r, mu, PSD=False, convex=True):
    if PSD:
        assert n == m
        U, mu_U = incoherent_factor(n, r, mu)
        V, mu_V = U, mu_U
    else:
        U, mu_U = incoherent_factor(n, r, mu)
        V, mu_V = incoherent_factor(m, r, mu)
    s = np.linspace(1.0, 0.2, r) 
    Sigma = np.diag(s)
    X_star = U @ Sigma @ V.T
    if convex:
        Z = np.zeros((n+m, n+m))
        Z[n:, :n] = X_star.T
        Z[:n, n:] = X_star
    else:
        Z = X_star
    return Z, max(mu_U, mu_V)

# def create_Z(n, m, r, mu, PSD=False): # Nonconvex
#     if PSD:
#         assert n == m
#         U, mu_U = incoherent_factor(n, r, mu)
#         V, mu_V = U, mu_U
#     else:
#         U, mu_U = incoherent_factor(n, r, mu)
#         V, mu_V = incoherent_factor(m, r, mu)
#     s = torch.linspace(1.0, 0.2, r) 
#     Sigma = torch.diag(s)
#     Z = U @ Sigma @ V.T
#     return Z, max(mu_U, mu_V)

def sample_mask(n, m, p, convex=True):
    mask = np.random.binomial(1, p, (n, m))
    if not convex: # Non-convex
        return mask
    indices = []
    for i in range(n):
        for j in range(m):
            if mask[i][j] == 1:
                indices.append((i, j))
    A = np.zeros((len(indices), m+n, m+n))
    for i, (x, y) in enumerate(indices):
        A[i][x][y+n] = 1
        A[i][y+n][x] = 1
    return mask, A

def spectral_initialization(p, r, Z, mask):
    Z_0 = Z * mask * (1 / p)
    U, S, Vh = svd(Z_0, full_matrices=False)
    L_0 = U[:,:r] @ torch.diag(torch.sqrt(S[:r]))
    R_0 = Vh[:r,:].T @ torch.diag(torch.sqrt(S[:r]))
    return L_0.requires_grad_(True), R_0.requires_grad_(True)