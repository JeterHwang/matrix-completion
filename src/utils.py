import torch
import random
import numpy as np
from numpy.linalg import qr
from torch.linalg import svd
import pandapower as pp
import pandapower.networks as pn
from scipy.sparse import csr_matrix
import networkx as nx

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

def augment_Z(X: np.ndarray):
    n, m = X.shape
    aug_X = np.zeros((n+m, n+m))
    aug_X[n:, :n] = X.T
    aug_X[:n, n:] = X
    return aug_X

def create_Z(n, m, r, mu, PSD=False):
    if PSD:
        assert n == m
        U, mu_U = incoherent_factor(n, r, mu)
        V, mu_V = U, mu_U
    else:
        U, mu_U = incoherent_factor(n, r, mu)
        V, mu_V = incoherent_factor(m, r, mu)
    s = np.linspace(1.0, 0.2, r) 
    Sigma = np.diag(s)
    L = U @ np.sqrt(Sigma)
    R = V @ np.sqrt(Sigma)
    return (L,R), max(mu_U, mu_V)

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

def augment_mask(mask: np.ndarray):
    n, m = mask.shape
    indices = []
    for i in range(n):
        for j in range(m):
            if mask[i][j] == 1:
                indices.append((i, j))
    A = np.zeros((len(indices), m+n, m+n))
    for i, (x, y) in enumerate(indices):
        A[i][x][y+n] = 1
        A[i][y+n][x] = 1
    return A

def sample_mask(n, m, p):
    mask = np.random.binomial(1, p, (n, m))
    return mask

def spectral_initialization(p, r, Z, mask, PSD=False):
    if PSD:
        Z_0 = ((Z * mask) + (Z * mask).T) / (2 * p)
    else:
        Z_0 = (Z * mask) / p
    U, S, Vh = svd(Z_0, full_matrices=False)
    if PSD:
        X_0 = (U[:,:r] @ torch.diag(torch.sqrt(S[:r]))).requires_grad_(True)
        parameters = [X_0]
    else:
        L_0 = (U[:,:r] @ torch.diag(torch.sqrt(S[:r]))).requires_grad_(True)
        R_0 = (Vh[:r,:].T @ torch.diag(torch.sqrt(S[:r]))).requires_grad_(True)
        parameters = [L_0, R_0]
    return parameters

def extract_measurement():
    Matrices = []
    # Load IEEE 14-bus system
    net = pn.case14()
    n = len(net.bus)

    # Build a graph
    G = nx.Graph()
    for _, line in net.line.iterrows():
        # print(line["from_bus"], line["to_bus"])
        G.add_edge(line["from_bus"], line["to_bus"])
    for _, trafo in net.trafo.iterrows():
        # print(trafo["hv_bus"], trafo["lv_bus"])
        G.add_edge(trafo["hv_bus"], trafo["lv_bus"])
    # for i in range(n):
    #     G.add_edge(i, i)

    # Solve power flow
    pp.runpp(net)
    Ybus = net._ppc["internal"]["Ybus"]
    Y = csr_matrix(Ybus).toarray()
    
    vm = np.array(net.res_bus['vm_pu'])       # voltage magnitude (per unit)
    va = np.array(net.res_bus['va_degree'])   # voltage angle (in degrees)
    z = (vm * np.exp(1j * va)).reshape(n,1)
    # print(z)

    identity = np.identity(n)
    # voltage magnitude
    for i in range(n):
        e_i = identity[:,i].reshape(n, 1)
        Matrices.append(((e_i @ e_i.T), np.zeros((n,n)), True))
    
    # branch power 
    # print(net.res_line['p_from_mw'])
    # print(net.res_line['p_to_mw'])
    for i in range(n):
        e_i = identity[:,i].reshape(n, 1)
        neighbors = list(G.neighbors(i))
        for j in neighbors:
            if i == j:
                continue
            e_j = identity[:,j].reshape(n, 1)
            # print(Y[i][j])
            S_i = np.conjugate(Y[i][j]) * ((e_i - e_j) @ e_i.T)
            P_i = 0.5 * (S_i + S_i.conj().T)
            Q_i = 0.5 * (S_i - S_i.conj().T)
            # print(i, j, z.conj().T @ P_i @ z, z.conj().T @ Q_i @ z)
            Matrices.append((P_i.real, P_i.imag, True))
            Matrices.append((Q_i.real, Q_i.imag, False))
    # nodal power injection
    # P_inj = net.res_bus['p_mw']
    # print(P_inj)
    for i in range(n):
        e_i = identity[:,i].reshape(n, 1)
        S_i = np.conjugate(Y[i][i]) * (e_i @ e_i.T)
        neighbors = list(G.neighbors(i))
        for j in neighbors:
            e_j = identity[:,j].reshape(n, 1)
            S_i = S_i + np.conjugate(Y[i][j]) * ((e_i - e_j) @ e_i.T)
        P_i = 0.5 * (S_i + S_i.conj().T)
        Q_i = 0.5 * (S_i - S_i.conj().T)
        # print(z.conj().T @ P_i @ z)
        Matrices.append((P_i.real, P_i.imag, True))
        Matrices.append((Q_i.real, Q_i.imag, False))
    return Matrices, z

def create_P(mats):
    P = []
    for p_r, p_i, real in mats:
        if real:
            P.append(torch.tensor(np.concatenate([np.concatenate([p_r, -p_i], axis=1), np.concatenate([p_i, p_r], axis=1)], axis=0), dtype=torch.float64))
        else:
            P.append(torch.tensor(np.concatenate([np.concatenate([p_i, p_r], axis=1), np.concatenate([-p_r, p_i], axis=1)], axis=0), dtype=torch.float64))
    random.shuffle(P) 
    return torch.stack(P) 

def PSSE_initialization(z, t):
    gaussian_vector = np.random.multivariate_normal(np.zeros(len(z)), np.identity(len(z)), size=1).reshape(z.shape)
    x_0 = z + t * gaussian_vector / np.linalg.norm(gaussian_vector, ord=np.inf)
    return x_0

def incoherence_proj(X, mu):
    n, r = X.size()
    # incoherence projection of X
    row_norm_X = torch.linalg.vector_norm(X, ord=2, dim=1)
    frob_norm_X = torch.linalg.matrix_norm(X, ord='fro')
    row_norm_X.div_(frob_norm_X * np.sqrt(mu / n))
    scale_X = torch.maximum(row_norm_X, torch.ones_like(row_norm_X))
    X.div_(scale_X.unsqueeze(-1))

def calc_loss_coeff(alpha_0, beta_0, alpha_t, beta_t, i, iters, scale_type='linear'):
    if scale_type == 'static':
        return alpha_t, beta_t
    elif scale_type == 'linear':
        lam = min(1.0, i / (0.5 * iters))
        alpha = alpha_0 + (alpha_t - alpha_0) * lam
        beta = beta_0 + (beta_t - beta_0) * lam
        return alpha, beta
    else:
        raise NotImplementedError