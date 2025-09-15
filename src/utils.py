import torch
import random
import numpy as np
from numpy.linalg import qr
from torch.linalg import svd
import scipy.io as sio
import pandapower as pp
import pandapower.networks as pn
from scipy.sparse import csr_matrix
from scipy import sparse
import networkx as nx
from entmax import entmax15
import logging
import time
from torch_sparse import transpose, spspmm

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

def load_mat_data(path):
    # --- 1.  Load the .mat file --------------------------------------------------
    mat = sio.loadmat(
        path,
        squeeze_me=True,            # drop singleton dimensions
        simplify_cells=True         # unwrap cell arrays / structs automatically
    )                               # spmatrix=True by default, so sparse MATLAB
                                    # variables come back as SciPy sparse objects:contentReference[oaicite:0]{index=0}

    # --- 2.  Grab the variables you need -----------------------------------------
    A   = mat["A"]      # sparse matrix
    bsol   = np.asarray(mat["b"]).ravel()       # 1-D NumPy vector
    xsol = np.asarray(mat["xsol"]).ravel()   # 1-D NumPy vector

    # --- 3.  (Optional) tidy-up / conversions ------------------------------------
    # Make sure A is the actual sparse matrix (not wrapped in a 1×1 object array)
    if isinstance(A, np.ndarray) and A.dtype == object:
        A = A.item()                # unwrap cell / struct entry:contentReference[oaicite:1]{index=1}
    
    values = A.data
    indices = np.vstack((A.row, A.col))
    matrix_shape = A.shape

    torch_values = torch.FloatTensor(values)
    torch_indices = torch.LongTensor(indices)
    torch_shape = torch.Size(matrix_shape)

    return torch.sparse_coo_tensor(torch_indices, torch_values, torch_shape).T.coalesce()

def create_P(mats):
    P = []
    for p_r, p_i, real in mats:
        if real:
            P.append(torch.tensor(np.concatenate([np.concatenate([p_r, -p_i[:,1:]], axis=1), np.concatenate([p_i[1:], p_r[1:,1:]], axis=1)], axis=0)))
        else:
            P.append(torch.tensor(np.concatenate([np.concatenate([p_i, p_r[:,1:]], axis=1), np.concatenate([-p_r[1:], p_i[1:,1:]], axis=1)], axis=0)))
    random.shuffle(P) 
    return torch.stack(P) 

def PSSE_initialization(z, t):
    gaussian_vector = np.random.multivariate_normal(np.zeros(len(z)), np.identity(len(z)), size=1).reshape(z.shape)
    x_0 = z + t * gaussian_vector / np.linalg.norm(gaussian_vector, ord=np.inf)
    return x_0

@torch.no_grad()
def incoherence_proj(X, mu):
    n, r = X.size()
    # incoherence projection of X
    row_norm_X = torch.linalg.vector_norm(X, ord=2, dim=1)
    frob_norm_X = torch.linalg.matrix_norm(X, ord='fro')
    row_norm_X.div_(frob_norm_X * np.sqrt(mu / n))
    scale_X = torch.maximum(row_norm_X, torch.ones_like(row_norm_X))
    X.div_(scale_X.unsqueeze(-1))

@torch.no_grad()
def mag_ang_proj(X, mag_high, mag_low, ang_high, ang_low):
    assert len(X) % 2 != 0
    n = (len(X) + 1) // 2
    X_comp = torch.complex(X[:n], torch.cat([torch.zeros((1,1), dtype=X.dtype), X[n:]]))
    # Magnitude projection
    mag = torch.abs(X_comp)
    scale_upper = torch.maximum(mag / mag_high, torch.ones_like(mag))
    scale_lower = torch.minimum(mag / mag_low, torch.ones_like(mag))
    new_mag = mag / (scale_upper * scale_lower)
    # Angle projection
    ang = torch.angle(X_comp)
    diff_upper = torch.maximum(ang - ang_high / 180 * torch.pi, torch.zeros_like(ang))
    diff_lower = torch.minimum(ang - ang_low / 180 * torch.pi, torch.zeros_like(ang))
    new_ang = ang - 2 * (diff_upper + diff_lower)
    new_X = torch.polar(new_mag, new_ang)
    X[:n] = new_X.real
    X[n:] = new_X.imag[1:]

def emb2rect(X: torch.Tensor, Vmin, Vmax, thmin, thmax):
    assert len(X) % 2 != 0
    n = (len(X) + 1) // 2
    mag = Vmin + (Vmax - Vmin) * torch.sigmoid(X[:n])
    # ang = 0.5 * (thmin + thmax) + 0.5 * (thmax - thmin) * torch.tanh(X[n:])
    ang = thmin + (thmax - thmin) * torch.sigmoid(X[n:])
    rect = torch.cat([mag[:1], mag[1:] * torch.cos(ang), mag[1:] * torch.sin(ang)], dim=0)
    return rect

def emb2polar(X: torch.Tensor, Vmin, Vmax, thmin, thmax):
    assert len(X) % 2 != 0
    n = (len(X) + 1) // 2
    mag = Vmin + (Vmax - Vmin) * torch.sigmoid(X[:n])
    # ang = 0.5 * (thmin + thmax) + 0.5 * (thmax - thmin) * torch.tanh(X[n:])
    ang = thmin + (thmax - thmin) * torch.sigmoid(X[n:])
    return torch.cat([mag, ang], dim=0)

def polar2emb(X: torch.Tensor, Vmin, Vmax, thmin, thmax):
    assert len(X) % 2 != 0
    n = (len(X) + 1) // 2
    V_emb = -torch.log((Vmax - Vmin) * torch.reciprocal(X[:n] - Vmin) - 1)
    ang_emb = -torch.log((thmax - thmin) * torch.reciprocal(X[n:] - thmin) - 1)
    return torch.cat([V_emb, ang_emb], dim=0)

def rect2emb(X: torch.Tensor, Vmin, Vmax, thmin, thmax):
    assert len(X) % 2 != 0
    n = (len(X) + 1) // 2
    real = X[:n]
    imag = torch.cat([torch.zeros_like(X)[0:1], X[n:]], dim=0)
    polar = torch.complex(real, imag)
    mag = torch.abs(polar)
    ang = torch.angle(polar)
    V_emb = -torch.log((Vmax - Vmin) * torch.reciprocal(mag - Vmin) - 1)
    ang_emb = -torch.log((thmax - thmin) * torch.reciprocal(ang[1:] - thmin) - 1)
    return torch.cat([V_emb, ang_emb], dim=0)

def create_mask(
    prob_mat: torch.Tensor,     # (n,n)
    top_k   : int,
    M_type  : str,
    M       : torch.Tensor,     # (n,n)
):
    if top_k <= 0:
        return M
    flatten = prob_mat.reshape(-1) - M.reshape(-1) * 1e9   # 1D
    # if M_type == 'entmax':
    #     if train:
    #         mask = flatten
    #     else:
    #         _, indices = torch.topk(flatten, k=top_k)
    #         mask = torch.zeros_like(flatten, dtype=prob_mat.dtype)
    #         mask[indices] = 1
    if M_type == 'STE':
        _, indices = torch.topk(flatten, k=top_k)
        mask = torch.zeros_like(flatten, dtype=prob_mat.dtype)
        mask[indices] = 1
    else:
        raise NotImplementedError
    
    return mask.reshape(M.size()).to(prob_mat.device)

def logits2prob(
    logits: torch.Tensor,
    M_type: str,
    tau: float = 0.5
):
    # flatten = logits.view(-1)
    # if M_type == 'entmax':
    #     prob_mat = top_k * entmax15(flatten, dim=0)
    if M_type == 'STE':
        flatten = logits.reshape(-1)
        prob_mat = torch.softmax(flatten / tau, dim=0)
    else:
        raise NotImplementedError
    return prob_mat.reshape(logits.size()).to(logits.device)

############### Sparse Matrix #################

def extend_mask_coo(
    M: torch.Tensor,            # COO 
    A: torch.Tensor = None,
):
    if A is not None: # A matrix for PSSE (TODO)
        masked_A = A.reshape((A.size(0), -1)) * M
        return masked_A.to_sparse_coo()
    else:
        n, m = M.size()
        inds = M.indices()
        vals = M.values()
        flat_inds = inds[0] * m + inds[1]
        return torch.sparse_coo_tensor(torch.stack([flat_inds, flat_inds]), vals, (n*m, n*m)).coalesce()
        
def symbasis(n: int) -> torch.Tensor:
    """
    Return V \in R^{(n^2) x (n(n+1)/2)} whose columns form an orthonormal basis
    for the space of n×n symmetric matrices under the Frobenius inner product.

    Vectorization matches MATLAB's vec(.) (column-major / Fortran order).
    """
    ndof = n * (n + 1) // 2

    # Enumerate lower-triangular indices (i >= j) in column-major order (match MATLAB)
    rows = torch.cat([torch.arange(j, n, dtype=torch.long) for j in range(n)])
    cols = torch.cat([torch.full((n - j,), j, dtype=torch.long) for j in range(n)])

    # Linear indices for vec(.) in column-major order
    ii1 = rows + cols * n          # positions (i, j)
    ii2 = cols + rows * n          # positions (j, i)

    dof = torch.arange(ndof, dtype=torch.long)

    # Values: sqrt(1/2) off-diagonals, and two 0.5 entries on diagonal (which sum to 1)
    w1 = torch.full((ndof,), np.sqrt(0.5))
    w2 = torch.full((ndof,), np.sqrt(0.5))
    diag = (rows == cols)
    w1[diag] = 0.5
    w2[diag] = 0.5

    values = torch.cat([w1, w2])
    row_idx = torch.cat([ii1, ii2])
    col_idx = torch.cat([dof, dof])
    indices = torch.stack([row_idx, col_idx])

    time1 = time.time()
    # V = torch.sparse_coo_tensor(indices, values, (n*n, ndof))
    indices_T, values_T = transpose(indices, values, n*n, ndof)
    phi_phi_T_idx, phi_phi_T_val = spspmm(indices, values, indices_T, values_T, n*n, ndof, n*n)
    time2 = time.time()
    # print("Create Symbasis: ", time2 - time1)
    return torch.sparse_coo_tensor(phi_phi_T_idx, phi_phi_T_val, (n*n, n*n)).coalesce()


def kron_X_I(X: torch.Tensor, dim):
    row_idx = torch.stack([torch.arange(X.size(0)*dim) for _ in range(X.size(1))]).T
    col_idx_1 = torch.arange(dim).reshape((dim, 1)) + torch.arange(X.size(1)).reshape((1, X.size(1))) * dim
    col_idx = torch.cat([col_idx_1 for _ in range(X.size(0))])
    values = torch.cat([X for _ in range(dim)], dim=1).reshape(-1)
    indices = torch.stack([row_idx.reshape(-1), col_idx.reshape(-1)])
    # dense = torch.kron(X.contiguous(), torch.eye(dim))
    # sparse = dense.to_sparse_coo()
    # return sparse
    return torch.sparse_coo_tensor(indices, values, (X.size(0)*dim, X.size(1)*dim)).coalesce().to(X.device)
    # return indices.to(X.device), values.to(X.device)

def kron_I_X(X: torch.Tensor, dim):
    # row_idx = torch.stack([torch.arange(X.size(0) * dim) for _ in range(X.size(1))]).T
    # col_idx_1 = torch.cat([torch.arange(X.size(1)) for _ in range(X.size(0))])
    # col_idx = torch.arange(dim).reshape((dim, 1)) * X.size(1) + col_idx_1.reshape((1, -1)) 
    # values = torch.cat([X for _ in range(dim)], dim=0).reshape(-1)
    # indices = torch.stack([row_idx.reshape(-1), col_idx.reshape(-1)])
    dense = torch.kron(torch.eye(dim), X.contiguous())
    return dense.to_sparse_coo()
    # return indices.to(X.device), values.to(X.device)
    # return torch.sparse_coo_tensor(indices, values, (X.size(0)*dim, X.size(1)*dim)).coalesce()

def create_mask_COO(
    prob_mat: torch.Tensor,     # COO
    top_k   : int,
    M_type  : str,
    M       : torch.Tensor,     # COO
):
    if top_k <= 0:
        return M
    flatten = prob_mat - M * 1e9
    # if M_type == 'entmax':
    #     if train:
    #         mask = flatten
    #     else:
    #         _, indices = torch.topk(flatten, k=top_k)
    #         mask = torch.zeros_like(flatten, dtype=prob_mat.dtype)
    #         mask[indices] = 1
    if M_type == 'STE':
        _, idx = torch.topk(flatten.values(), k=top_k)
        mask_val = torch.ones(len(idx))
        mask_idx = flatten.indices()[:,idx]
    else:
        raise NotImplementedError
    mask = torch.sparse_coo_tensor(torch.cat([mask_idx, M.indices()], dim=1), torch.cat([mask_val, M.values()]), (M.size(0), M.size(1))).coalesce()
    return mask.to(prob_mat.device)

def logits2prob_COO(
    logits: torch.Tensor,
    M_type: str,
    tau: float = 0.5
):
    # flatten = logits.view(-1)
    # if M_type == 'entmax':
    #     prob_mat = top_k * entmax15(flatten, dim=0)
    if M_type == 'STE':
        prob_mat_idx = torch.stack([torch.arange(len(logits)), torch.arange(len(logits))])
        prob_mat_val = torch.softmax(logits / tau, dim=0)
        prob_mat = torch.sparse_coo_tensor(prob_mat_idx, prob_mat_val, (len(logits), len(logits))).coalesce()
    else:
        raise NotImplementedError
    return prob_mat.to(logits.device)

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

@torch.no_grad()
def print_busses(title, parameters, constraints, top_k, M=None, M_type='STE'):
    logging.info(f"==================== {title} ====================")
    if 'X' in parameters:
        logging.info("=> X")
        n = (len(parameters['X']) + 1) // 2
        X_polar = emb2polar(parameters['X'], *constraints)
        X_mag, X_ang = X_polar[:n], X_polar[n:] / torch.pi * 180
        for i in range(len(X_mag)):
            if i == 0:
                logging.info(f"Bus-{i}: mag = {X_mag[i].item():.4f}, ang = {0:.2f}")
            else:
                logging.info(f"Bus-{i}: mag = {X_mag[i].item():.4f}, ang = {X_ang[i-1].item():.2f}")
    if 'Z' in parameters:
        logging.info("=> Z")
        n = (len(parameters['Z']) + 1) // 2
        Z_polar = emb2polar(parameters['Z'], *constraints)
        Z_mag, Z_ang = Z_polar[:n], Z_polar[n:] / torch.pi * 180
        for i in range(len(Z_mag)):
            if i == 0:
                logging.info(f"Bus-{i}: mag = {Z_mag[i].item():.4f}, ang = {0:.2f}")
            else:
                logging.info(f"Bus-{i}: mag = {Z_mag[i].item():.4f}, ang = {Z_ang[i-1].item():.2f}")
    if 'P' in parameters:
        P = parameters['P']
        logging.info(f"{'=> P'} {'=':^1}")
        logging.info(P.data)
        logging.info(f"{'=> Probability Matrix'} {'=':^1}")
        logging.info(logits2prob(P, top_k, M_type, M).data)
        logging.info(f"{'=> Mask'} {'=':^1}")
        logging.info(create_mask(P, top_k, False, 'STE', M).to(torch.long).data)
    else:
        logging.info(f"{'=> Mask'} {'=':^1}")
        logging.info(M.data)
    logging.info("===================================================")

@torch.no_grad()
def print_counter_MC(title, top_k, parameters, losses, M=None, M_type='STE'):
    logging.info(f"==================== {title} ====================")
    logging.info(f"{'=> # of samples (m)':<27} {'=':^1} {top_k:>10}")
    logging.info(f"{'=> Total loss':<27} {'=':^1} {losses[-1][0]:>10.2e}")
    logging.info(f"{'=> f(Z) - f(X)':<27} {'=':^1} {losses[-1][1]:>10.2e}")
    logging.info(f"{'=> |gradient(X)|':<27} {'=':^1} {losses[-1][2]:>10.2e}")
    logging.info(f"{'=> -lambda_min(hessian(X))':<27} {'=':^1} {losses[-1][3]:>10.2e}")
    logging.info(f"{'=> |XX^T-ZZ^T|_F':<27} {'=':^1} {losses[-1][6]:>10.2e}")
    logging.info(f"{'=> X'} {'=':^1}")
    logging.info(parameters['X'].data)
    if 'Z' in parameters:
        logging.info(f"{'=> Z'} {'=':^1}")
        logging.info(parameters['Z'].data)
    if 'P' in parameters:
        P = parameters['P']
        logging.info(f"{'=> P'} {'=':^1}")
        logging.info(P.data)
        logging.info(f"{'=> Probability Matrix'} {'=':^1}")
        logging.info(logits2prob(P, top_k, M_type, M).data)
        logging.info(f"{'=> Mask'} {'=':^1}")
        logging.info(create_mask(P, top_k, False, 'STE', M).to(torch.long).data)
    else:
        logging.info(f"{'=> Mask'} {'=':^1}")
        logging.info(M.data)
    logging.info("===================================================")

if __name__ == '__main__':
    V = torch.randn((2, 3))
    print(V)
    print(kron_I_X(V, 2).to_dense())