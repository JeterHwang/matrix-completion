import cvxpy as cp
import numpy as np
from utils import create_Z, sample_mask

def SDP(n, m, r, p, mu, PSD=False):
    Z, mu_Z = create_Z(n, m, r, mu, PSD)
    mask, A = sample_mask(n, m, p)
    
    X = cp.Variable((n+m, n+m), symmetric=True)
    constraints = [X >> 0]
    constraints += [
        cp.trace(A[i] @ X) == cp.trace(A[i] @ Z) for i in range(len(A))
    ]
    prob = cp.Problem(
        cp.Minimize(
            0.5 * cp.trace(np.identity(n+m) @ X)
        ),
        constraints
    )
    prob.solve(solver=cp.MOSEK, verbose=False)

    W1 = X.value[:n, :n]
    W2 = X.value[n:, n:]
    X_star = X.value[:n, n:]

    return Z[:n, n:], W1, W2, X_star, mu_Z, np.sum(mask), np.sum((Z[:n, n:] - X_star)**2)