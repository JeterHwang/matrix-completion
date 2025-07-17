import cvxpy as cp
import numpy as np
from src.utils import augment_mask, augment_Z

def SDP(_Z: np.ndarray, mask: np.ndarray):
    n, m = Z.shape
    A = augment_mask(mask)
    Z = augment_Z(_Z)
    X = cp.Variable(Z.shape, symmetric=True)
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

    return W1, W2, X_star, np.sum((Z[:n, n:] - X_star)**2)