import numpy as np

def compute_rbfn_matrix(C_lr, sigma=1.0):
    n = C_lr.shape[0]
    Phi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = C_lr[i] - C_lr[j]
            Phi[i, j] = np.exp(-np.linalg.norm(diff) ** 2 / (2 * sigma ** 2))
    return Phi

def map_lr_to_hr(C_lr, C_hr, sigma=1.0):
    Phi = compute_rbfn_matrix(C_lr, sigma)
    W = np.linalg.pinv(Phi) @ C_hr
    return W, Phi
