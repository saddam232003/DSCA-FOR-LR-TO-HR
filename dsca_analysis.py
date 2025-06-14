import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

def spearman_rank_matrix(X):
    return np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), axis=1, arr=X)

def rbf_kernel(X, sigma=1.0):
    pairwise_sq_dists = cdist(X, X, 'sqeuclidean')
    K = np.exp(-pairwise_sq_dists / (2 * sigma**2))
    return K

def deep_spearman_corr(X_lr, X_hr, reg_lambda=1e-3):
    X_lr_rank = spearman_rank_matrix(X_lr)
    X_hr_rank = spearman_rank_matrix(X_hr)
    K_lr = rbf_kernel(X_lr_rank)
    K_hr = rbf_kernel(X_hr_rank)
    M_lr = K_lr.T @ K_lr
    M_hr = K_hr.T @ K_hr
    A = K_lr.T @ K_hr
    left = np.block([[np.zeros_like(A), A], [A.T, np.zeros_like(A)]])
    right = np.block([[M_lr + reg_lambda * np.eye(M_lr.shape[0]), np.zeros_like(M_lr)],
                      [np.zeros_like(M_hr), M_hr + reg_lambda * np.eye(M_hr.shape[0])]])
    eigvals, eigvecs = eigh(left, right)
    alpha = eigvecs[:K_lr.shape[0], -1]
    beta = eigvecs[K_lr.shape[0]:, -1]
    Cl = alpha @ M_lr
    Ch = beta @ M_hr
    return Cl, Ch, alpha, beta
