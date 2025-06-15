def adaptive_dim_reduction(features, threshold=0.95):
    cov = np.cov(features.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    total = np.sum(eigvals)
    running_total = 0
    k = 0
    for val in reversed(eigvals):
        running_total += val
        k += 1
        if running_total / total >= threshold:
            break
    selected_components = eigvecs[:, -k:]
    return features @ selected_components