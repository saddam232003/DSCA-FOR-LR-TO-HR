def hilbert_lift(features, centers, bandwidth):
    lifted_features = []
    for x in features:
        lifted = np.exp(-np.linalg.norm(x - centers, axis=1)**2 / (2 * bandwidth**2))
        lifted_features.append(lifted)
    return np.array(lifted_features)