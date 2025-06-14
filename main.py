import torch
import numpy as np
from torch.utils.data import DataLoader
from dsca_model import SmallDCNN
from dsca_analysis import deep_spearman_corr
from rbfn_mapping import map_lr_to_hr
from loader import PairedImageDataset

def extract_features(model, loader):
    model.eval()
    features_lr, features_hr = [], []
    with torch.no_grad():
        for lr, hr in loader:
            feat_lr = model(torch.tensor(lr))
            feat_hr = model(torch.tensor(hr))
            features_lr.append(feat_lr.numpy())
            features_hr.append(feat_hr.numpy())
    return np.vstack(features_lr), np.vstack(features_hr)

dataset = PairedImageDataset("./imagery/low_res", "./imagery/high_res", img_size=(32, 32))
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = SmallDCNN()
X_lr, X_hr = extract_features(model, loader)
Cl, Ch, alpha, beta = deep_spearman_corr(X_lr, X_hr)
W, Phi = map_lr_to_hr(Cl, Ch)

np.save("projection_matrix_W.npy", W)
print("✅ Completed D-SCA + RBFN projection matrix training")
