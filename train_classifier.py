import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from xception_classifier import XceptionClassifier

class HRFeatureDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.resize(img, (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

image_paths = ["./imagery/high_res/img1.png", "./imagery/high_res/img2.png"]
labels = [0, 1]

dataset = HRFeatureDataset(image_paths, labels, transform=transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = XceptionClassifier(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    for imgs, lbls in loader:
        preds = model(imgs)
        loss = criterion(preds, lbls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("✅ Xception Classifier training complete.")
