import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def show_image_pair(lr_path, hr_path):
    lr_img = cv2.imread(lr_path, cv2.IMREAD_GRAYSCALE)
    hr_img = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(lr_img, cmap='gray')
    axs[0].set_title('Low-Res')
    axs[1].imshow(hr_img, cmap='gray')
    axs[1].set_title('High-Res')
    for ax in axs:
        ax.axis('off')
    plt.show()

def compare_classification(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, output_dict=False)
    print("Classification Report:
", report)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# Example usage
# show_image_pair("./imagery/low_res/img1.png", "./imagery/high_res/img1.png")
# compare_classification([0, 1, 1], [0, 1, 0])
