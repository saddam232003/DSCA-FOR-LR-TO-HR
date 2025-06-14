# DSCA-FOR-LR-TO-HR
This project implements D-SCA: a deep learning pipeline to map low-resolution medical images to high-resolution representations using Spearman correlation, RBFN, and Xception-based classification. Ideal for enhancing diagnostic accuracy from limited image data. Ready-to-run and fully customizable.
#  D-SCA: Advanced Nonlinear Optimization of Medical Images using Deep Spearman Correlation Analysis
This repository contains the full implementation:
 "Advanced Nonlinear Optimization of Low- and High-Resolution Medical Images Using Adaptive Deep Spearman Correlation Analysis (D-SCA)"
This deep learning pipeline enhances medical images using DCNN + D-SCA and classifies them using an Xception-based model.
#  Key Components
1. SmallDCNN - Feature extractor for LR and HR medical images.
2. D-SCA - Spearman rank-based non-linear optimization using kernel Hilbert space mapping.
3. RBFN Mapping - Learns transformation matrix to map LR features to HR space.
4. Xception Classifier - Classifies enhanced HR images using transfer learning.
5. Visualization & Comparison - Visualizes results, evaluates performance.
#  Installation
```bash
git clone (https://github.com/saddam232003/DSCA-FOR-LR-TO-HR)
cd dsca-project
pip install -r requirements.txt

