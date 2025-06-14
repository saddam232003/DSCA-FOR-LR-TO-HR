D-SCA: Advanced Nonlinear Optimization of Medical Images using Deep Spearman Correlation Analysis

This repository contains the full implementation of the research paper:
> "Advanced Nonlinear Optimization of Low- and High-Resolution Medical Images Using Adaptive Deep Spearman Correlation Analysis (D-SCA)"

This deep learning pipeline enhances medical images using DCNN + D-SCA and classifies them using an Xception-based model.

#  Key Components

1. SmallDCNN - Feature extractor for LR and HR medical images.
2. D-SCA - Spearman rank-based non-linear optimization using kernel Hilbert space mapping.
3. RBFN Mapping - Learns transformation matrix to map LR features to HR space.
4. Xception Classifier** - Classifies enhanced HR images using transfer learning.
5. Visualization & Comparison** - Visualizes results, evaluates performance.

#  Installation
```bash
git clone 
cd dsca-project
pip install -r requirements.txt

Created by Saddam. Reach me via GitHub issues Saddam_khokhar@hotmail.com.
