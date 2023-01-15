# Optimal Threshold Labeling for Ordinal Regression Methods

Published in Transactions on Machine Learning Research

pdf: https://openreview.net/forum?id=mHSAy1n65Z

## for face-age datasets

### preparations
1. download MORPH-2, CACD, and AFAD datasets from https://ebill.uncw.edu/C20231_ustores/web/, http://bcsiriuschen.github.io/CARC/ (Original face images), and https://github.com/afad-dataset/tarball into ./datasets/
2. preprocessing MORPH-2 and AFAD datasets: python preprocess-morph2.py, preprocess-cacd.py at ./faceage/datasets/

### experiments
1. experiments: e.g., python afad-AD.py --cuda 0 at ./faceage/afad/ (for all datasets and methods)
2. make tables: python table.py at ./faceage/results/
3. make figures: python figure.py at ./faceage/results/

## for various-domain datasets

1. make tables: python bash.py and table.py in ./various/trials/
2. make tables: python bash.py and ratio.py in ./various/trials/
3. make figures: python bash.py and python plot.py at ./various/plot/