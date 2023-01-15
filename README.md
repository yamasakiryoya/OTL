# Optimal Threshold Labeling for Ordinal Regression Methods

Published in Transactions on Machine Learning Research

pdf: https://openreview.net/forum?id=mHSAy1n65Z

## preparations
1. download MORPH-2, CACD, and AFAD datasets from https://ebill.uncw.edu/C20231_ustores/web/, http://bcsiriuschen.github.io/CARC/, and https://github.com/afad-dataset/tarball into ./datasets/
2. preprocessing MORPH-2 and AFAD datasets: python preprocess-morph2.py, preprocess-cacd.py at ./datasets/
3. separate training, validation, and test sets: python makecsv-morph2.py, python makecsv-cacd.py, python makecsv-afad.py at ./datasets/

## experiments for learning curve
1. experiments: e.g., python afad-ANLCL-LC.py --cuda 0 at ./afad/ (for all datasets)
2. make tables: python figure.py at ./results/

## experiments on classification performance
1. experiments: e.g., python afad-ANLCL.py --cuda 0 at ./afad/ (for all datasets and methods)
2. make tables: python table.py at ./results/

## experiments on computation time
1. experiments: e.g., python afad-ANLCL-Time.py --cuda 0 at ./afad/ (for all datasets)
2. make tables: e.g., read ./afad/MTM/ANLCL-Time/training.log