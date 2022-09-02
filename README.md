# Modified Threshold Method for Ordinal Regression

## for face-age datasets

### preparations
1. download MORPH-2, CACD, and AFAD datasets from https://ebill.uncw.edu/C20231_ustores/web/, http://bcsiriuschen.github.io/CARC/ (Original face images), and https://github.com/afad-dataset/tarball into ./datasets/
2. preprocessing MORPH-2 and AFAD datasets: python preprocess-morph2.py, preprocess-cacd.py at ./faceage/datasets/

### experiments for learning curve
1. experiments: e.g., python afad-LC.py --cuda 0 at ./faceage/afad/ (for all datasets)
2. make figures: python figure.py at ./faceage/results/

### experiments on classification performance
1. experiments: e.g., python afad-ANLCL.py --cuda 0 at ./faceage/afad/ (for all datasets and methods)
2. make tables: python table.py at ./faceage/results/

### experiments on computation time
1. experiments: python afad-Time.py --cuda 0 at ./faceage/afad/
2. make tables: read ./faceage/afad/MTM/Time/training.log

## for various-domain datasets

1. make tables: python bash.py and test.ipynb in ./various/trials/
2. make figures: python bash.py and python plot.py at ./various/plot/