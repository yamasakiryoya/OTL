# Modified Threshold Method for Ordinal Regression

1. download MORPH-2, CACD, and AFAD datasets from https://ebill.uncw.edu/C20231_ustores/web/, http://bcsiriuschen.github.io/CARC/, and https://github.com/afad-dataset/tarball into ./datasets/
2. preprocessing MORPH-2 and AFAD datasets: python preprocess-morph2.py, preprocess-cacd.py at ./datasets/
3. separate training, validation, and test sets: python makecsv-morph2.py, python makecsv-cacd.py, python makecsv-afad.py at ./datasets/
4. experiments: e.g., python afad-NLL.py --cuda 0 at ./afad/ (for all datasets and methods)
5. make tables: tables.ipynb at ./results/
