import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

""" Data preparation """
w            = [30,    30,     50,     20,     50,     40,    80,    30,   60,    60,    80,    80, 20, 20]
l            = [30,    30,     30,     40,     50,     80,    80,    30,   60,    60,    80,    20, 80, 20]
alt_peak_IC  = [11.43, 16.68,  27.05,  21.08,  40.54,  42.12, 71.94, 4.42, 13.27, 26.85,  25.13, 37.85, 17.60, 19.58]
ust_peak_IC  = [6.58,  10.29,  17.77,  13.00,  28.37,  29.59, 53.00, 2.61, 9.80,  19.68,  18.97, 25.47, 12.01, 12.44]
alt_peak_dis = [8.36, 12.74,  23.09,  16.98,  37.08,  38.63, 64.73, 3.22, 11.18, 23.32, 22.01, 32.32, 15.78, 17.32]
ust_peak_dis = [5.41, 8.97,   16.29,  11.19,  27.07,  28.64, 51.08, 2.15, 8.85,  18.77, 18.07, 24.44, 11.50, 11.40]

W = np.asarray(w*2)
L = np.asarray(l*2)

IC_param = np.divide(ust_peak_IC,alt_peak_IC)
dis_param = np.divide(ust_peak_dis,alt_peak_dis)
params = np.hstack((IC_param,dis_param))

""" The data is not linearly seperable """
y = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
data = np.asarray([W*L, params, np.asarray(y)])
X = data[:-1].T

from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
# Scaling is important for the SVM, use standart scaler


""" Linear SVM with polynomial features """
from sklearn.svm import LinearSVC
polynomial_svm_clf = Pipeline((
                                ("poly_features", PolynomialFeatures(degree=3)),
                                ("scaler", StandardScaler()),
                                ("svm_clf", LinearSVC(C=3, loss="hinge")) 
                                # C: Defines wideness of the support vectors, reduce C if overfitting, increase if underfitting.
                            ))

""" SVM with Polynomial kernel trick """
poly_kernel_svm_clf = Pipeline((
                                ("scaler", StandardScaler()),
                                ("svm_clf", SVC(kernel="poly", degree=5, coef0=2, C=5))
                                # Coef0: influence of high degree polynomials against low degree polynomials

                              ))     

""" SVM with Gaussian RBF similarity feature """
rbf_kernel_svm_clf = Pipeline((
                                ("scaler", StandardScaler()),
                                ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
                                # Gamma: regularization parameter reduce if overfitting, increase if underfitting.
                              ))                 

""" Grid search with SVM """
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1],
              'gamma': [5, 4, 3 ,2],
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, refit = True)
grid.fit(X, y)
print(grid.best_estimator_)

# rbf_kernel_svm_clf.fit(X, y)    
poly_kernel_svm_clf.fit(X, y)          
# rbf_kernel_svm_clf.fit(X, y)

print(poly_kernel_svm_clf.predict(X))

