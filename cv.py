from sklearn import svm
from sklearn import datasets
import numpy as np
import pandas as pd
import math
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.cross_validation import KFold


iris = datasets.load_iris()
X = iris.data
y = iris.target

kfold = KFold(len(X), n_folds=5)
svc = svm.SVC(kernel='linear')

#printing score as well as mean and std
print cross_val_score(svc, X, y, cv=kfold)
print np.mean(cross_val_score(svc, X, y, cv=kfold))
print np.std(cross_val_score(svc, X, y, cv=kfold))

#printing f1, precision, recall
print cross_val_score(svc, X, y, cv=kfold, scoring='f1')
print cross_val_score(svc, X, y, cv=kfold, scoring='r2')
print cross_val_score(svc, X, y, cv=kfold, scoring='recall')