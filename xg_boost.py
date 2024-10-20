from sklearn import datasets
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from xgboost import XGBClassifier

ESTIMATORS = 4
MAX_DEPTH = 4

#iris = datasets.load_iris()
#X, y = iris.data, iris.target

#wine = datasets.load_wine()
#X, y = wine.data, wine.target


cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target

for i in range(2, ESTIMATORS+1):
    for j in range(1, MAX_DEPTH+1):
        xgbst = XGBClassifier(n_estimators=i, max_depth=j, learning_rate=1, objective='binary:logistic')
        
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        
        scores = cross_val_score(xgbst, X, y, cv=skf)
        
        print(f"[{i},{j}], Mean stratified validation score: {np.mean(scores)}")