from sklearn import datasets
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

ESTIMATORS = 30
MIN_SPLIT = 5

#iris = datasets.load_iris()
#X, y = iris.data, iris.target

#wine = datasets.load_wine()
#X, y = wine.data, wine.target


cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target

for i in range(10, ESTIMATORS+1, 10):
    for j in range(2, MIN_SPLIT+1):
        random_forest = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)

        skf = StratifiedKFold(n_splits=10, shuffle=True)

        scores = cross_val_score(random_forest, X, y, cv=skf)

        #print("Cross-validation scores:", scores)
        print(f"[{i},{j}], Mean stratified validation score: {np.mean(scores)}")