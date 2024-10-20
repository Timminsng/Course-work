from sklearn import datasets
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

MAX_DEPTH = 5
MIN_SPLIT = 6
ESTIMATORS = 150

#iris = datasets.load_iris()
#X, y = iris.data, iris.target

#wine = datasets.load_wine()
#X, y = wine.data, wine.target


cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target

for i in range(50, ESTIMATORS+1, 50):
    for j in range(2, MIN_SPLIT+1, 2):
        for k in range(1, MAX_DEPTH+1, 2):
            dt = DecisionTreeClassifier(max_depth=k, criterion='gini', random_state=0, min_samples_split=j)
            ada_boost = AdaBoostClassifier(estimator=dt, n_estimators=i, algorithm="SAMME")

            skf = StratifiedKFold(n_splits=10, shuffle=True)

            scores = cross_val_score(ada_boost, X, y, cv=skf)

            #print("Cross-validation scores:", scores)
            print(f"[{i},[{j},{k}]], Mean stratified validation score: {np.mean(scores)}")