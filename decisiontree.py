from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Calculate the accuracy of the model
def evaluate(self, X, y):
    y_pred = self.predict(X)
    total = 0
    for i in range(len(y_pred)):
        total += (y_pred[i] == y[i])
    
    return total / len(y_pred)*100


#iris = datasets.load_iris()
#X, y = iris.data, iris.target


#wine = datasets.load_wine()
#X, y = wine.data, wine.target


cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                    test_size=0.3, 
                                    random_state=1,
                                    stratify=y)

dt = DecisionTreeClassifier(max_depth=10,
                            criterion='gini',
                            random_state=0,
                            min_samples_split=10,
                            min_samples_leaf=5)

dt = dt.fit(X_train, y_train)

print('Training Accuracy: %.2f' % evaluate(dt, X_train, y_train))

print('Testing Accuracy: %.2f' % evaluate(dt, X_test, y_test))

tree.plot_tree(dt)

plt.show()
