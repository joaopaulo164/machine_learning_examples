__author__ = 'john'

from sklearn import tree, datasets
from sklearn.metrics import classification_report, confusion_matrix

iris = datasets.load_iris()
x = iris.data
y = iris.target
clf = tree.DecisionTreeClassifier(max_leaf_nodes=3)
clf = clf.fit(x, y)
labels = clf.predict(x)
print("Matriz de Confusao")
print(confusion_matrix(y, labels))
print("\n Relatorio")
print(classification_report(y, labels))



# http://scikit-learn.org/stable/modules/tree.html