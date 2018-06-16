__author__ = 'john'


import numpy as np
from sklearn import tree, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

iris = datasets.load_iris()

#iris = pydataset.data('iris')
#print(iris)

x = iris.data
y = iris.target
print(len(x))
print(x)
print(len(y))
print(y)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.25, random_state=42)

print("Treino")
print(len(x_treino))
print(x_treino)

print("Teste")
print(len(x_teste))
print(x_teste)

clf = tree.DecisionTreeClassifier(max_leaf_nodes=3)
#clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_treino, y_treino)
#clf.predict(x)

labels = clf.predict(x_teste)

print("Labels")
print(len(labels))
print(labels)

#acuracia = clf.score(x, y)
#print("Acuracia")
#print(acuracia)

print("\n")
print(np.sum(labels == y_teste))
print("\n")
print((labels == y_teste).sum())
print("\n")
print("Acuracia")
print(100 * (labels == y_teste).sum() / len(x_teste))

print("\n")
print("Acuracia 2")
print(clf.score(x_teste, y_teste))

print("\n")
print("Matriz de Confusao")
print(confusion_matrix(y_teste, labels))

print("\n")
print("Relatorio")
print(classification_report(y_teste, labels))



# http://scikit-learn.org/stable/modules/tree.html