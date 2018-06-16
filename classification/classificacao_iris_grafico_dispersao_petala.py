import pydataset as pydataset

__author__ = 'john'


import numpy as np
from sklearn import tree, datasets
import matplotlib.pyplot as plt
import pydataset

def specie_color(x):
    if x == 'setosa':
        return 'red'
    elif x == 'versicolor':
        return 'yellow'
    return 'blue'

#iris = datasets.load_iris()

iris = pydataset.data('iris')
print(iris)

#x = iris.data
#print(x)
#len(x)

#y = iris.target
#print(y)
#len(y)

#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(x, y)
#clf.predict(x)

#acuracia = clf.score(x, y)
#print("Acuracia")
#print(acuracia)

iris['SpeciesNumber'] = iris['Species'].apply(specie_color)

plt.scatter(
    iris['Petal.Length'], iris['Petal.Width'], sizes=20 * iris['Petal.Length'],
    c=iris['SpeciesNumber'], cmap='viridis', alpha=0.8
)

plt.show()

# http://scikit-learn.org/stable/modules/tree.html