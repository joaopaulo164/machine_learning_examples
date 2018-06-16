__author__ = 'john'

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

iris = load_iris()
clf = tree.DecisionTreeClassifier(max_leaf_nodes=3)
#clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

print(iris.data)
print(iris.target)

# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.format = 'pdf'
# graph.render("iris")

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")

# http://scikit-learn.org/stable/modules/tree.html