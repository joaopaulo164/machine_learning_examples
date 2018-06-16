
# coding: utf-8

from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

boston = load_boston()
print(boston.DESCR)

K = 9
# knn = KNeighborsRegressor(n_neighbors=K)
# knn.fit(boston.data, boston.target)
# print(boston.target[0])
# print(knn.predict([boston.data[0]]))
#
# y_ = knn.fit(boston.data, boston.target).predict([boston.data[12]])
# print(y_)
# print(boston.target[12])


knn = KNeighborsRegressor(n_neighbors=K)
x, y = boston.data[:100], boston.target[:100]
y_ = knn.fit(x, y).predict(x)
plt.plot(np.linspace(-1, 1, 100), y, label='Dado', color='black')
plt.plot(np.linspace(-1, 1, 100), y_, label='Predição', color='red')
plt.legend()
plt.show()

