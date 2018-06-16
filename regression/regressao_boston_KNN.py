
# coding: utf-8


from sklearn.datasets import load_boston

boston = load_boston()

print(boston.DESCR)


print(boston.data.shape)

print(boston.feature_names)


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

K = 9
knn = KNeighborsRegressor(n_neighbors=K)
knn.fit(boston.data, boston.target)
print(boston.target[0])
print(knn.predict([boston.data[0]]))


y_ = knn.fit(boston.data, boston.target).predict([boston.data[12]])
print(y_)
print(boston.target[12])

import matplotlib.pyplot as plt
import numpy as np
knn = KNeighborsRegressor(n_neighbors=K)
x, y = boston.data[:50], boston.target[:50]
y_ = knn.fit(x, y).predict(x)
plt.plot(np.linspace(-1, 1, 50), y, label='data', color='black')
plt.plot(np.linspace(-1, 1, 50), y_, label='prediction', color='red')
plt.legend()
plt.show()

