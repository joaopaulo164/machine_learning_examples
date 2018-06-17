# coding: utf-8

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

boston = load_boston()
lm = LinearRegression()
x, y = boston.data[:100], boston.target[:100]
y_ = lm.fit(x, y).predict(x)
plt.plot(np.linspace(0, 1, 100), y, label='Dado', color='black')
plt.plot(np.linspace(0, 1, 100), y_, label='Predição', color='red')
plt.legend()
plt.title("Valor médio dos imóveis em Boston (1978)")
plt.ylabel("Cada ponto representa $1000")
plt.xlabel("Conjunto de dados (%)")
plt.show()