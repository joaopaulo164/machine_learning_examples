
# coding: utf-8

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

boston = load_boston()
print(boston.DESCR)

#x, y = boston.data, boston.target
lm = LinearRegression()

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
# lm.fit(X_train,y_train)
#
# print("Printando a intercepçao")
# print(lm.intercept_)


#coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
#coeff_df = pd.DataFrame(lm.coef_,boston.feature_names,columns=['Coefficient'])
#print(coeff_df)


x, y = boston.data[:100], boston.target[:100]
y_ = lm.fit(x, y).predict(x)
plt.plot(np.linspace(-1, 1, 100), y, label='Dado', color='black')
plt.plot(np.linspace(-1, 1, 100), y_, label='Predição', color='red')
plt.legend()
plt.show()

