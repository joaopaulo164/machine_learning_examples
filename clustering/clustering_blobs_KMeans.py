# coding: utf-8

__author__ = 'john'

#Importa a biblioteca
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

#Assume que você tem X (previsor) e Y (alvo) para dados de treino e x_test(previsor)
data = make_blobs(n_samples=200, n_features=2,
                           centers=4, cluster_std=1.8,random_state=101)


plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
plt.show()


# x = data.data
# y = data.target
# x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.25, random_state=42)

#Cria modelo de objeto de classificação KMeans
kmeans = KMeans(n_clusters=4)

#Treina o modelo usando os dados de treine e confere o score
kmeans.fit(data[0])

print("Cluster Centers")
print(kmeans.cluster_centers_)

print("\n")
print("LABELS")
print(kmeans.labels_)

print("\n")
print("LABELS 2")
print(kmeans.predict(data[0]))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

plt.show()
