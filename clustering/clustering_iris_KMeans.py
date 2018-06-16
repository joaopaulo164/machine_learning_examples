# coding: utf-8

__author__ = 'john'

#Importa a biblioteca
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

#Assume que você tem X (previsor) e Y (alvo) para dados de treino e x_test(previsor)
data = load_iris()

x = data.data
y = data.target
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.25, random_state=42)

#Cria modelo de objeto de classificação KMeans
k_means = KMeans(n_clusters=3, random_state=0)

#Treina o modelo usando os dados de treine e confere o score
k_means.fit(x_treino, y_treino)
#Prevê o resultado
predicted = k_means.predict(x_teste)

labels = k_means.predict(x_teste)

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
print(k_means.score(x_teste, y_teste))

print("\n")
print("Matriz de Confusao")
print(confusion_matrix(y_teste, labels))

print("\n")
print("Relatorio")
print(classification_report(y_teste, labels))
