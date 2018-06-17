# coding: utf-8
__author__ = 'john'

# http://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat

from sklearn import cluster
import numpy as np
from matplotlib import pyplot

def clustering(data, n):
    k = n
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    for i in range(k):
        ds = data[np.where(labels==i)]
        pyplot.plot(ds[:,0],ds[:,1],'o')
        lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
        pyplot.setp(lines,ms=15.0)
        pyplot.setp(lines,mew=2.0)
    pyplot.title('Numero de Clusters = ' + str(k))
    pyplot.xlabel('Tempo de erupção (min)')
    pyplot.ylabel('Tempo entre erupções (min)')
    pyplot.show()

data = np.genfromtxt('faithful_data.csv', delimiter=',')
print(data)
clustering(data, 2)
clustering(data, 6)



