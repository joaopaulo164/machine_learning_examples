# coding: utf-8
__author__ = 'john'

# Old Faithful
# Old Faithful é um géiser localizado no Parque Nacional de Yellowstone, em Wyoming, nos Estados Unidos. Old Faithful foi nomeado em 1870 durante a Expedição Washburn e foi o primeiro géiser do parque a receber um nome.
# http://fromdatawithlove.thegovans.us/2013/05/clustering-using-scikit-learn.html
# http://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat

from sklearn import cluster, datasets
import numpy as np
from matplotlib import pyplot

def clustering(data, n):
    k = n
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(data)

    #And to get the locations of the centroids and the label of the owning cluster for each observation in the data set:

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    #Using these, we can now plot the chosen clusters and their calculated centroids:


    for i in range(k):
        # select only data observations with cluster label == i
        ds = data[np.where(labels==i)]
        # plot the data observations
        pyplot.plot(ds[:,2],ds[:,3],'o')
        # plot the centroids
        lines = pyplot.plot(centroids[i,2],centroids[i,3],'kx')
        # make the centroid x's bigger
        pyplot.setp(lines,ms=15.0)
        pyplot.setp(lines,mew=2.0)
    pyplot.title('Numero de Clusters = ' + str(k))
    pyplot.xlabel('Comprimento da Petala (cm)')
    pyplot.ylabel('Largura da Petala (cm)')
    pyplot.show()


iris = datasets.load_iris()
data = iris.data
print(data)
clustering(data, 3)
clustering(data, 6)



