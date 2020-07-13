# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 23:20:42 2020

@author: Jimmy
"""

from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# load iris data
iris = datasets.load_iris()

# convert iris data into dataframe
df = pd.DataFrame(iris['data'])

# use Elbow method to find Kmeans cluster number
distortions = []
K = range(1,10)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    distortions.append(kmeans.inertia_)
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# from Elbow method we get the best value of k is 3
kmeans_best = KMeans(n_clusters=3)
kmeans_best.fit(df)
cluster_center = kmeans_best.cluster_centers_
df['k_means']=kmeans_best.predict(df)
df.sort_values(by=0)