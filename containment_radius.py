# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb

# load iris data
iris = datasets.load_iris()

# convert iris data into dataframe
# df = pd.DataFrame(iris['data'])

# do PCA for visualization
pca = PCA(n_components=2)
df = pd.DataFrame(pca.fit_transform(iris['data']))

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

# from Elbow method we get the best value of k is 3
k_best = 3
kmeans_best = KMeans(n_clusters=k_best)
kmeans_best.fit(df)
cluster_center = kmeans_best.cluster_centers_
df['k_means'] = kmeans_best.predict(df)
df['dist'] = np.linalg.norm(df.iloc[:,:-1].to_numpy()-cluster_center[df['k_means']], axis=1)

# find 50% containment radius of each cluster
percentage = 0.5
radius_list = []
fig, axes = plt.subplots(figsize=(16,8))
for i in range(k_best):
    radius_list.append(df.loc[df['k_means']==i]['dist'].quantile(percentage)) # use qunatile to get customized portion
    draw_circle = plt.Circle((cluster_center[i,0], cluster_center[i,1]), radius_list[i], fill=False)
    axes.set_aspect(1)
    axes.add_artist(draw_circle)
plt.scatter(df[0], df[1], c=df['k_means'], s=25, cmap='viridis')
plt.scatter(cluster_center[:,0], cluster_center[:,1], c='black', s=75, alpha=0.5)
plt.title('Clusters with 50% containtment radius')
plt.show()

