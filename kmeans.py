from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1.713,1.586], [0.180,1.786], [0.353,1.240],[0.940,1.566],[1.486,0.759],[1.266,1.106],[1.540,0.419],[0.459,1.799],[0.773,0.186]])
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

print("Number of iterations:", kmeans.n_iter_)
print("Cluster centers:", kmeans.cluster_centers_)
print("Labels:", kmeans.labels_)
print('Cluster number of Data Point (0.906,0.606) is:', kmeans.predict([[0.906, 0.606]]))
