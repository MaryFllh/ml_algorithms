import numpy as np

import matplotlib.pyplot as plt 
from sklearn import datasets

from utils import euclidean_distance

class KMeans:
    def __init__(self, clusters_num, iters_num=100):
       self.clusters_num = clusters_num
       self.iters_num = iters_num
       self.centroids = []
       self.clusters = [[] for _ in range(self.clusters_num)]

    def fit(self, X_train):
        self.X = X_train
        self.n_samples, self.n_features = X.shape
        random_idx = np.random.choice(len(X_train), self.clusters_num, replace=False)
        self.centroids = [X_train[idx] for idx in random_idx]
        for _ in range(self.iters_num):
            self.clusters = self._create_clusters(self.centroids)
            
            self.plot()
            
            centroids_before_update = self.centroids
            self.centroids = self._update_centroids(self.clusters)
            if self._is_converged(centroids_before_update, self.centroids):
                break
            self.plot()
    
    def _is_converged(self, centroids_old, centroids):
        # distances between old and new centroids, for all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.clusters_num)]
        return sum(distances) == 0

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.clusters_num)]
        for idx, sample in enumerate(self.X):
            distance_to_centroids = [euclidean_distance(sample, c) for c in centroids]
            clusters[np.argmin(distance_to_centroids)].append(idx)
        return clusters

    def _update_centroids(self, clusters):
        
        centroids = np.zeros((self.clusters_num, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def predict(self, X):
        for x in X:
            distance_to_centroids = [euclidean_distance(x, c) for c in self.centroids]
            return self.clusters[np.argmin(distance_to_centroids)]
    
    def plot(self):
        _, ax = plt.subplots(figsize=(12, 8))

        for _, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()


if __name__=="__main__":
    np.random.seed(42)
    X, y = datasets.make_blobs(centers=3, n_samples=200, n_features=2, shuffle=True, random_state=40)

    k = KMeans(len(np.unique(y)))
    k.fit(X)
    plt.show()
