import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from utils import euclidean_distance

class KMeans:
    def __init__(self, k: int, iter_nums=100):
        self.k = k
        self.iter_nums = iter_nums

    def fit(self, X: np.array):
        self.X = X
        self.sample_nums, _ = self.X.shape
        
        # initialise centroids
        random_idx = np.random.choice(self.sample_nums, self.k, replace=False)
        centroids = [self.X[idx] for idx in random_idx]
        
        for _ in range(self.iter_nums):
            clusters = self._create_clusters(centroids)
            centroids_before_updates = centroids
    
            self.plot(clusters, centroids)
    
            centroids = self._update_centroids(clusters)
            
            if self._has_converged(centroids_before_updates, centroids):
                break

            self.plot(clusters, centroids)
            
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for sample_idx in range(self.sample_nums):
            distance_to_centroids = [euclidean_distance(self.X[sample_idx], c) for c in centroids]
            clusters[np.argmin(distance_to_centroids)].append(sample_idx)
        return clusters

    def _update_centroids(self, clusters):
        centroids = []
        for _, point_idx in enumerate(clusters):
            centroids.append(np.mean(self.X[point_idx], axis=0))
        print(centroids)
        return centroids
    
    def plot(self, clusters, centroids):
        _, ax = plt.subplots()

        for _, idx in enumerate(clusters):
            points = self.X[idx].T
            ax.scatter(*points)
        
        for c in centroids:
            ax.scatter(*c, marker='x', color='black', linewidth=3)

        plt.show()
    
    def _has_converged(self, old_centroids, centroids):
        distances = [euclidean_distance(old_centroids[i], centroids[i]) for i in range(self.k)]
        return sum(distances) == 0


if __name__=="__main__":
    np.random.seed(42)
    X, y = make_blobs(centers=3, n_samples=200, n_features=2, shuffle=True, random_state=40)

    k = KMeans(len(np.unique(y)))
    k.fit(X)
