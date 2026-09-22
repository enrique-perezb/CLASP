import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.manifold import spectral_embedding
from sklearn.preprocessing import normalize

class StabilityLoss:

    def __init__(self, X, pct, num_subsamples, k):
        self.X = X   # nxp, data
        self.pct = pct   # scalar, percentage of participants per subsample
        self.num_subsamples = num_subsamples   # scalar, number of subsamples for consensus matrix
        self.k = k   # scalar, number of clusters
        self.sigma = np.median(pairwise_distances(X))
        if self.sigma == 0.0:
            self.sigma = 1e-5

    def get_loss(self):
        silhouette_vals = self._get_silhouette_vals(self.X)
        return np.mean(silhouette_vals)

    def _get_silhouette_vals(self, data):
        n_participants = data.shape[0]
        silhouette_vals = []

        for t in range(self.num_subsamples):

            # Subsample participants
            n_sub = int(self.pct * n_participants)
            idx = np.random.choice(n_participants, n_sub, replace=False)
            X_sub = data[idx]

            # Cluster subsample
            labels = self._cluster(X_sub)

            # Reconstruct the EXACT affinity matrix used in _cluster
            affinity = np.exp(-pairwise_distances(X_sub)**2 / (2 * self.sigma**2))

            # Extract the spectral embedding
            embedding = spectral_embedding(affinity, n_components=self.k)
            
            # Row-normalize the embedding (this is what assign_labels='kmeans' does under the hood)
            embedding_normalized = normalize(embedding, norm='l2', axis=1)

            # Evaluate silhouette score on the properly normalized space
            score = silhouette_score(embedding_normalized, labels)

            silhouette_vals.append(score)

        return silhouette_vals
        
    def _cluster(self, data):         
        sc = SpectralClustering(
            n_clusters=self.k,
            affinity='rbf',
            gamma=1.0 / (2 * self.sigma**2),
            assign_labels='kmeans',
            n_init=1
        )
        
        return sc.fit_predict(data)