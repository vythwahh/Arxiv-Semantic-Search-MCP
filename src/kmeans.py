import torch


class CustomKMeans:
    """
    K-Means Clustering implemented from scratch using PyTorch.

    Groups high-dimensional embedding vectors into k clusters based on
    Euclidean distance minimization — the mathematical foundation behind
    FAISS's IVF (Inverted File Index) for approximate nearest neighbor search.

    Algorithm:
        1. Initialize k centroids randomly from data points
        2. Assign each point to nearest centroid (E-step)
        3. Recompute centroids as cluster means (M-step)
        4. Repeat until convergence or max_iter reached
    """

    def __init__(self, n_clusters: int = 10, max_iter: int = 100, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, X: torch.Tensor) -> "CustomKMeans":
        """
        Fits K-Means on input matrix X of shape (n_samples, n_features).
        """
        n_samples = X.shape[0]

        # Step 1: Initialize centroids by randomly sampling data points
        indices = torch.randperm(n_samples)[:self.n_clusters]
        self.centroids = X[indices].clone()

        for iteration in range(self.max_iter):
            # Step 2: Assign each point to nearest centroid
            self.labels = self._assign_clusters(X)

            # Step 3: Recompute centroids as cluster means
            new_centroids = torch.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                mask = self.labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(dim=0)
                else:
                    # Handle empty cluster by reinitializing randomly
                    new_centroids[k] = X[torch.randint(n_samples, (1,))]

            # Step 4: Check convergence
            centroid_shift = torch.max(torch.norm(new_centroids - self.centroids, dim=1))
            self.centroids = new_centroids

            if centroid_shift < self.tol:
                print(f"K-Means converged at iteration {iteration + 1}")
                break

        return self

    def _assign_clusters(self, X: torch.Tensor) -> torch.Tensor:
        """
        Assigns each sample to the nearest centroid using Euclidean distance.
        Uses broadcasting for efficient batch computation.
        """
        # Shape: (n_samples, n_clusters)
        distances = torch.cdist(X, self.centroids, p=2)
        return torch.argmin(distances, dim=1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predicts cluster label for new data points.
        """
        return self._assign_clusters(X)

    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.fit(X).labels

    def inertia(self, X: torch.Tensor) -> float:
        """
        Computes within-cluster sum of squared distances (WCSS).
        Lower inertia = tighter, better-defined clusters.
        Used for Elbow Method to choose optimal k.
        """
        labels = self._assign_clusters(X)
        total = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                diff = X[mask] - self.centroids[k]
                total += (diff ** 2).sum().item()
        return total