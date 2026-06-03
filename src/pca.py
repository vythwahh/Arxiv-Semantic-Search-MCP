import torch


class CustomPCA:
    """
    Principal Component Analysis implemented from scratch using PyTorch SVD.

    Uses Singular Value Decomposition directly on the centered data matrix
    instead of explicit covariance matrix computation — more numerically stable
    and memory-efficient for high-dimensional embeddings (e.g., 768-dim).

    Mathematical equivalence:
        X_centered = U @ diag(S) @ V^T  (SVD)
        Eigenvalues of covariance = S^2 / (n - 1)
        Principal components = V (right singular vectors)
    """

    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.total_variance = None  # Store full variance for accurate ratio

    def fit(self, X: torch.Tensor) -> "CustomPCA":
        """
        Fits PCA on input matrix X of shape (n_samples, n_features).
        Uses SVD for numerical stability over explicit covariance computation.
        """
        # Step 1: Center the data
        self.mean = X.mean(dim=0)
        X_centered = X - self.mean

        n = X_centered.shape[0]

        # Step 2: SVD decomposition — more stable than X^T @ X for large matrices
        # U: (n, n), S: (min(n,d),), Vh: (d, d)
        _, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

        # Step 3: Eigenvalues = S^2 / (n - 1)
        eigenvalues = (S ** 2) / (n - 1)

        # Step 4: Store TOTAL variance before truncation for accurate ratio
        self.total_variance = eigenvalues.sum()

        # Step 5: Keep top k components
        self.components = Vh[:self.n_components].T  # Shape: (d, k)
        self.explained_variance = eigenvalues[:self.n_components]

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Projects X onto principal components.
        Handles device consistency automatically.
        """
        # Ensure same device as stored components
        mean = self.mean.to(X.device)
        components = self.components.to(X.device)

        X_centered = X - mean
        return X_centered @ components

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.fit(X).transform(X)

    def explained_variance_ratio(self) -> torch.Tensor:
        """
        Returns proportion of TOTAL variance explained by each selected component.
        Now correctly reflects how much information is retained vs. discarded.

        Example: ratio.sum() = 0.85 means top-k components retain 85% of information.
        """
        return self.explained_variance / self.total_variance