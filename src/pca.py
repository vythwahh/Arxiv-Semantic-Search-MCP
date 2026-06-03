import torch


class CustomPCA:
    """
    Principal Component Analysis (PCA) implemented from scratch using PyTorch.
    
    Reduces high-dimensional embedding vectors (e.g., 768-dim) to a lower 
    dimensional space (e.g., 128-dim) by finding the directions of maximum 
    variance in the data — the principal components.
    
    Mathematical foundation:
        1. Center the data: X = X - mean(X)
        2. Compute covariance matrix: C = (X^T @ X) / (n - 1)
        3. Eigendecomposition: C = V @ diag(λ) @ V^T
        4. Project: X_reduced = X @ V[:, :k]
    """

    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.mean = None
        self.components = None  # Principal components (eigenvectors)
        self.explained_variance = None

    def fit(self, X: torch.Tensor) -> "CustomPCA":
        """
        Fits PCA on the input matrix X of shape (n_samples, n_features).
        """
        # Step 1: Center the data
        self.mean = X.mean(dim=0)
        X_centered = X - self.mean

        # Step 2: Compute covariance matrix
        n = X_centered.shape[0]
        cov_matrix = (X_centered.T @ X_centered) / (n - 1)

        # Step 3: Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        # Sort by descending eigenvalue (largest variance first)
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Step 4: Keep top k components
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Projects X onto the principal components.
        Returns shape (n_samples, n_components).
        """
        X_centered = X - self.mean
        return X_centered @ self.components

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.fit(X).transform(X)

    def explained_variance_ratio(self) -> torch.Tensor:
        """
        Returns the proportion of variance explained by each component.
        Useful for choosing optimal n_components.
        """
        return self.explained_variance / self.explained_variance.sum()