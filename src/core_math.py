import torch

def custom_cosine_similarity(vector_a: torch.Tensor, vector_b: torch.Tensor) -> float:
    """
    Computes the Cosine Similarity between two embedding vectors based on 
    Linear Algebra principles: Dot_Product(A, B) / (Norm(A) * Norm(B)).
    
    This custom implementation avoids high-level framework wrappers to maintain 
    granular control over the vector space optimization.
    """
    dot_product = torch.dot(vector_a, vector_b)
    norm_a = torch.norm(vector_a)
    norm_b = torch.norm(vector_b)
    
    # Handle potential zero-division vectors to ensure system stability
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return (dot_product / (norm_a * norm_b)).item()

def l2_normalize(vector: torch.Tensor) -> torch.Tensor:
    """
    Applies L2 Normalization to project a vector onto the unit hypersphere.
    
    After normalization, Cosine Similarity is equivalent to Euclidean dot product,
    which improves convergence speed in vector space search. This is the mathematical
    foundation behind FAISS's IndexFlatIP (Inner Product) index optimization.
    """
    norm = torch.norm(vector, p=2)
    if norm == 0:
        return vector
    return vector / norm


def batch_cosine_similarity(query: torch.Tensor, corpus: torch.Tensor) -> torch.Tensor:
    """
    Computes Cosine Similarity between one query vector and a corpus matrix
    using L2-normalized dot products — optimized for batch retrieval.
    
    Args:
        query:  shape (d,)
        corpus: shape (n, d)
    Returns:
        scores: shape (n,) — similarity score for each document
    """
    query_norm = l2_normalize(query)
    corpus_norm = torch.nn.functional.normalize(corpus, p=2, dim=1)
    return torch.mv(corpus_norm, query_norm)