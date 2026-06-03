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