import torch
from typing import List
from dataclasses import dataclass, field
from kmeans import CustomKMeans
from vector_ops import l2_normalize

@dataclass
class UserBehaviorProfile:
    user_id: str
    gamma: float = 0.9  # Time-decay factor to prioritize recent interests
    
    # Historical registry storing all logged behavior vectors (384-dim)
    behavior_history: List[torch.Tensor] = field(default_factory=list)
    
    # Global representation vector used during Cold Start phase
    global_profile_tensor: torch.Tensor = None
    
    # Discovered multi-interest centroids computed via K-Means
    interest_centroids: torch.Tensor = None

    def log_action(self, action_vector: torch.Tensor, weight: float = 1.0):
        """
        Logs a new user interaction signal (Search = 0.2, Click = 0.5, RAG = 1.0).
        """
        # 1. Project input action vector onto the unit hypersphere
        v_action = l2_normalize(action_vector.flatten().float()) * weight
        self.behavior_history.append(v_action)

        # 2. Update the global profile using Exponential Moving Average (EMA)
        if self.global_profile_tensor is None:
            self.global_profile_tensor = v_action.clone()
        else:
            self.global_profile_tensor = self.gamma * self.global_profile_tensor + (1 - self.gamma) * v_action
            self.global_profile_tensor = l2_normalize(self.global_profile_tensor)

        # 3. Trigger K-Means clustering once the history threshold (>= 10) is met
        if len(self.behavior_history) >= 10:
            self._update_interest_clusters()

    def _update_interest_clusters(self, k: int = 3):
        """
        Partitions historical interaction vectors into K distinct core interest centroids.
        """
        # Stack the historical list into a dense matrix representation (n_samples, 384)
        X = torch.stack(self.behavior_history)
        
        # Invoke the custom K-Means engine
        kmeans = CustomKMeans(n_clusters=k, max_iter=50)
        kmeans.fit(X)
        
        # Store computed centroids representing the user's primary latent topics
        self.interest_centroids = kmeans.centroids