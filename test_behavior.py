import sys
sys.path.append("src")

import torch
from user_profile import UserBehaviorProfile

# Initialize the profile registry for a mock user
user_test = UserBehaviorProfile(user_id="vythu_123")

print("--- PHASE 1: Testing Cold Start Operations (Actions < 10) ---")
for i in range(5):
    # Simulate random 384-dim dense interaction embedding vectors
    mock_vector = torch.randn(384)
    user_test.log_action(mock_vector, weight=0.5)

print(f"Logged Actions Counter: {len(user_test.behavior_history)}")
print(f"Global Profile Tensor Shape: {user_test.global_profile_tensor.shape}")
print(f"K-Means Clustering Status: {user_test.interest_centroids}")

print("\n--- PHASE 2: Testing Automated K-Means Trigger (Actions >= 10) ---")
for i in range(6):
    mock_vector = torch.randn(384)
    user_test.log_action(mock_vector, weight=1.0)


print(f"Final Total Actions: {len(user_test.behavior_history)}")
print(f"Final Centroids Shape: {user_test.interest_centroids.shape}")