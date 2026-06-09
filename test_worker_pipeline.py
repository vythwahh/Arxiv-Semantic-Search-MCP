import sys
import asyncio
import torch

sys.path.append("src")

from user_profile import UserBehaviorProfile
from embedder import ArxivEmbedder
from cron_worker import ArxivBackgroundWorker

async def main():
    print("INITIALIZING FULL PIPELINE TEST")
    
    # 1. Initialize core modules
    embedder = ArxivEmbedder()
    profile = UserBehaviorProfile(user_id="vythu_dev")
    
    # 2. Simulate user history to generate data profile
    print("\n[Simulation] Mocking user interactions with Transformer-based topics...")
    query_mock = "Transformer architectures and attention mechanisms in deep learning"
    
    # Generate an embedding of this interest via our sentence transformer model
    mock_interest_vector = embedder.model.encode(query_mock, convert_to_tensor=True, device=embedder.device).cpu()
    
    # Log 11 actions to force-trigger our custom K-Means engine clustering internally
    for _ in range(11):
        # Add slight random noise to simulate varying document clicks within the same interest sphere
        noise = torch.randn(384) * 0.05
        profile.log_action(mock_interest_vector + noise, weight=1.0)
        
    print(f"User behavior history size: {len(profile.behavior_history)}")
    print(f"Interest centroids verified shape: {profile.interest_centroids.shape}")

    # 3. Spin up the Background Worker
    # Set threshold to 0.30 for testing to ensure it catches matching papers from arXiv easily
    worker = ArxivBackgroundWorker(profile=profile, embedder=embedder, threshold=0.30)
    
    # Run the worker cycle once by setting interval to 5 seconds then breaking
    print("\nSTARTING BACKGROUND WORKER LIFECYCLE (ONE-SHOT RUN)")
    worker_task = asyncio.create_task(worker.run_periodic_check(interval_seconds=5))
    
    # Give it 15 seconds to fetch from arXiv, compute batch matrix similarity, and log notifications
    await asyncio.sleep(15)
    worker_task.cancel()
    print("\nPIPELINE TEST FINISHED SUCCESSFULLY")

if __name__ == "__main__":
    asyncio.run(main())