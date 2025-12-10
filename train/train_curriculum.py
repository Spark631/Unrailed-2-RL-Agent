import sys
import os
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
# from stable_baselines3.common.monitor import Monitor

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.unrailed_env import UnrailedEnv
from grid_extractor import GridInvExtractor

print("CUDA available:", torch.cuda.is_available())

def make_env(config, rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = UnrailedEnv(config=config)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

class SuccessRateCallback(BaseCallback):
    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_successes = []
        
    def _on_step(self) -> bool:
        # Check for done episodes in all environments
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        
        if dones is not None and infos is not None:
            for i, done in enumerate(dones):
                if done:
                    info = infos[i]
                    is_success = info.get("is_success", False)
                    self.episode_successes.append(1 if is_success else 0)
        
        if self.n_calls % self.check_freq == 0 and len(self.episode_successes) > 0:
            success_rate = np.mean(self.episode_successes[-100:]) * 100  
            self.logger.record("custom/success_rate", success_rate)
        
        return True

def main():
    #  PHASE 1: MOVEMENT 
    # print("\n=== PHASE 1: LEARNING TO MOVE (Config 1) ===")
    # env_move = UnrailedEnv(config=1)
    
    # policy_kwargs = dict(
    #     features_extractor_class=GridInvExtractor,
    #     features_extractor_kwargs=dict(features_dim=256),
    # )

    # model = PPO(
    #     "MultiInputPolicy",
    #     env_move,
    #     policy_kwargs=policy_kwargs,
    #     verbose=1,
    #     tensorboard_log="./tb_logs",
    #     learning_rate=0.0003,
    #     ent_coef=0.05,        #for exploration
    #     n_steps=2048,
    #     batch_size=64,
    #     gamma=0.99,           # discount factor for long-term rewards
    #     gae_lambda=0.95,      # GAE parameter for advantage estimation
    #     n_epochs=10,          
    #     clip_range=0.2,       
    # )

    # success_callback = SuccessRateCallback(check_freq=2048, verbose=1)
    
    # # model = PPO.load("ppo_phase1_movement", env=env_move, tensorboard_log="./tb_logs")
    # model.learn(
    #     total_timesteps=500000, 
    #     tb_log_name="PPO_Phase1_Movement",
    #     callback=success_callback
    # )
    # model.save("ppo_phase1_movement")
    
    # if len(success_callback.episode_successes) > 0:
    #     final_success_rate = np.mean(success_callback.episode_successes[-100:]) * 100
    #     print(f"\n=== Phase 1 Training Complete ===")
    #     print(f"Final Success Rate (last 100 eps): {final_success_rate:.1f}%")
    #     print(f"Total Episodes: {len(success_callback.episode_successes)}")
    #     print(f"Model saved to: ppo_phase1_movement.zip")
    # else:
    #     print("Phase 1 Complete. Model saved.")
    
    # PHASE 2: GATHERING 
    print("\n=== PHASE 2: LEARNING TO GATHER (Config 2) ===")
    
    num_cpu = 8
    # Create the vectorized environment
    env_gather = SubprocVecEnv([make_env(config=2, rank=i) for i in range(num_cpu)])
    
    # Load Phase 1 model to start
    if not os.path.exists("ppo_phase1_movement.zip"):
        print("Error: ppo_phase1_movement.zip not found. Please train Phase 1 first.")
        return

    print("Loading weights...")
    # Resume from Phase 2 if available, otherwise start from Phase 1
    if os.path.exists("ppo_phase2_gathering.zip"):
        load_path = "ppo_phase2_gathering"
        print("Resuming Phase 2 training...")
    else:
        load_path = "ppo_phase1_movement"
        print("Starting Phase 2 from Phase 1 weights...")

    model = PPO.load(
        load_path, 
        env=env_gather, 
        tensorboard_log="./tb_logs",
        n_steps=512, # 512 * 8 = 4096 steps per update
        batch_size=128,
        custom_objects={
            "features_extractor_class": GridInvExtractor,
            "features_extractor_kwargs": dict(features_dim=256),
        }
    )
    
    # Fine Tuning
    model.learning_rate = 0.00005  # Very low LR for stable fine-tuning
    model.ent_coef = 0.01          # Reduce exploration noise
    
    
    success_callback_p2 = SuccessRateCallback(check_freq=500, verbose=1)
    
    model.learn(
        total_timesteps=2000000,  # Increased for vectorized training
        tb_log_name="PPO_Phase2_Gathering_Vec",
        callback=success_callback_p2,
        reset_num_timesteps=False 
    )
    model.save("ppo_phase2_gathering")
    
    # Print final statistics
    if len(success_callback_p2.episode_successes) > 0:
        final_success_rate = np.mean(success_callback_p2.episode_successes[-100:]) * 100
        print(f"\n=== Phase 2 Training Complete ===")
        print(f"Final Success Rate (last 100 eps): {final_success_rate:.1f}%")
        print(f"Total Episodes: {len(success_callback_p2.episode_successes)}")
        print(f"Model saved to: ppo_phase2_gathering.zip")
    else:
        print("Phase 2 Complete. Model saved.")
    
    return True

if __name__ == "__main__":
    main()
