import sys
import os
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
# from stable_baselines3.common.monitor import Monitor

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.unrailed_env import UnrailedEnv
from grid_extractor import GridInvExtractor

print("CUDA available:", torch.cuda.is_available())

class SuccessRateCallback(BaseCallback):
    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_successes = []
        self.current_episode_reward = 0
        self.current_episode_success = False
        
    def _on_step(self) -> bool:
        reward = self.locals.get("rewards")[0]
        self.current_episode_reward += reward
        
        # Check info for success (if available) or fallback to reward threshold
        infos = self.locals.get("infos")
        if infos and len(infos) > 0:
            if infos[0].get("is_success", False):
                self.current_episode_success = True
        elif reward > 90: # Fallback for Phase 1
            self.current_episode_success = True

        if self.locals.get("dones")[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_successes.append(1 if self.current_episode_success else 0)
            
            self.current_episode_reward = 0
            self.current_episode_success = False
        
        if self.n_calls % self.check_freq == 0 and len(self.episode_successes) > 0:
            success_rate = np.mean(self.episode_successes[-100:]) * 100  
            mean_reward = np.mean(self.episode_rewards[-100:])
            
            self.logger.record("custom/success_rate", success_rate)
            self.logger.record("custom/mean_episode_reward", mean_reward)
            
            if self.verbose > 0:
                print(f"\nStep {self.n_calls}: Success Rate (last 100 eps): {success_rate:.1f}%, Mean Reward: {mean_reward:.2f}")
        
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
    
    env_gather = UnrailedEnv(config=2)
    
    # Load Phase 1 model to start
    if not os.path.exists("ppo_phase1_movement.zip"):
        print("Error: ppo_phase1_movement.zip not found. Please train Phase 1 first.")
        return

    print("Loading Phase 1 weights...")
    # Try to load existing Phase 2 model if it exists, else load Phase 1
    if os.path.exists("ppo_phase2_gathering.zip"):
        print("Resuming Phase 2 training...")
        model = PPO.load(
            "ppo_phase2_gathering", 
            env=env_gather, 
            tensorboard_log="./tb_logs",
            custom_objects={
                "features_extractor_class": GridInvExtractor,
                "features_extractor_kwargs": dict(features_dim=256),
            }
        )
    else:
        print("Starting Phase 2 from Phase 1 weights...")
        model = PPO.load(
            "ppo_phase1_movement", 
            env=env_gather, 
            tensorboard_log="./tb_logs",
            custom_objects={
                "features_extractor_class": GridInvExtractor,
                "features_extractor_kwargs": dict(features_dim=256),
            }
        )
    
    # Fine Tuning
    model.learning_rate = 0.00005  # Very low LR for stable fine-tuning
    model.ent_coef = 0.01          # Reduce exploration noise
    
    
    success_callback_p2 = SuccessRateCallback(check_freq=2048, verbose=1)
    
    model.learn(
        total_timesteps=500000,  # Increased to give time to learn gathering
        tb_log_name="PPO_Phase2_Gathering_3",
        callback=success_callback_p2,
        reset_num_timesteps=False # WE WANT SEPARATE TENSORBOARD GRAPHS
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
