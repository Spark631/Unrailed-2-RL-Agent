import sys
import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Add the project root to the path so we can import envs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the train folder to the path so we can import grid_extractor directly
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))

from envs.unrailed_env import UnrailedEnv
# Import directly to match how it was saved
import grid_extractor 

def main():
    # Ask user which phase to watch
    print("Which phase do you want to watch?")
    print("1. Movement (Config 1)")
    print("2. Gathering (Config 2)")
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == '1':
        config = 1
        model_path = "ppo_phase1_movement.zip"
    else:
        config = 2
        model_path = "ppo_phase2_gathering.zip"

    # Ask for episode settings
    start_ep = 1
    try:
        val = input("Start from episode # (default 1): ").strip()
        if val: start_ep = int(val)
    except ValueError:
        pass
        
    num_eps = 5
    try:
        val = input("Number of episodes to watch (default 5): ").strip()
        if val: num_eps = int(val)
    except ValueError:
        pass

    # 1. Create the environment
    env = UnrailedEnv(config=config, render_mode='human')

    # 2. Load the trained model
    # We need to pass the custom objects so SB3 knows how to rebuild the CNN
    custom_objects = {
        "features_extractor_class": grid_extractor.GridInvExtractor,
        "features_extractor_kwargs": dict(features_dim=256),
    }
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, env=env, custom_objects=custom_objects)

    # 3. Run the agent
    action_names = {
        0: "STAY", 
        1: "UP", 
        2: "DOWN", 
        3: "LEFT", 
        4: "RIGHT",
        5: "INTERACT"
    }

    for ep in range(start_ep, start_ep + num_eps):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0
        step_count = 0
        max_steps = 1000  # Prevent infinite loops
        
        print(f"\n--- Episode {ep} ---")
        env.render()
        time.sleep(0.05)

        while not done and step_count < max_steps:
            step_count += 1
            
            # Predict the action
            action, _states = model.predict(obs, deterministic=True)
            
            # Take the action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Render
            os.system('cls' if os.name == 'nt' else 'clear') 
            print(f"Episode {ep} | Step: {step_count}/{max_steps}")
            print(f"Action: {action_names[int(action)]} | Reward: {reward:.2f} | Total Reward: {total_reward:.2f}")
            env.render()
            time.sleep(0.05)

        if step_count >= max_steps:
            print("Episode timed out!")
        else:
            print(f"Episode finished! Total Reward: {total_reward:.2f}")
        
        time.sleep(1)

if __name__ == "__main__":
    main()
