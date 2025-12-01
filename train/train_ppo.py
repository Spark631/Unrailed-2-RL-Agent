import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from envs.unrailed_env import UnrailedEnv
from grid_inv_extractor import GridInvExtractor

env = UnrailedEnv()
check_env(env, warn=True)

policy_kwargs = dict(
    features_extractor_class=GridInvExtractor,
    features_extractor_kwargs=dict(features_dim=256),
)

model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./tb_logs",
)

model.learn(total_timesteps=10_000)
model.save("ppo_unrailed_v0")

