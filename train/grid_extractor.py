import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from configs.ppo_config import PPOConfig
class GridInvExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[2]
        height = observation_space.shape[0]
        width = observation_space.shape[1]

        self.cnn = nn.Sequential(
            #we pad so then the map size will stay the same
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            # to automatically determine output size of cnn instead of hardcoding
            dummy_input = torch.zeros(1, n_input_channels, height, width)
            n_flatten = self.cnn(dummy_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # obs = (Batch, Height, Width, Channels)
        # gotta switch to (Batch, Channels, Height, Width) for pytorch
        if observations.shape[-1] == 11: 
            observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations.float()))
