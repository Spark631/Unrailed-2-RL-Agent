import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from configs.ppo_config import PPOConfig
class GridInvExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        grid_space = observation_space.spaces['grid']
        n_input_channels = grid_space.shape[2]
        height = grid_space.shape[0]
        width = grid_space.shape[1]

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

        # inventory: 5, train_position: 2, agent_position: 2 = 9 total
        n_extra = 5 + 2 + 2
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + n_extra, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations) -> torch.Tensor:
        # observations is a Dict of Tensors
        grid = observations['grid'] # (Batch, Height, Width, Channels)
        inventory = observations['inventory'] # (Batch, 5)
        train_pos = observations['train_position'] # (Batch, 2)
        agent_pos = observations['agent_position'] # (Batch, 2)
        
        # Permute to (Batch, Channels, Height, Width) for pytorch
        grid = grid.permute(0, 3, 1, 2)
        
        # Process grid through CNN
        grid_features = self.cnn(grid.float())
        
        # Concatenate all features
        combined = torch.cat([
            grid_features,
            inventory.float(),
            train_pos.float(),
            agent_pos.float()
        ], dim=1)
        
        return self.linear(combined)
