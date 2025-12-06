from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

class GridInvExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        grid_shape = observation_space["grid"].shape  # (C, H, W)
        inv_dim = observation_space["inv"].shape[0]

        C, H, W = grid_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # compute flat CNN size
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            cnn_out_dim = self.cnn(dummy).shape[1]

        self.inv_net = nn.Sequential(
            nn.Linear(inv_dim, 64),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        grid = obs["grid"]
        grid = grid.transpose(2, 0, 1)   # (H,W,C) → (C,H,W)
        inv = obs["inv"]

        cnn_out = self.cnn(grid)
        inv_out = self.inv_net(inv)
        cat = torch.cat([cnn_out, inv_out], dim=1)
        return self.fc(cat)
