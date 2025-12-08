import numpy as np
import gymnasium as gym
from gymnasium import spaces
STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

AGENT = 0
TRAIN = 1
RAILROADS = 2
OBSTACLES = 3
TREES = 4
STONE = 5
WOOD = 6
METAL = 7
STATION = 8
PICKAXE = 9
AXE = 10

class UnrailedEnv(gym.Env):

    def __init__(self, reward_config=None, width=10, height=6, config=1):
        self.rewards = reward_config
        self.config = config

        self.width = width
        self.height = height
        self.channels = 11 # agent, train, railroads, obstacles, resources, stations, items
        
        # stay, up, down, left, right
        self.action_space = spaces.Discrete(5) 
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.height, self.width, self.channels), dtype=np.int32)
        
        self.grid = None
        self.agent_position = None
        self.goal_position = (2, 8)
        
        self.inventory = {
            'wood': 0,
            'stone': 0,
            'metal': 0,
            'pickaxe': False,
            'axe': False,
        }

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.config == 1:
            pass
        
        self.grid = np.zeros((self.height, self.width, self.channels), dtype=np.int32)

        # hard code map for now
        # walls channel 3
        self.grid[0, :, OBSTACLES] = 1  # top Row
        self.grid[-1, :, OBSTACLES] = 1 # bottom Row
        self.grid[:, 0, OBSTACLES] = 1  # left Col
        self.grid[:, -1, OBSTACLES] = 1 # right Col

        # goal c6
        self.grid[2, 8, STATION] = 1 

        # train c1
        self.grid[2, 1, TRAIN] = 1 

        #events 
        # events = {
        #     "c"
        # }

        # resources c4
        trees = [(1, 4), (3, 4), (4, 6), (1, 7)] 
        for r, c in trees:
            self.grid[r, c, 4] = 1 

        # agent c0
        self.agent_position = [1, 1]
        self.grid[self.agent_position[0], self.agent_position[1], 0] = 1
        
        return self.grid, {}

    def chop_tree(self, r, c):
        if self.grid[r, c, TREES] == 1:
            self.grid[r, c, TREES] = 0
            return True
        return False

    def step(self, action):
        r, c = self.agent_position

        if action == UP:
            r -= 1
        elif action == DOWN:
            r += 1
        elif action == LEFT:
            c -= 1
        elif action == RIGHT:
            c += 1
        elif action == STAY:
            pass

        if 0 <= r < self.height and 0 <= c < self.width:
            if self.grid[r, c, 3] == 0: #checking for wall
                self.grid[self.agent_position[0], self.agent_position[1], 0] = 0
                self.agent_position = [r, c]
                self.grid[r, c, 0] = 1 

        reward = -0.01 # living penalty, maybe remove later, we will see
        done = False
        truncated = False

        if self.grid[self.agent_position[0], self.agent_position[1], 6] == 1: #c6 goal
            reward += 100
            done = True

        return self.grid, reward, done, truncated, {}

    def render(self):
        print("-" * self.width)
        for r in range(self.height):
            line = ""
            for c in range(self.width):
                char = '.'
                if self.grid[r, c, OBSTACLES] == 1: char = '#' 
                elif self.grid[r, c, STATION] == 1: char = 'S' 
                elif self.grid[r, c, TRAIN] == 1: char = 'R' 
                elif self.grid[r, c, TREES] == 1: char = 'T' 
                elif self.grid[r, c, AGENT] == 1: char = 'A' 
                line += char
            print(line)
        print("-" * self.width)