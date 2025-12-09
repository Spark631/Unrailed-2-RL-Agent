import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from utils.map_gen import generate_map

STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
INTERACT = 5

# Channel indices (Must match map_gen.py)
AGENT          = 0
TRAIN_STORAGE  = 1
TRAIN_CRAFTER  = 2
TRAIN_HEAD     = 3
RAILROADS      = 4
OBSTACLES      = 5
TREES          = 6
STONE          = 7
WOOD           = 8
METAL          = 9
TRACK_ITEM     = 10
STATION        = 11
PICKAXE        = 12
AXE            = 13

from utils.reward import compute_reward
from configs.ppo_config import MOVEMENT_REWARDS, GATHERING_REWARDS, FULL_GAME_REWARDS

class UnrailedEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, reward_config=None, width=16, height=4, config=1, max_steps=50, render_mode=None):
        self.render_mode = render_mode
        self.config = config
        self.max_steps = max_steps
        
        if config == 1:
            base_rewards = MOVEMENT_REWARDS
        elif config == 2:
            base_rewards = GATHERING_REWARDS
        else:
            base_rewards = FULL_GAME_REWARDS
            
        if reward_config:
            self.rewards = base_rewards.copy()
            self.rewards.update(reward_config)
        else:
            self.rewards = base_rewards

        self.width = width
        self.height = height
        self.channels = 14 
        
        # stay, up, down, left, right, interact
        self.action_space = spaces.Discrete(6) 
        
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(self.height, self.width, self.channels), dtype=np.int32),
            "inventory": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int32),  # wood, metal, rail, axe, pickaxe
            "train_position": spaces.Box(low=0, high=max(height, width), shape=(2,), dtype=np.int32),
            "agent_position": spaces.Box(low=0, high=max(height, width), shape=(2,), dtype=np.int32),
        })        

        # STATE
        
        self.grid = None
        self.agent_position = None
        self.train_position = None
        
        self.inventory = {'held_item': None}
        
        self.crafting = {
            'in_progress': False,
            'progress': 0,
            'time_to_craft': 5,
            'materials': {'wood': 0, 'metal': 0}
        }
        
        self.gathering = {
            'in_progress': False,
            'type': None,
            'target': None,
            'progress': 0,
            'time_required': 10
        }

        self.remaining_trees = 0
        self.remaining_rocks = 0
        
        # osculation tracker
        self.pos_history = deque(maxlen=5)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.inventory = {'held_item': None}
        self.crafting = {
            'in_progress': False,
            'progress': 0,
            'time_to_craft': 5,
            'materials': {'wood': 0, 'metal': 0}
        }
        
        self.gathering = {
            'in_progress': False,
            'type': None,
            'target': None,
            'progress': 0,
            'time_required': 5
        }

        map_config = {"seed": seed}
        
        if self.config == 1:
            map_config["p_obstacle"] = 0.0
            map_config["p_tree"] = 0.0
            map_config["p_rock"] = 0.0
            
        self.grid, agent_pos, train_pos, station_pos = generate_map(map_config)
        
        self.height, self.width, _ = self.grid.shape
        
        self.agent_position = list(agent_pos)
        self.train_position = list(train_pos)
        
        self.pos_history.clear()
        self.pos_history.append(tuple(self.agent_position))
        
        # default station position from map_gen
        self.station_position = station_pos

        # config 1 override: Randomize station position to prevent overfitting
        if self.config == 1:
            # clear existing station
            self.grid[:, :, STATION] = 0
            
            # place new random station
            while True:
                r = self.np_random.integers(0, self.height)
                c = self.np_random.integers(0, self.width)
                
                # ensure not on top of agent or train
                is_agent = (r == self.agent_position[0] and c == self.agent_position[1])
                is_train = (self.grid[r, c, TRAIN_HEAD] == 1 or 
                            self.grid[r, c, TRAIN_STORAGE] == 1 or 
                            self.grid[r, c, TRAIN_CRAFTER] == 1)
                
                if not is_agent and not is_train:
                    self.grid[r, c, STATION] = 1
                    self.station_position = (r, c)
                    break

        # TREE AND ROCK COUNT        
        if self.config != 1:
            self.remaining_trees = int(self.grid[:, :, TREES].sum())
            self.remaining_rocks = int(self.grid[:, :, STONE].sum())
        else:
            self.remaining_trees = 0
            self.remaining_rocks = 0

        return self._get_observation(), {}

    def get_adjacent_cells(self, r, c):
        """Returns list of adjacent cell coordinates (up, down, left, right)"""
        adjacent = []
        if r - 1 >= 0:
            adjacent.append((r - 1, c))  # up
        if r + 1 < self.height:
            adjacent.append((r + 1, c))  # down
        if c - 1 >= 0:
            adjacent.append((r, c - 1))  # left
        if c + 1 < self.width:
            adjacent.append((r, c + 1))  # right
        return adjacent

    def interact(self):
        """handle interaction with adjacent objects (chop, mine, pickup, drop, craft)"""
        events = []
        r, c = self.agent_position
        
        # pickup/drop
        if self.inventory['held_item'] is None:
            # pickup
            if self.grid[r, c, WOOD] == 1:
                self.inventory['held_item'] = 'wood'
                self.grid[r, c, WOOD] = 0
                return ['pickup']
            elif self.grid[r, c, METAL] == 1:
                self.inventory['held_item'] = 'metal'
                self.grid[r, c, METAL] = 0
                return ['pickup']
            elif self.grid[r, c, TRACK_ITEM] == 1:
                self.inventory['held_item'] = 'rail'
                self.grid[r, c, TRACK_ITEM] = 0
                return ['pickup']
            elif self.grid[r, c, AXE] == 1:
                self.inventory['held_item'] = 'axe'
                self.grid[r, c, AXE] = 0
                return ['pickup']
            elif self.grid[r, c, PICKAXE] == 1:
                self.inventory['held_item'] = 'pickaxe'
                self.grid[r, c, PICKAXE] = 0
                return ['pickup']
            elif self.grid[r, c, RAILROADS] == 1: # Pickup placed track
                self.inventory['held_item'] = 'rail'
                self.grid[r, c, RAILROADS] = 0
                return ['pickup_rail']
        
        # interact with adjacent (chop, mine, craft)
        interacted = False
        for ar, ac in self.get_adjacent_cells(r, c):
            # crafting (Train - Crafter or Storage)
            if self.grid[ar, ac, TRAIN_CRAFTER] == 1 or self.grid[ar, ac, TRAIN_STORAGE] == 1:
                if self.inventory['held_item'] in ['wood', 'metal']:
                    item = self.inventory['held_item']
                    self.crafting['materials'][item] += 1
                    self.inventory['held_item'] = None
                    
                    # check start crafting
                    if self.crafting['materials']['wood'] >= 1 and self.crafting['materials']['metal'] >= 1:
                        self.crafting['materials']['wood'] -= 1
                        self.crafting['materials']['metal'] -= 1
                        self.crafting['in_progress'] = True
                        self.crafting['progress'] = 0
                    return ['deposit']

            # chopping
            # Removed manual interaction for chopping

            # mining
            # Removed manual interaction for mining
        
        if interacted:
            return events

        # drop (if no interaction happened)
        if self.inventory['held_item'] is not None:
            # check if cell is empty of items
            has_item = (self.grid[r, c, WOOD] or self.grid[r, c, METAL] or 
                        self.grid[r, c, TRACK_ITEM] or self.grid[r, c, AXE] or 
                        self.grid[r, c, PICKAXE] or self.grid[r, c, RAILROADS])
            
            if not has_item:
                item = self.inventory['held_item']
                if item == 'wood': self.grid[r, c, WOOD] = 1
                elif item == 'metal': self.grid[r, c, METAL] = 1
                elif item == 'rail': self.grid[r, c, RAILROADS] = 1
                elif item == 'axe': self.grid[r, c, AXE] = 1
                elif item == 'pickaxe': self.grid[r, c, PICKAXE] = 1
                
                self.inventory['held_item'] = None
                if item == 'rail': return ['place_rail']
                return ['drop']
                    
        return []

    def _get_observation(self):
        inv_vec = np.zeros(5, dtype=np.int32)
        item = self.inventory['held_item']
        if item == 'wood': inv_vec[0] = 1
        elif item == 'metal': inv_vec[1] = 1
        elif item == 'rail': inv_vec[2] = 1
        elif item == 'axe': inv_vec[3] = 1
        elif item == 'pickaxe': inv_vec[4] = 1

        return {
            "grid": self.grid.copy(),
            "inventory": inv_vec,
            "train_position": np.array(self.train_position, dtype=np.int32),
            "agent_position": np.array(self.agent_position, dtype=np.int32),
        }

    def step(self, action):
        self.current_step += 1
        r, c = self.agent_position
        events = []

        if self.crafting['in_progress']:
            self.crafting['progress'] += 1
            if self.crafting['progress'] >= self.crafting['time_to_craft']:
                self.crafting['in_progress'] = False
                tr, tc = self.train_position
                for ar, ac in self.get_adjacent_cells(tr, tc):
                     if self.grid[ar, ac, OBSTACLES] == 0 and self.grid[ar, ac, TRACK_ITEM] == 0:
                         self.grid[ar, ac, TRACK_ITEM] = 1
                         events.append('craft_complete')
                         break

        if action == UP:
            r -= 1
        elif action == DOWN:
            r += 1
        elif action == LEFT:
            c -= 1
        elif action == RIGHT:
            c += 1
        elif action == STAY:
            events.append('action_stay')
        elif action == INTERACT:
            events.append('action_interact')
            interaction_events = self.interact()
            events.extend(interaction_events)

        # storing old position for distance comparison
        old_pos = tuple(self.agent_position)
        
        if 0 <= r < self.height and 0 <= c < self.width:
            is_obstacle = (self.grid[r, c, OBSTACLES] == 1)
            is_train = (self.grid[r, c, TRAIN_HEAD] == 1 or 
                        self.grid[r, c, TRAIN_STORAGE] == 1 or 
                        self.grid[r, c, TRAIN_CRAFTER] == 1)
            
            if not is_obstacle and not is_train:
                self.grid[self.agent_position[0], self.agent_position[1], AGENT] = 0
                self.agent_position = [r, c]
                self.grid[r, c, AGENT] = 1 
            else:
                events.append('wall_bump')
        else:
            events.append('wall_bump')

        new_pos = tuple(self.agent_position)

        # Auto-Gathering Logic (After movement)
        # Only gather if we are standing still (STAY)
        if action == STAY:
            # 1. Check if current gathering is still valid
            if self.gathering['in_progress']:
                gr, gc = self.gathering['target']
                # Check conditions: Adjacent + Tool
                # Use new agent position
                nr, nc = self.agent_position
                is_adjacent = (abs(nr - gr) + abs(nc - gc) == 1)
                
                has_tool = False
                if self.gathering['type'] == 'chop' and self.inventory['held_item'] == 'axe':
                    has_tool = True
                elif self.gathering['type'] == 'mine' and self.inventory['held_item'] == 'pickaxe':
                    has_tool = True
                    
                if not is_adjacent or not has_tool:
                    # Reset if moved away or lost tool
                    self.gathering['in_progress'] = False
                    self.gathering['progress'] = 0
                    self.gathering['target'] = None
                    self.gathering['type'] = None

            # 2. If not gathering (or just reset), look for new target
            if not self.gathering['in_progress']:
                nr, nc = self.agent_position
                for ar, ac in self.get_adjacent_cells(nr, nc):
                    # Check for trees
                    if self.grid[ar, ac, TREES] == 1 and self.inventory['held_item'] == 'axe':
                        self.gathering['in_progress'] = True
                        self.gathering['type'] = 'chop'
                        self.gathering['target'] = (ar, ac)
                        self.gathering['progress'] = 0
                        break # Only gather one thing at a time
                    
                    # Check for rocks
                    if self.grid[ar, ac, STONE] == 1 and self.inventory['held_item'] == 'pickaxe':
                        self.gathering['in_progress'] = True
                        self.gathering['type'] = 'mine'
                        self.gathering['target'] = (ar, ac)
                        self.gathering['progress'] = 0
                        break

            # 3. Increment progress if gathering
            if self.gathering['in_progress']:
                self.gathering['progress'] += 1
                if self.gathering['progress'] >= self.gathering['time_required']:
                    # Complete
                    gr, gc = self.gathering['target']
                    self.gathering['in_progress'] = False
                    self.gathering['target'] = None
                    self.grid[gr, gc, OBSTACLES] = 0
                    
                    if self.gathering['type'] == 'chop':
                        self.grid[gr, gc, TREES] = 0
                        self.grid[gr, gc, WOOD] = 1
                        self.remaining_trees -= 1
                        events.append('chop')
                    elif self.gathering['type'] == 'mine':
                        self.grid[gr, gc, STONE] = 0
                        self.grid[gr, gc, METAL] = 1
                        self.remaining_rocks -= 1
                        events.append('mine')
                    
                    self.gathering['type'] = None
        else:
            # If we moved or interacted, we stop gathering
            self.gathering['in_progress'] = False
            self.gathering['progress'] = 0
            self.gathering['target'] = None
            self.gathering['type'] = None

        delta = 0
        if new_pos != old_pos:
            prev_dist = self._get_bfs_distance(old_pos, self.station_position)
            new_dist = self._get_bfs_distance(new_pos, self.station_position)
            delta = prev_dist - new_dist


        done = False
        truncated = False

        if self.grid[self.agent_position[0], self.agent_position[1], STATION] == 1:  # reached goal
            events.append('goal_reached')
            # Only terminate on goal for configs 1 and 3 (not config 2 - gathering)
            if self.config in [1, 3]:
                done = True
            
        if self.current_step >= self.max_steps:
            truncated = True
            # events.append('timeout')
        
        reward = compute_reward(events, self.rewards, delta)

        return self._get_observation(), reward, done, truncated, {}
    
    def _get_bfs_distance(self, start, goal):
        q = [(start, 0)]
        visited = set([start])
        
        while q:
            (r, c), dist = q.pop(0)
            if (r, c) == goal:
                return dist
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if (nr, nc) not in visited:
                        is_obstacle = (self.grid[nr, nc, OBSTACLES] == 1)
                        is_train = (self.grid[nr, nc, TRAIN_HEAD] == 1 or 
                                    self.grid[nr, nc, TRAIN_STORAGE] == 1 or 
                                    self.grid[nr, nc, TRAIN_CRAFTER] == 1)
                        
                        if not is_obstacle and not is_train:
                            visited.add((nr, nc))
                            q.append(((nr, nc), dist + 1))
                            
        return 999 

    def render(self):
        print("-" * self.width)
        for r in range(self.height):
            line = ""
            for c in range(self.width):
                char = '.'
                if self.grid[r, c, AGENT] == 1: char = 'A'
                elif self.grid[r, c, TRAIN_HEAD] == 1: char = 'H'
                elif self.grid[r, c, TRAIN_CRAFTER] == 1: char = 'C'
                elif self.grid[r, c, TRAIN_STORAGE] == 1: char = 'S'
                elif self.grid[r, c, STATION] == 1: char = 'X'
                elif self.grid[r, c, OBSTACLES] == 1: char = '#'
                elif self.grid[r, c, TREES] == 1: char = 'T'
                elif self.grid[r, c, STONE] == 1: char = 'O'
                elif self.grid[r, c, WOOD] == 1: char = 'w'
                elif self.grid[r, c, METAL] == 1: char = 'm'
                elif self.grid[r, c, TRACK_ITEM] == 1: char = 'r'
                elif self.grid[r, c, RAILROADS] == 1: char = '='
                elif self.grid[r, c, AXE] == 1: char = 'x'
                elif self.grid[r, c, PICKAXE] == 1: char = 'p'
                line += char
            print(line)
        print("-" * self.width)
    