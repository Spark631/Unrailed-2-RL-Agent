import numpy as np
STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
class UnrailedEnv():

    def __init__(self, grid_size=(4, 16), agent_position=(0, 0), goal_position=(3, 15), obstacles=[(1, 3), (2, 6), (3, 9)]):
        self.grid_size = grid_size
        self.agent_position = agent_position
        self.goal_position = goal_position  
        self.obstacles = obstacles  
        self.reset()
    
    def reset(self):
        self.agent_position = (0, 0)
        return self.agent_position

    def step(self, action):
        x, y = self.agent_position

        if action == UP and y < self.grid_size[1] - 1:
            y += 1
        elif action == DOWN and y > 0:
            y -= 1
        elif action == LEFT and x > 0:
            x -= 1
        elif action == RIGHT and x < self.grid_size[0] - 1:
            x += 1
        elif action == STAY:
            pass

        new_position = (x, y)

        if new_position in self.obstacles:
            reward = -10
            done = True
        elif new_position == self.goal_position:
            reward = 100
            done = True
        else:
            reward = -1
            done = False

        self.agent_position = new_position
        return new_position, reward, done, {}

    def render(self):
        x, y = self.grid_size
        grid = [['.' for _ in range(x)] for _ in range(y)]

        for ox, oy in self.obstacles:
            grid[oy][ox] = 'X'

        ax, ay = self.agent_position
        gx, gy = self.goal_position
        grid[ay][ax] = 'A'
        grid[gy][gx] = 'G'

        for row in reversed(grid):
            print(' '.join(row))
        print()