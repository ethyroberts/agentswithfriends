import numpy as np

class Environment:
    def __init__(self, size=10, obstacle_ratio=0.1, regeneration_rate=0.01):
        self.size = size
        self.grid = np.zeros((size, size, 2))  # 2D grid with 2 resource types
        self.obstacles = np.random.choice([0, 1], size=(size, size), p=[1-obstacle_ratio, obstacle_ratio])
        self.agent_pos = self.get_random_free_position()
        self.regeneration_rate = regeneration_rate
        self.place_resources()

    def get_random_free_position(self):
        while True:
            pos = np.array([np.random.randint(0, self.size), np.random.randint(0, self.size)])
            if self.obstacles[tuple(pos)] == 0:
                return pos

    def place_resources(self):
        for _ in range(10):  # Place 10 of each resource type
            pos = self.get_random_free_position()
            self.grid[pos[0], pos[1], 0] = np.random.uniform(0.0, 1.0)
            pos = self.get_random_free_position()
            self.grid[pos[0], pos[1], 1] = np.random.uniform(0.0, 1.0)

    def step(self, action):
        # Move agent based on action (0: up, 1: right, 2: down, 3: left)
        move = {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}[action]
        new_pos = np.clip(self.agent_pos + move, 0, self.size - 1)
        
        if self.obstacles[tuple(new_pos)] == 0:
            self.agent_pos = new_pos
        
        # Collect resources at the new position
        resources = self.grid[tuple(self.agent_pos)].copy()
        self.grid[tuple(self.agent_pos)] = [0, 0]  # Remove collected resources
        
        # Regenerate resources
        self.regenerate_resources()
        
        return self.agent_pos, resources

    def regenerate_resources(self):
        regeneration = np.random.rand(self.size, self.size, 2) < self.regeneration_rate
        self.grid[regeneration] += np.random.uniform(0, 0.1, size=self.grid[regeneration].shape)
        self.grid = np.clip(self.grid, 0, 1)

    def reset(self):
        self.agent_pos = self.get_random_free_position()
        self.grid.fill(0)
        self.place_resources()
        return self.agent_pos