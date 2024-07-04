import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import scipy.signal

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
            self.grid[pos[0], pos[1], 0] = np.random.uniform(0.5, 1.0)
            pos = self.get_random_free_position()
            self.grid[pos[1], pos[1], 1] = np.random.uniform(0.5, 1.0)

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

class ActiveInferenceAgent:
    def __init__(self, env_size, learning_rate=0.1, temperature=0.5):
        self.env_size = env_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.beliefs = np.random.uniform(0.1, 0.9, (env_size, env_size, 2))
        self.beliefs /= np.sum(self.beliefs)
        self.transition_model = np.ones((env_size, env_size, 4, env_size, env_size)) / (env_size * env_size)
        self.resource_levels = np.zeros(2)
        self.policy = np.ones(4) / 4  # Uniform initial policy

    def update_beliefs(self, observation, resources):
        # Update beliefs for the current position
        self.beliefs[tuple(observation)] = resources

        # Diffuse beliefs to neighboring cells
        kernel = np.array([[0.05, 0.1, 0.05],
                           [0.1,  0.4, 0.1],
                           [0.05, 0.1, 0.05]])
        
        for r in range(2):  # For each resource type
            self.beliefs[:,:,r] = scipy.signal.convolve2d(self.beliefs[:,:,r], kernel, mode='same', boundary='wrap')

        # Normalize beliefs
        self.beliefs /= np.sum(self.beliefs)

        # Add some uncertainty
        self.beliefs = 0.95 * self.beliefs + 0.05 / (self.env_size * self.env_size * 2)

    def act(self):
        expected_free_energy = np.zeros(4)  # Up, Right, Down, Left
        current_pos = tuple(self.agent_pos)
        
        for action in range(4):
            move = {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}[action]
            next_pos = np.clip(current_pos[0] + move[0], 0, self.env_size-1), np.clip(current_pos[1] + move[1], 0, self.env_size-1)
            
            # Calculate expected free energy components
            expected_resources = self.beliefs[next_pos]
            information_gain = -np.sum(expected_resources * np.log(expected_resources + 1e-12))
            expected_reward = np.sum(expected_resources * (1 - self.resource_levels))
            
            expected_free_energy[action] = information_gain + expected_reward
        
        # Update policy using softmax function
        self.policy = softmax(-expected_free_energy / self.temperature)
        
        # Choose action based on updated policy
        return np.random.choice(4, p=self.policy)

    def update_model(self, prev_pos, action, new_pos):
        # Update transition model using exponential moving average
        update = np.zeros((self.env_size, self.env_size))
        update[new_pos[0], new_pos[1]] = 1
        self.transition_model[:, :, action, prev_pos[0], prev_pos[1]] = (
            (1 - self.learning_rate) * self.transition_model[:, :, action, prev_pos[0], prev_pos[1]] +
            self.learning_rate * update
        )

    def update_resources(self, collected_resources):
        # Update resource levels with decay
        self.resource_levels = 0.99 * self.resource_levels + collected_resources
        self.resource_levels = np.clip(self.resource_levels, 0, 1)

def run_simulation(env_size=10, n_steps=500):
    env = Environment(env_size)
    agent = ActiveInferenceAgent(env_size)
    
    states = []
    beliefs = []
    resource_levels = []

    state = env.reset()
    agent.agent_pos = state
    for _ in range(n_steps):
        states.append(state)
        beliefs.append(agent.beliefs.copy())
        resource_levels.append(agent.resource_levels.copy())

        action = agent.act()
        prev_state = state
        state, resources = env.step(action)
        agent.agent_pos = state
        agent.update_beliefs(state, resources)
        agent.update_model(prev_state, action, state)
        agent.update_resources(resources)

    return np.array(states), np.array(beliefs), np.array(resource_levels), env.grid, env.obstacles

def visualize_simulation(states, beliefs, resource_levels, final_grid, obstacles):
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    
    # Plot agent's position over time
    axes[0, 0].plot(states[:, 0], states[:, 1], 'b-', alpha=0.5)
    axes[0, 0].plot(states[0, 0], states[0, 1], 'go', label='Start')
    axes[0, 0].plot(states[-1, 0], states[-1, 1], 'ro', label='End')
    axes[0, 0].imshow(obstacles, cmap='binary', alpha=0.3)
    axes[0, 0].imshow(np.sum(final_grid, axis=2), cmap='YlOrRd', alpha=0.5)
    axes[0, 0].set_title('Agent Movement')
    axes[0, 0].legend()
    
    # Plot resource levels over time
    axes[0, 1].plot(resource_levels[:, 0], label='Resource 1')
    axes[0, 1].plot(resource_levels[:, 1], label='Resource 2')
    axes[0, 1].set_title('Resource Levels')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Resource Level')
    axes[0, 1].legend()
    
    # Plot final belief distribution for resource 1
    im1 = axes[1, 0].imshow(beliefs[-1, :, :, 0], cmap='viridis')
    axes[1, 0].set_title('Final Belief Distribution (Resource 1)')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Plot final belief distribution for resource 2
    im2 = axes[1, 1].imshow(beliefs[-1, :, :, 1], cmap='viridis')
    axes[1, 1].set_title('Final Belief Distribution (Resource 2)')
    plt.colorbar(im2, ax=axes[1, 1])
    
    # Plot final resource distribution
    im3 = axes[0, 2].imshow(np.sum(final_grid, axis=2), cmap='YlOrRd')
    axes[0, 2].set_title('Final Resource Distribution')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Plot obstacles
    im4 = axes[1, 2].imshow(obstacles, cmap='binary')
    axes[1, 2].set_title('Obstacles')
    plt.colorbar(im4, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.show()

# Run the simulation and visualize the results
env_size = 15
n_steps = 1000
states, beliefs, resource_levels, final_grid, obstacles = run_simulation(env_size, n_steps)
visualize_simulation(states, beliefs, resource_levels, final_grid, obstacles)