import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import scipy.signal

class Environment:
    def __init__(self, size=10, obstacle_ratio=0.1, regeneration_rate=0.01):
        self.size = size
        self.grid = np.zeros((size, size, 4))  # 2D grid with 4 resource types: food, water, air, sleep
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
        for _ in range(5):  # Place 5 of each resource type
            for i in range(4):
                pos = self.get_random_free_position()
                self.grid[pos[0], pos[1], i] = np.random.uniform(0.5, 1.0)

    def step(self, action):
        # Move agent based on action (0: up, 1: right, 2: down, 3: left, 4: sleep)
        if action == 4:  # Sleep action
            resources = np.zeros(4)
            resources[3] = min(1.0, self.grid[tuple(self.agent_pos)][3] + 0.2)  # Gain sleep resource
            self.grid[tuple(self.agent_pos)][3] = 0  # Remove sleep resource from grid
        else:
            move = {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}[action]
            new_pos = np.clip(self.agent_pos + move, 0, self.size - 1)
            
            if self.obstacles[tuple(new_pos)] == 0:
                self.agent_pos = new_pos
            
            # Collect resources at the new position
            resources = self.grid[tuple(self.agent_pos)].copy()
            self.grid[tuple(self.agent_pos)] = [0, 0, 0, 0]  # Remove collected resources
        
        # Air is always available
        resources[2] = 1.0
        
        # Regenerate resources
        self.regenerate_resources()
        
        return self.agent_pos, resources

    def regenerate_resources(self):
        regeneration = np.random.rand(self.size, self.size, 4) < self.regeneration_rate
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
        self.beliefs = np.random.uniform(0.1, 0.9, (env_size, env_size, 4))
        self.beliefs /= np.sum(self.beliefs)
        self.transition_model = np.ones((env_size, env_size, 5, env_size, env_size)) / (env_size * env_size)
        self.resource_levels = np.zeros(4)
        self.policy = np.ones(5) / 5  # Uniform initial policy (including sleep action)

    def update_beliefs(self, observation, resources):
        # Update beliefs for the current position
        self.beliefs[tuple(observation)] = resources

        # Diffuse beliefs to neighboring cells
        kernel = np.array([[0.05, 0.1, 0.05],
                           [0.1,  0.4, 0.1],
                           [0.05, 0.1, 0.05]])
        
        for r in range(4):  # For each resource type
            self.beliefs[:,:,r] = scipy.signal.convolve2d(self.beliefs[:,:,r], kernel, mode='same', boundary='wrap')

        # Normalize beliefs
        self.beliefs /= np.sum(self.beliefs)

        # Add some uncertainty
        self.beliefs = 0.95 * self.beliefs + 0.05 / (self.env_size * self.env_size * 4)

    def act(self):
        expected_free_energy = np.zeros(5)  # Up, Right, Down, Left, Sleep
        current_pos = tuple(self.agent_pos)
        
        for action in range(5):
            if action == 4:  # Sleep action
                next_pos = current_pos
                expected_resources = np.zeros(4)
                expected_resources[3] = min(1.0, self.beliefs[current_pos][3] + 0.2)
            else:
                move = {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}[action]
                next_pos = np.clip(current_pos[0] + move[0], 0, self.env_size-1), np.clip(current_pos[1] + move[1], 0, self.env_size-1)
                expected_resources = self.beliefs[next_pos]
            
            # Calculate expected free energy components
            information_gain = -np.sum(expected_resources * np.log(expected_resources + 1e-12))
            expected_reward = np.sum(expected_resources * (1 - self.resource_levels))
            
            expected_free_energy[action] = information_gain + expected_reward
        
        # Update policy using softmax function
        self.policy = softmax(-expected_free_energy / self.temperature)
        
        # Choose action based on updated policy
        return np.random.choice(5, p=self.policy)

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
        self.resource_levels = 0.95 * self.resource_levels + collected_resources
        self.resource_levels = np.clip(self.resource_levels, 0, 1)

def run_simulation(env_size=10, n_steps=500):
    env = Environment(env_size)
    agent = ActiveInferenceAgent(env_size)
    
    states = []
    beliefs = []
    resource_levels = []
    agent_actions = []

    state = env.reset()
    agent.agent_pos = state
    for _ in range(n_steps):
        states.append(state)
        beliefs.append(agent.beliefs.copy())
        resource_levels.append(agent.resource_levels.copy())

        action = agent.act()
        agent_actions.append(action)
        prev_state = state
        state, resources = env.step(action)
        agent.agent_pos = state
        agent.update_beliefs(state, resources)
        agent.update_model(prev_state, action, state)
        agent.update_resources(resources)

    return np.array(states), np.array(beliefs), np.array(resource_levels), env.grid, env.obstacles, agent_actions

def visualize_simulation(states, beliefs, resource_levels, final_grid, obstacles, agent_actions):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    
    # Plot agent's position over time
    axes[0, 0].plot(states[:, 0], states[:, 1], 'b-', alpha=0.5)
    axes[0, 0].plot(states[0, 0], states[0, 1], 'go', label='Start')
    axes[0, 0].plot(states[-1, 0], states[-1, 1], 'ro', label='End')
    axes[0, 0].imshow(obstacles, cmap='binary', alpha=0.3)
    axes[0, 0].imshow(np.sum(final_grid, axis=2), cmap='YlOrRd', alpha=0.5)
    axes[0, 0].set_title('Agent Movement')
    axes[0, 0].legend()
    
    # Plot resource levels over time
    axes[0, 1].plot(resource_levels[:, 0], label='Food')
    axes[0, 1].plot(resource_levels[:, 1], label='Water')
    axes[0, 1].plot(resource_levels[:, 2], label='Air')
    axes[0, 1].plot(resource_levels[:, 3], label='Sleep')
    axes[0, 1].set_title('Resource Levels')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Resource Level')
    axes[0, 1].legend()
    
    # Plot final resource distribution
    im3 = axes[0, 2].imshow(np.sum(final_grid, axis=2), cmap='YlOrRd')
    axes[0, 2].set_title('Final Resource Distribution')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Plot final belief distributions for each resource
    resource_names = ['Food', 'Water', 'Air', 'Sleep']
    for i in range(4):
        ax = axes[1 + i // 2, i % 2]
        im = ax.imshow(beliefs[-1, :, :, i], cmap='viridis')
        ax.set_title(f'Final Belief Distribution ({resource_names[i]})')
        plt.colorbar(im, ax=ax)
    
    # Plot obstacles
    im4 = axes[2, 2].imshow(obstacles, cmap='binary')
    axes[2, 2].set_title('Obstacles')
    plt.colorbar(im4, ax=axes[2, 2])
    
    # Plot agent's action distribution over time
    action_names = ['Up', 'Right', 'Down', 'Left', 'Sleep']
    action_counts = np.array([agent_actions.count(i) for i in range(5)])
    axes[1, 2].bar(action_names, action_counts)
    axes[1, 2].set_title('Agent Action Distribution')
    axes[1, 2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

# Run the simulation and visualize the results
env_size = 15
n_steps = 500
states, beliefs, resource_levels, final_grid, obstacles, agent_actions = run_simulation(env_size, n_steps)
visualize_simulation(states, beliefs, resource_levels, final_grid, obstacles, agent_actions)