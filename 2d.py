import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, size=10):
        self.size = size
        self.grid = np.zeros((size, size, 2))  # 2D grid with 2 resource types
        self.agent_pos = np.array([np.random.randint(0, size), np.random.randint(0, size)])
        self.place_resources()

    def place_resources(self):
        # Randomly place resources of type 1 and 2
        for _ in range(5):  # Place 5 of each resource type
            self.grid[np.random.randint(0, self.size), np.random.randint(0, self.size), 0] = 1
            self.grid[np.random.randint(0, self.size), np.random.randint(0, self.size), 1] = 1

    def step(self, action):
        # Move agent based on action (0: up, 1: right, 2: down, 3: left)
        move = {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}[action]
        self.agent_pos = np.clip(self.agent_pos + move, 0, self.size - 1)
        
        # Collect resources at the new position
        resources = self.grid[tuple(self.agent_pos)]
        self.grid[tuple(self.agent_pos)] = [0, 0]  # Remove collected resources
        
        return self.agent_pos, resources

    def reset(self):
        self.agent_pos = np.array([np.random.randint(0, self.size), np.random.randint(0, self.size)])
        self.grid.fill(0)
        self.place_resources()
        return self.agent_pos

class ActiveInferenceAgent:
    def __init__(self, env_size, learning_rate=0.1):
        self.env_size = env_size
        self.learning_rate = learning_rate
        self.beliefs = np.ones((env_size, env_size, 2)) / (env_size * env_size)  # Beliefs for each resource
        self.transition_model = np.ones((env_size, env_size, 4, env_size, env_size)) / (env_size * env_size)
        self.resource_levels = np.zeros(2)

    def update_beliefs(self, observation, resources):
        # Update beliefs based on observation and collected resources
        self.beliefs[tuple(observation)] = resources
        self.beliefs /= np.sum(self.beliefs)

    def act(self):
        expected_free_energy = np.zeros(4)  # Up, Right, Down, Left
        current_pos = tuple(self.agent_pos)
        
        for action in range(4):
            for next_x in range(self.env_size):
                for next_y in range(self.env_size):
                    state_probability = self.transition_model[next_x, next_y, action, current_pos[0], current_pos[1]]
                    expected_resources = self.beliefs[next_x, next_y]
                    
                    # Calculate expected free energy components
                    information_gain = -np.sum(state_probability * np.log(state_probability + 1e-12))
                    expected_reward = np.sum(expected_resources * (1 - self.resource_levels))
                    
                    expected_free_energy[action] += information_gain + expected_reward
        
        return np.argmin(expected_free_energy)

    def update_model(self, prev_pos, action, new_pos):
        # Update transition model based on observed state transition
        self.transition_model[new_pos[0], new_pos[1], action, prev_pos[0], prev_pos[1]] += self.learning_rate
        self.transition_model[:, :, action, prev_pos[0], prev_pos[1]] /= np.sum(self.transition_model[:, :, action, prev_pos[0], prev_pos[1]])

    def update_resources(self, collected_resources):
        self.resource_levels += collected_resources
        self.resource_levels = np.clip(self.resource_levels, 0, 1)  # Ensure resources are between 0 and 1

def run_simulation(env_size=10, n_steps=100):
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

    return np.array(states), np.array(beliefs), np.array(resource_levels), env.grid

def visualize_simulation(states, beliefs, resource_levels, final_grid):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    # Plot agent's position over time
    axes[0, 0].plot(states[:, 0], states[:, 1], 'b-')
    axes[0, 0].plot(states[0, 0], states[0, 1], 'go', label='Start')
    axes[0, 0].plot(states[-1, 0], states[-1, 1], 'ro', label='End')
    axes[0, 0].imshow(np.sum(final_grid, axis=2), cmap='Greys', alpha=0.5)
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
    
    plt.tight_layout()
    plt.show()

# Run the simulation and visualize the results
env_size = 10
n_steps = 1000
states, beliefs, resource_levels, final_grid = run_simulation(env_size, n_steps)
visualize_simulation(states, beliefs, resource_levels, final_grid)