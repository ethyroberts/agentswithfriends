import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, size):
        self.size = size
        self.goal = np.random.randint(0, size)  # Random goal position
        self.state = np.random.randint(0, size)  # Random initial state

    def step(self, action):
        # Update state based on action, keeping it within bounds
        self.state = max(0, min(self.size - 1, self.state + action))
        reward = 1 if self.state == self.goal else 0
        return self.state, reward

    def reset(self):
        self.state = np.random.randint(0, self.size)
        return self.state

class ActiveInferenceAgent:
    def __init__(self, env_size, learning_rate=0.1):
        self.env_size = env_size
        self.learning_rate = learning_rate
        self.beliefs = np.ones(env_size) / env_size  # Initialize uniform beliefs
        self.transition_model = np.eye(env_size)  # Initialize identity transition model

    def update_beliefs(self, observation):
        # Update beliefs based on observation and transition model
        self.beliefs = self.beliefs * (self.transition_model[observation, :] + 1e-12)
        self.beliefs /= np.sum(self.beliefs)  # Normalize beliefs

    def act(self):
        # Compute expected free energy for each action
        expected_free_energy = np.zeros(3)  # Left, Stay, Right
        for action in [-1, 0, 1]:
            for next_state in range(self.env_size):
                state_probability = self.transition_model[next_state, :]
                # Calculate negative entropy (information gain)
                expected_free_energy[action + 1] += -np.sum(state_probability * np.log(state_probability + 1e-12))
        
        # Choose action with minimum expected free energy
        return np.argmin(expected_free_energy) - 1

    def update_model(self, prev_state, action, new_state):
        # Update transition model based on observed state transition
        self.transition_model[new_state, prev_state] += self.learning_rate
        self.transition_model[new_state, :] /= np.sum(self.transition_model[new_state, :])

def run_simulation(env_size=10, n_steps=100):
    env = Environment(env_size)
    agent = ActiveInferenceAgent(env_size)
    
    states = []
    beliefs = []

    state = env.reset()
    for _ in range(n_steps):
        states.append(state)
        beliefs.append(agent.beliefs.copy())

        agent.update_beliefs(state)
        action = agent.act()
        prev_state = state
        state, _ = env.step(action)
        agent.update_model(prev_state, action, state)

    return np.array(states), np.array(beliefs), env.goal

def visualize_simulation(states, beliefs, goal, env_size):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot agent's position over time
    ax1.plot(states)
    ax1.axhline(y=goal, color='r', linestyle='--')
    ax1.set_ylabel('State')
    ax1.set_title('Agent Movement')
    
    # Plot belief distribution over time
    im = ax2.imshow(beliefs.T, aspect='auto', cmap='viridis')
    ax2.set_ylabel('State')
    ax2.set_xlabel('Time Step')
    ax2.set_title('Belief Distribution')
    
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.show()

# Run the simulation and visualize the results
env_size = 10
n_steps = 100
states, beliefs, goal = run_simulation(env_size, n_steps)
visualize_simulation(states, beliefs, goal, env_size)