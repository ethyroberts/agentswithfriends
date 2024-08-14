import numpy as np
import scipy.signal
from scipy.special import softmax

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