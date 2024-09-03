import numpy as np
from environ import Environment

class QLearningAgent:
    def __init__(self, action_space, state_space, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.action_space = action_space
        self.state_space = state_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros(state_space + (action_space,))

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_space)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state + (best_next_action,)]
        td_error = td_target - self.q_table[state + (action,)]
        self.q_table[state + (action,)] += self.learning_rate * td_error

def train_agent(env, agent, episodes=10000, max_steps=100):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action = agent.get_action(tuple(state))
            next_state, resources = env.step(action)
            reward = np.sum(resources)
            agent.update(tuple(state), action, reward, tuple(next_state))
            state = next_state
            total_reward += reward

        if episode % 1000 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    return agent

# Create the environment and agent
env = Environment()
agent = QLearningAgent(action_space=4, state_space=(env.size, env.size))

# Train the agent
trained_agent = train_agent(env, agent)

# Test the trained agent
test_episodes = 100
total_rewards = []
max_test_steps = 100

# Test loop
for _ in range(test_episodes):
    state = env.reset()
    episode_reward = 0

    for _ in range(max_test_steps):
        action = trained_agent.get_action(tuple(state))
        next_state, resources = env.step(action)
        reward = np.sum(resources)
        episode_reward += reward
        state = next_state

    total_rewards.append(episode_reward)

print(f"Average reward over {test_episodes} test episodes: {np.mean(total_rewards)}")