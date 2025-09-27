import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle  # Add this import for saving

class QLearningAgent:
    def __init__(self, n_states=20, n_actions=2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_states, n_states, n_states, n_actions))
        
        # Hyperparameters
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def discretize_state(self, state):
        """Convert continuous state to discrete"""
        bounds = [
            (-2.4, 2.4),    # Cart position
            (-3.0, 3.0),    # Cart velocity
            (-0.2, 0.2),    # Pole angle
            (-2.0, 2.0)     # Pole velocity
        ]
        
        discrete = []
        for i, s in enumerate(state):
            low, high = bounds[i]
            s = max(low, min(high, s))  # Clip to bounds
            discrete.append(int((s - low) / (high - low) * (self.n_states - 1)))
        return tuple(discrete)
    
    def choose_action(self, state, greedy=False):  # Add greedy parameter
        """Epsilon-greedy action selection"""
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            discrete_state = self.discretize_state(state)
            return np.argmax(self.q_table[discrete_state])
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update rule"""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        current_q = self.q_table[discrete_state][action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[discrete_next_state])
        
        # Q-learning update
        self.q_table[discrete_state][action] += self.learning_rate * (target_q - current_q)
        
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath='q_table.pkl'):  # Add save method
        """Save Q-table to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"✅ Q-table saved to {filepath}")
    
    def load(self, filepath='q_table.pkl'):  # Add load method
        """Load Q-table from file"""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"✅ Q-table loaded from {filepath}")

# Training the Q-learning agent
def train_q_learning(episodes=1000):
    env = gym.make('CartPole-v1')
    agent = QLearningAgent()
    rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        agent.decay_epsilon()
        rewards.append(total_reward)
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Avg Reward (last 50): {np.mean(rewards[-50:]):.2f}, Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    # Plot learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.5)
    plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), label='50-episode average')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning on CartPole')
    plt.legend()
    plt.show()
    
    return agent

# Add visualization function
def visualize_agent(agent, episodes=5):
    """Watch the trained agent play"""
    env = gym.make('CartPole-v1', render_mode='human')
    
    print("\n🎮 Visualizing trained agent...")
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state, greedy=True)  # Use greedy=True
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        print(f"Episode {episode + 1}: Reward = {total_reward}")
    
    env.close()

# Train the agent
trained_agent = train_q_learning()

# Save the trained agent
trained_agent.save('q_table.pkl')

# Visualize the trained agent
visualize_agent(trained_agent, episodes=10)