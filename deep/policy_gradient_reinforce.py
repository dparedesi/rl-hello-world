import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, output_size=2):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)  # Return action probabilities

class REINFORCEAgent:
    def __init__(self, lr=0.001):
        self.policy_network = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = 0.99
        
        # Store episode history
        self.saved_log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy_network(state)
        distribution = Categorical(probs)
        action = distribution.sample()
        
        # Save log probability for training
        self.saved_log_probs.append(distribution.log_prob(action))
        return action.item()
    
    def update(self):
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate discounted returns (backwards)
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        # Normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Update network
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode history
        self.saved_log_probs = []
        self.rewards = []

def train_reinforce(episodes=1000):
    env = gym.make('CartPole-v1')
    agent = REINFORCEAgent()
    scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            total_reward += reward
            
            if terminated or truncated:
                break
                
            state = next_state
        
        # Update policy after episode completes
        agent.update()
        scores.append(total_reward)
        
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")
    
    env.close()
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(scores, alpha=0.5)
    plt.plot(np.convolve(scores, np.ones(50)/50, mode='valid'), label='50-episode average')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('REINFORCE Policy Gradient on CartPole')
    plt.legend()
    plt.show()
    
    return agent

def visualize_policy_gradient(agent, episodes=5):
    """Visualize trained policy gradient agent"""
    env = gym.make('CartPole-v1', render_mode='human')
    
    print("\n🎮 Visualizing Policy Gradient agent...")
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1}: Reward = {total_reward}")
    
    env.close()

# Train the agent
print("🚀 Starting Policy Gradient training...")
trained_agent = train_reinforce(episodes=1000)

# Visualize
visualize_policy_gradient(trained_agent, episodes=5)