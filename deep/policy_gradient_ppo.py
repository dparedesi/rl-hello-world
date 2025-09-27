import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=2):  # More conservative network size
        super(PolicyNetwork, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, output_size)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
    def get_action_probs(self, x):
        x = self.forward(x)
        return F.softmax(self.actor(x), dim=-1)
    
    def get_value(self, x):
        x = self.forward(x)
        return self.critic(x)

class PPOAgent:
    def __init__(self, lr=3e-4):  # Conservative learning rate
        self.policy_network = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        
        # PPO hyperparameters - conservative for stability
        self.gamma = 0.99
        self.eps_clip = 0.1  # Tighter clipping for stability
        self.k_epochs = 4    # Fewer epochs to prevent overfitting
        self.gae_lambda = 0.95
        
        # Storage for batch training
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def select_action(self, state, training=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # Disable gradient computation during inference
        if not training:
            with torch.no_grad():
                probs = self.policy_network.get_action_probs(state)
                dist = Categorical(probs)
                action = dist.sample()
                return action.item()
        
        # Training mode
        probs = self.policy_network.get_action_probs(state)
        value = self.policy_network.get_value(state)
        
        distribution = Categorical(probs)
        action = distribution.sample()
        
        # Store for training
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(distribution.log_prob(action).detach())
        self.values.append(value)
        
        return action.item()
    
    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_returns_advantages(self):
        """Compute returns and advantages"""
        discounted_reward = 0
        returns = []
        
        # Calculate returns (backwards)
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        # Convert to tensors
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Calculate advantages
        if len(self.values) > 0:
            values = torch.cat(self.values).squeeze().detach()
            advantages = returns - values
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = returns
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns, advantages
    
    def update(self):
        """PPO update with clipping"""
        if len(self.states) == 0:
            return
            
        # Get returns and advantages
        returns, advantages = self.compute_returns_advantages()
        
        # Convert to tensors
        old_states = torch.cat(self.states)
        old_actions = torch.cat(self.actions)
        old_log_probs = torch.cat(self.log_probs)
        
        # Multiple update epochs
        for _ in range(self.k_epochs):
            # Get current predictions
            current_probs = self.policy_network.get_action_probs(old_states)
            current_values = self.policy_network.get_value(old_states).squeeze()
            
            dist = Categorical(current_probs)
            current_log_probs = dist.log_prob(old_actions)
            
            # Calculate ratio for PPO
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Calculate losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * F.mse_loss(current_values, returns)
            entropy_loss = -0.01 * dist.entropy().mean()  # Entropy bonus
            
            # Total loss
            loss = actor_loss + critic_loss + entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear memory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

def train_ppo(episodes=1000):
    env = gym.make('CartPole-v1')
    agent = PPOAgent()
    scores = []
    
    # Update every episode for CartPole (it's fast enough)
    update_timesteps = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_reward(reward, done)
            total_reward += reward
            update_timesteps += 1
            
            if done:
                break
                
            state = next_state
        
        scores.append(total_reward)
        
        # Update every 10 episodes (batch collection)
        if (episode + 1) % 10 == 0:
            agent.update()
        
        # Progress report
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")
    
    env.close()
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.5, label='Episode Score')
    if len(scores) >= 50:
        plt.plot(np.convolve(scores, np.ones(50)/50, mode='valid'), 
                 label='50-episode average', linewidth=2)
    plt.axhline(y=195, color='r', linestyle='--', label='Solved threshold')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('PPO Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    recent_scores = scores[-200:] if len(scores) > 200 else scores
    plt.plot(recent_scores, alpha=0.5)
    plt.axhline(y=195, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Episode (recent)')
    plt.ylabel('Score')
    plt.title('Recent Performance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Final Training Summary:")
    print(f"  Total Episodes: {len(scores)}")
    print(f"  Final Average (last 100): {np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores):.2f}")
    print(f"  Best Score: {max(scores):.0f}")
    print(f"  Solved: {'Yes ✅' if np.mean(scores[-100:] if len(scores) >= 100 else scores) >= 195 else 'No ❌'}")
    
    return agent

def visualize_ppo(agent, episodes=5):
    """Visualize trained PPO agent"""
    env = gym.make('CartPole-v1', render_mode='human')
    
    print("\n🎮 Visualizing PPO agent...")
    episode_scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            if done:
                break
        
        episode_scores.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward}")
    
    print(f"\nAverage Performance: {np.mean(episode_scores):.2f}")
    env.close()

# Train the agent
print("🚀 Starting PPO training...")
trained_agent = train_ppo(episodes=2000)

# Visualize
visualize_ppo(trained_agent, episodes=5)