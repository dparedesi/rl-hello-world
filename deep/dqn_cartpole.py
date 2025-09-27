import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size=4, action_size=2, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.gamma = 0.95
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural network
        self.q_network = DQN(state_size, 24, action_size).to(self.device)
        self.target_network = DQN(state_size, 24, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.update_target_network()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def visualize_dqn(agent, episodes=5):
    """Visualize the trained DQN agent playing CartPole"""
    env = gym.make('CartPole-v1', render_mode='human')
    episode_rewards = []
    
    print("\n🎮 Visualizing trained DQN agent...")
    print("="*50)
    
    # Store original epsilon and set to 0 for visualization
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during visualization
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        episode_rewards.append(total_reward)
        print(f"  Episode {episode + 1}: Reward = {total_reward}")
    
    env.close()
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    print("="*50)
    print(f"📊 Performance Summary:")
    print(f"  Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Best Episode:   {max(episode_rewards)}")
    print(f"  Worst Episode:  {min(episode_rewards)}")
    
    return episode_rewards

# Train DQN
def train_dqn(episodes=1000):
    env = gym.make('CartPole-v1')
    agent = DQNAgent()
    scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        score = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Modify reward for better learning
            if done and score < 195:
                reward = -10
                
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break
            
            if len(agent.memory) > 32:
                agent.replay(32)
        
        scores.append(score)
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
        
        if episode % 20 == 0:
            print(f"Episode {episode}, Score: {score:.0f}, Avg Score: {np.mean(scores[-100:]):.2f}, Epsilon: {agent.epsilon:.3f}")
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.5, label='Episode Score')
    plt.plot(np.convolve(scores, np.ones(50)/50, mode='valid'), label='50-episode average', linewidth=2)
    plt.axhline(y=195, color='r', linestyle='--', label='Solved threshold')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(scores[-100:], label='Last 100 episodes')
    plt.axhline(y=195, color='r', linestyle='--', label='Solved threshold')
    plt.xlabel('Episode (Last 100)')
    plt.ylabel('Score')
    plt.title('Recent Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🏆 Final Performance:")
    print(f"  Last 100 episodes average: {np.mean(scores[-100:]):.2f}")
    print(f"  Best episode score: {max(scores)}")
    print(f"  Episodes to solve (>195): {next((i for i, score in enumerate(scores) if score >= 195), 'Not solved')}")
    
    return agent, scores

# Run training
print("🚀 Starting DQN training...")
trained_dqn, scores = train_dqn()

# Visualize the trained agent
visualize_dqn(trained_dqn, episodes=5)