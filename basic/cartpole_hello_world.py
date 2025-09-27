import gymnasium as gym
import numpy as np

# Create the CartPole environment
env = gym.make('CartPole-v1', render_mode='human')

# Simple random agent to start
def random_agent(env, episodes=5):
    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Take random action (0=left, 1=right)
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    env.close()

# Run the random agent
random_agent(env)