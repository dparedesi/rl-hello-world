# Reinforcement Learning: Hello World 🤖

A comprehensive implementation of fundamental reinforcement learning algorithms applied to the CartPole environment. This repository demonstrates the progression from simple tabular methods to state-of-the-art deep reinforcement learning techniques.

## 🎯 Project Overview

This project implements and compares five different approaches to solving the CartPole-v1 environment:

1. **Random Agent** - Baseline performance
2. **Q-Learning** - Tabular reinforcement learning
3. **Deep Q-Network (DQN)** - Deep value-based RL
4. **REINFORCE** - Basic policy gradient method
5. **Proximal Policy Optimization (PPO)** - State-of-the-art policy gradient

## 📊 Performance Summary

| Algorithm | Average Score | Peak Score | Stability | Sample Efficiency |
|-----------|---------------|------------|-----------|-------------------|
| Random Agent | ~23 | ~66 | High | N/A |
| Q-Learning | ~100 | ~150 | High | Good |
| DQN | ~200 | ~300 | Medium | Good |
| REINFORCE | ~300 | ~500 | Low | Poor |
| PPO | **~485** | **500** | **High** | Medium |

*CartPole-v1 is considered "solved" when achieving an average score of 195+ over 100 consecutive episodes. Maximum possible score is 500.*

## 🗂️ Project Structure

```
rl-hello-world/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── basic/                             # Simple RL implementations
│   ├── cartpole_hello_world.py       # Random agent baseline
│   └── q_learning_cartpole.py         # Tabular Q-learning
└── deep/                              # Deep RL implementations
    ├── dqn_cartpole.py               # Deep Q-Network
    ├── policy_gradient_reinforce.py  # REINFORCE algorithm
    └── policy_gradient_ppo.py        # Proximal Policy Optimization

```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dparedesi/rl-hello-world.git
cd rl-hello-world
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Algorithms

**Random Agent (Baseline):**
```bash
python3 basic/cartpole_hello_world.py
```

**Q-Learning:**
```bash
python3 basic/q_learning_cartpole.py
```

**Deep Q-Network:**
```bash
python3 deep/dqn_cartpole.py
```

**REINFORCE Policy Gradient:**
```bash
python3 deep/policy_gradient_reinforce.py
```

**PPO (Best Performance):**
```bash
python3 deep/policy_gradient_ppo.py
```

## 🧠 Algorithm Explanations

### 1. Random Agent (`basic/cartpole_hello_world.py`)
- **Purpose**: Baseline performance measurement
- **Method**: Randomly selects actions (left or right)
- **Performance**: ~23 average score
- **Key Learning**: Establishes minimum performance threshold

### 2. Q-Learning (`basic/q_learning_cartpole.py`)
- **Type**: Tabular value-based method
- **Key Features**:
  - Discretizes continuous state space into 20×20×20×20 grid
  - Uses epsilon-greedy exploration
  - Learns Q-values through temporal difference updates
- **Performance**: ~100 average score
- **Limitations**: Discretization loses information; limited by state representation

### 3. Deep Q-Network (`deep/dqn_cartpole.py`)
- **Type**: Deep value-based method
- **Key Features**:
  - Neural network approximates Q-values
  - Experience replay buffer for stable learning
  - Target network for stable updates
  - Handles continuous state space naturally
- **Performance**: ~200 average score
- **Advantages**: No discretization needed; better function approximation

### 4. REINFORCE (`deep/policy_gradient_reinforce.py`)
- **Type**: Basic policy gradient method
- **Key Features**:
  - Directly learns policy (action probabilities)
  - Uses complete episode returns for updates
  - Monte Carlo policy gradient
- **Performance**: ~300 average score
- **Characteristics**: High variance but can achieve high performance

### 5. PPO (`deep/policy_gradient_ppo.py`)
- **Type**: Advanced policy gradient method
- **Key Features**:
  - Actor-Critic architecture (policy + value function)
  - Clipped surrogate objective prevents large policy updates
  - Batch updates for stability
  - Generalized Advantage Estimation (GAE)
- **Performance**: ~485 average score (97% of theoretical maximum)
- **Advantages**: Most stable and highest performing

## 🔧 Technical Implementation Details

### Q-Learning Implementation
- **State Discretization**: Continuous states mapped to discrete grid
- **Action Selection**: Epsilon-greedy with decay (1.0 → 0.01)
- **Learning Rate**: 0.1
- **Discount Factor**: 0.99
- **Q-Table Size**: 160,000 entries (20⁴ × 2)

### DQN Implementation
- **Network Architecture**: 4 → 24 → 24 → 2 (fully connected)
- **Experience Replay**: Buffer size 2000
- **Target Network**: Updated every 10 episodes
- **Exploration**: Epsilon-greedy (1.0 → 0.01)
- **Batch Size**: 32

### REINFORCE Implementation
- **Network Architecture**: 4 → 16 → 2
- **Policy**: Stochastic (Categorical distribution)
- **Updates**: After each complete episode
- **Baseline**: None (high variance)
- **Learning Rate**: 0.002

### PPO Implementation
- **Network Architecture**: 4 → 64 → 64 → 2 (shared backbone)
- **Actor-Critic**: Separate heads for policy and value
- **Clipping Parameter**: 0.1 (conservative)
- **Batch Updates**: Every 10 episodes
- **GAE Lambda**: 0.95
- **Learning Rate**: 3e-4

## 📈 Key Insights and Lessons Learned

### 1. Algorithm Progression
The performance improvement follows a logical progression:
- **Tabular methods** are limited by state representation
- **Deep methods** handle continuous states better
- **Policy methods** can achieve higher performance than value methods
- **Advanced policy methods** (PPO) provide the best balance of performance and stability

### 2. Stability vs Performance Trade-offs
- **Q-Learning**: Stable but limited performance
- **DQN**: Good balance of stability and performance
- **REINFORCE**: High performance potential but unstable
- **PPO**: Excellent performance with high stability

### 3. Sample Efficiency
- **Q-Learning**: Most sample efficient
- **DQN**: Good sample efficiency with experience replay
- **REINFORCE**: Poor sample efficiency (high variance)
- **PPO**: Moderate sample efficiency but reliable

### 4. Implementation Complexity
The more sophisticated algorithms require more careful hyperparameter tuning and implementation details.

## 🎮 Environment Details

**CartPole-v1 Environment:**
- **Objective**: Balance a pole on a moving cart
- **Actions**: Push cart left (0) or right (1)
- **State Space**: 4 continuous values
  - Cart position: [-2.4, 2.4]
  - Cart velocity: [-∞, ∞]
  - Pole angle: [-0.2, 0.2] radians
  - Pole angular velocity: [-∞, ∞]
- **Reward**: +1 for each timestep the pole remains upright
- **Episode Termination**: 
  - Pole angle > 15°
  - Cart position > 2.4 units from center
  - Episode length > 500 timesteps
- **Success Criterion**: Average reward ≥ 195 over 100 consecutive episodes

## 📋 Requirements

- Python 3.9+
- gymnasium >= 0.29.0
- numpy >= 1.24.0
- torch >= 2.0.0
- matplotlib >= 3.7.0
- pygame >= 2.1.3 (for visualization)

See `requirements.txt` for complete dependency list.

## 🏆 Results and Visualizations

Each algorithm generates training curves showing:
- Episode scores over time
- Moving averages for trend analysis
- Performance distribution histograms
- Comparison with "solved" threshold (195)

The PPO implementation achieves near-theoretical maximum performance (~485/500 average), demonstrating the effectiveness of modern deep reinforcement learning techniques.

## 🔬 Future Extensions

Potential improvements and extensions:
- **Multi-environment testing**: Apply algorithms to other environments
- **Hyperparameter optimization**: Systematic tuning for optimal performance
- **Advanced algorithms**: A2C, SAC, TD3 implementations
- **Comparative analysis**: Statistical significance testing
- **Continuous control**: Extend to continuous action spaces

## 📚 References and Learning Resources

This implementation serves as a practical introduction to reinforcement learning, progressing from basic concepts to state-of-the-art methods. It's designed for educational purposes and demonstrates the evolution of RL techniques.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for improvements or additional algorithms.

## 📄 License

This project is open source and available under the MIT License.
