import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np 
from collections import deque

class DQNSnake(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNSnake, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SimpleDQNAgent:
    def __init__(self, input_dim, output_dim):
        self.model = DQNSnake(input_dim, output_dim)
    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        # print("Q-values:", q_values)
        return q_values.argmax().item()
    
    def observe(self, state, reward):
        print(f"Observed state: {state}, reward: {reward}")
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, game_over):
        self.buffer.append((state, action, reward, next_state, game_over))

    def sample(self, batch_size):
        state, action, reward, next_state, game_over = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(game_over)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, replay_buffer: ReplayBuffer):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.batch_size = 64 # number of experiences each step
        self.gamma = 0.99 # immediate vs future rewards
        self.epsilon = 1.0 # exploration rate
        self.epsilon_decay = 0.995 # Decay rate of exploration rate (0.5% after each episode) 
        self.epsilon_min = 0.01  # Minimum epsilon 
        self.learning_rate = 0.001

        self.model = DQNSnake(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            q_values = self.model(state)
        # print("Q-values:", q_values)
        return q_values.argmax().item()
    
    def observe(self, state, reward, action, next_state, game_over):
        self.replay_buffer.push(state, action, reward, next_state, game_over)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, game_over = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        game_over = torch.FloatTensor(game_over)

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        target_q_values = q_values.clone()
        for i in range(self.batch_size):
            target_q_values[i, action[i]] = reward[i] + self.gamma * torch.max(next_q_values[i]) * (1 - game_over[i])
        
        loss = self.loss_fn(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    pass
