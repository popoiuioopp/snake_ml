import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np 
from collections import deque
import os
from game import Point, SnakeGame
import matplotlib.pyplot as plt
from IPython import display

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 #learning rate

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

class DQNSnake(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))

class QTrainer:
    def __init__(self, model, lr, gamma) -> None:
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) #optimizer
        self.criterion = nn.MSELoss() #loss function

    def train_step(self, state, action, reward, next_state, done): #trainer
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1: #if there 1 dimension
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)
        pred = self.model(state) 

        target = pred.clone() 
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

class DQNAgent:
    def __init__(self, load_model=False):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  
        self.model = DQNSnake(12, 4, 256) #input size, hidden size, output size
        if load_model:
            self.model.load()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: SnakeGame):
        head = game.snake_list[0]
        point_l = Point(head.x - SnakeGame.GRID_SIZE, head.y)
        point_r = Point(head.x + SnakeGame.GRID_SIZE, head.y)
        point_u = Point(head.x, head.y - SnakeGame.GRID_SIZE)
        point_d = Point(head.x, head.y + SnakeGame.GRID_SIZE)

        dir_l = game.direction == "LEFT"
        dir_r = game.direction == "RIGHT"
        dir_u = game.direction == "UP"
        dir_d = game.direction == "DOWN"

        state = [
            (dir_r and game.checkCollision(point_r)) or # Danger straight
            (dir_l and game.checkCollision(point_l)) or
            (dir_u and game.checkCollision(point_u)) or
            (dir_d and game.checkCollision(point_d)),

            (dir_u and game.checkCollision(point_r)) or # Danger right
            (dir_d and game.checkCollision(point_l)) or
            (dir_l and game.checkCollision(point_u)) or
            (dir_r and game.checkCollision(point_d)),

            (dir_d and game.checkCollision(point_r)) or # Danger left
            (dir_u and game.checkCollision(point_l)) or
            (dir_r and game.checkCollision(point_u)) or
            (dir_l and game.checkCollision(point_d)),

            (dir_d and game.checkCollision(point_u)) or # Danger behind
            (dir_u and game.checkCollision(point_d)) or # Should always be true
            (dir_r and game.checkCollision(point_l)) or
            (dir_l and game.checkCollision(point_r)),

            dir_l, #direction
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
            if len(self.memory) > BATCH_SIZE:
                mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
            else:
                mini_sample = self.memory

            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
            self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 1 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

if __name__ == "__main__":
    game = SnakeGame(speed=100)
    agent = DQNAgent(load_model=True)
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    plt.ion()
    while True:
        game.drawScreen()
        gameState = agent.get_state(game)
        move = agent.get_action(gameState)
        reward, done, score = game.playStep(game.decodeOneHotDir(move))
        newGameState = agent.get_state(game)

        print(gameState, game.decodeOneHotDir(move), reward, done, score)
        agent.train_short_memory(state=gameState, action=move, reward=reward, next_state=newGameState, done=done)
        agent.remember(state=gameState, action=move, reward=reward, next_state=newGameState, done=done)

        if done:
            agent.n_games += 1
            game.setGameStart()
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

