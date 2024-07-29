import pygame
import random
from agent import DQNAgent, ReplayBuffer
import matplotlib.pyplot as plt
import numpy as np

class SnakeGame:
    GRID_SIZE = 10

    def __init__(self, width=800, height=800, speed=15, move_limit=500) -> None:
        pygame.init()
        self.width = width
        self.height = height
        self.speed = speed
        self.move_limit = move_limit
        self.cur_move = 0
        self.game_over_reason = ""

        # Colors
        self.white = (255, 255, 255)
        self.yellow = (255, 255, 102)
        self.black = (0, 0, 0)
        self.red = (213, 50, 80)
        self.green = (0, 255, 0)
        self.blue = (50, 153, 213)

        # Fonts
        self.font_style = pygame.font.SysFont(None, 50)
        self.score_font = pygame.font.SysFont(None, 35)
        
        # Pygame display and clock
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake game")
        self.clock = pygame.time.Clock()

        self.setGameStart()

    def gameLoop(self, agent, train=False):
        """Main game loop."""
        self.setGameStart()
        total_reward = 0
        while not self.game_over:
            self.handleEvents()
            state = self.get_game_state()
            action = agent.act(state)
            self.direction = self.get_direction_from_action(action)
            self.updateGame()
            if self.cur_move > self.move_limit:
                self.game_over_reason = "Move limit exceeded"
                self.game_over = True
            next_state = self.get_game_state()
            reward = self.get_reward(state, next_state)
            total_reward += reward
            done = self.game_over
            agent.observe(state, reward, action, next_state, done)
            if train:
                agent.train()
            self.drawScreen()
            self.cur_move += 1
        print(f"Dying reason: {self.game_over_reason}")
        return total_reward

    def handleEvents(self):
        """Handles user inputs."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(1)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.direction != "DOWN":
                    self.changeDirection("UP")
                if event.key == pygame.K_DOWN and self.direction != "UP":
                    self.changeDirection("DOWN")
                if event.key == pygame.K_LEFT and self.direction != "RIGHT":
                    self.changeDirection("LEFT")
                if event.key == pygame.K_RIGHT and self.direction != "LEFT":
                    self.changeDirection("RIGHT")

    def updateGame(self):
        """Updates the game state."""
        self.clock.tick(self.speed)
        self.walk(self.direction)
        self.checkOutOfBound()
        self.checkEat()
        self.checkCollisionWithSelf()

    def drawScreen(self):
        """Draws the game screen."""
        self.display.fill(self.black)
        self.drawSnake()
        self.drawFood()
        self.drawScore()
        pygame.display.update()

    def drawSnake(self):
        """Draws the snake."""
        for segment in self.snake_list:
            pygame.draw.rect(self.display, self.green, [segment[0], segment[1], self.GRID_SIZE, self.GRID_SIZE])

    def walk(self, direction):
        """Moves the snake in the given direction."""
        if direction == "UP":
            self.headY -= self.GRID_SIZE
        elif direction == "DOWN":
            self.headY += self.GRID_SIZE
        elif direction == "RIGHT":
            self.headX += self.GRID_SIZE
        elif direction == "LEFT":
            self.headX -= self.GRID_SIZE

        self.snake_list.insert(0, (self.headX, self.headY))
        if not self.full:
            self.snake_list.pop()
        else: 
            self.full = False

    def changeDirection(self, direction):
        """Changes the direction of the snake."""
        self.direction = direction

    def checkOutOfBound(self):
        """Checks if the snake has gone out of bounds."""
        if self.headX >= self.width or self.headY >= self.height or self.headX < 0 or self.headY < 0:
            self.game_over_reason = "Hit the wall"
            self.game_over = True

    def drawGameOverScreen(self):
        """Displays the game over screen."""
        self.display.fill(self.red)
        self.message("Game Over: " + self.game_over_reason, self.black)
        pygame.display.update()

    def handleGameOver(self):
        """Handles user input on the game over screen."""
        while True:
            self.clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit(1)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        self.setGameStart()
                    if event.key == pygame.K_q:
                        pygame.quit()
                        exit(1)
            if not self.game_over:
                break

    def message(self, msg, color):
        """Displays a message on the screen."""
        mesg = self.font_style.render(msg, True, color)
        self.display.blit(mesg, [self.width / 6, self.height / 3])

    def drawScore(self):
        """Displays the current score on the screen."""
        value = self.score_font.render("Your Score: " + str(self.score), True, self.red)
        self.display.blit(value, [0, 0])

    def setGameStart(self):
        """Initializes the game state."""
        self.direction = "RIGHT"
        self.game_over = False
        self.game_over_reason = ""
        self.display.fill(self.black)
        self.headX = self.width / 2
        self.headY = self.height / 2
        self.foodExists = False
        self.spawnFood()
        self.score = 0
        self.full = False
        self.snake_list = [(self.headX, self.headY)]
        self.cur_move = 0

    def spawnFood(self):
        """Spawns food at a random location."""
        if self.foodExists:
            return
        self.foodX = random.randint(0, (self.width // self.GRID_SIZE) - 1) * self.GRID_SIZE
        self.foodY = random.randint(0, (self.height // self.GRID_SIZE) - 1) * self.GRID_SIZE 
        self.foodExists = True
    
    def drawFood(self):
        """Draws the food on the screen."""
        pygame.draw.rect(self.display, self.blue, [self.foodX, self.foodY, self.GRID_SIZE, self.GRID_SIZE])

    def checkEat(self):
        """Checks if the snake has eaten the food."""
        if (self.headX == self.foodX) and (self.headY == self.foodY):
            self.foodExists = False
            self.full = True
            self.score += 1
            self.cur_move = 0
            self.spawnFood()

    def checkCollisionWithSelf(self):
        """Checks if the snake has collided with itself."""
        for segment in self.snake_list[1:]:
            if (self.headX, self.headY) == segment:
                self.game_over_reason = "Hit itself"
                self.game_over = True

    def get_game_state(self):
        state = [
            self.headX, self.headY,  
            self.foodX, self.foodY,
            self.cur_move
        ]
        direction_one_hot = self.encode_direction(self.direction)
        state.extend(direction_one_hot)

        for segment in self.snake_list:
            state.extend(segment)

        max_segments = 50
        state.extend([0] * 2 * (max_segments - len(self.snake_list)))
        return state

    def get_reward(self, state, next_state):
        head_x, head_y = state[:2]
        next_head_x, next_head_y = next_state[:2]
        
        distance_to_food_before = self.get_distance_to_food(head_x, head_y)
        distance_to_food_after = self.get_distance_to_food(next_head_x, next_head_y)

        if self.game_over:
            if self.game_over_reason == "Move limit exceeded":
                return -200
            elif self.game_over_reason == "Hit itself":
                return -150  
            else:
                return -100 
        elif self.full:
            return 50 * len(self.snake_list)
        else:
            reward = 1
            if distance_to_food_after < distance_to_food_before:
                reward += 1
            else:
                reward -= 1 
            return reward
        
    def get_direction_from_action(self, action):
        actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        return actions[action]
    
    def encode_direction(self, direction):
        """Encodes the direction as a one-hot vector."""
        if direction == "UP":
            return [1, 0, 0, 0]
        elif direction == "DOWN":
            return [0, 1, 0, 0]
        elif direction == "LEFT":
            return [0, 0, 1, 0]
        elif direction == "RIGHT":
            return [0, 0, 0, 1]
    
    def get_distance_to_food(self, headX, headY):
        return abs(headX - self.foodX) + abs(headY - self.foodY)

if __name__ == '__main__':
    game = SnakeGame(400, 400, speed=60)
    input_dim = 2 + 2 + 4 + 1 + 2 * 50  # head (2) + food (2) + direction (4) + current move (1) + snake segments (2 * 50)
    output_dim = 4  # UP, DOWN, LEFT, RIGHT
    replay_buffer = ReplayBuffer(10000)
    agent = DQNAgent(input_dim, output_dim, replay_buffer)

    scores = []

    plt.ion()  
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlim(0, 1000)
    ax.set_ylim(-200, 20000)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Reward per Episode')

    for episode in range(1000):
        total_reward = game.gameLoop(agent, train=True)
        scores.append(total_reward)
        print(f"Episode {episode}: Total Reward: {total_reward}")
        line.set_data(np.arange(len(scores)), scores)
        ax.set_xlim(0, len(scores) + 1)  # Adjust the x-axis limit to the number of episodes
        ax.set_ylim(min(scores) - 10, max(scores) + 10)

        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.show()

    # Test the trained agent
    # game.setGameStart()
    # game.gameLoop(agent, train=False)
