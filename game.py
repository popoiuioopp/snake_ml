from collections import namedtuple
import pygame
import random

Point = namedtuple('Point', 'x, y')

class SnakeGame:
    GRID_SIZE = 10

    def __init__(self, width=800, height=800, speed=15) -> None:
        pygame.init()
        self.width = width
        self.height = height
        self.speed = speed

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

    def gameLoop(self):
        """Main game loop."""
        while True:
            self.setGameStart()
            while not self.game_over:
                self.handleEvents()
                self.updateGame()
                self.drawScreen()
            self.drawGameOverScreen()
            self.handleGameOver()

    def handleEvents(self):
        """Handles user inputs."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(1)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.direction != "DOWN":
                    self.changeDirection("UP")
                    return
                if event.key == pygame.K_DOWN and self.direction != "UP":
                    self.changeDirection("DOWN")
                    return
                if event.key == pygame.K_LEFT and self.direction != "RIGHT":
                    self.changeDirection("LEFT")
                    return
                if event.key == pygame.K_RIGHT and self.direction != "LEFT":
                    self.changeDirection("RIGHT")
                    return

    def updateGame(self):
        """Updates the game state."""
        self.clock.tick(self.speed)
        self.walk(self.direction)
        self.checkEat()
        self.game_over = self.checkCollision(self.head)

    def drawScreen(self):
        """Draws the game screen."""
        self.display.fill(self.black)
        self.drawFood()
        self.drawSnake()
        self.drawScore()
        pygame.display.update()

    def drawSnake(self):
        """Draws the snake."""
        for segment in self.snake_list:
            pygame.draw.rect(self.display, self.green, [segment.x, segment.y, self.GRID_SIZE, self.GRID_SIZE])

    def walk(self, direction):
        """Moves the snake in the given direction."""
        newPosX = self.head.x
        newPosY = self.head.y

        if direction == "UP":
            newPosY -= self.GRID_SIZE
        elif direction == "DOWN":
            newPosY += self.GRID_SIZE
        elif direction == "RIGHT":
            newPosX += self.GRID_SIZE
        elif direction == "LEFT":
            newPosX -= self.GRID_SIZE

        new_head = Point(newPosX, newPosY)
        self.snake_list.insert(0, new_head)
        if not self.full:
            self.snake_list.pop()
        else: 
            self.full = False
        self.head = self.snake_list[0]

    def changeDirection(self, direction):
        """Changes the direction of the snake."""
        self.direction = direction

    def drawGameOverScreen(self):
        """Displays the game over screen."""
        self.display.fill(self.red)
        self.message("Game Over", self.black)
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
                        return
                    if event.key == pygame.K_q:
                        pygame.quit()
                        exit(1)

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
        self.display.fill(self.black)
        self.head = Point(self.width / 2, self.height / 2)
        self.foodExists = False
        self.available_points  = set(Point(x, y) for x in range(0, self.width, self.GRID_SIZE) for y in range(0, self.height, self.GRID_SIZE))
        self.score = 0
        self.curMove = 0
        self.full = False
        self.snake_list = [self.head]
        self.spawnFood()

    def spawnFood(self):
        """Spawns food at a random location."""
        if self.foodExists:
            return
        self.available_points = self.available_points - set(self.snake_list)
        pointList = list(self.available_points)

        self.food = random.choice(pointList)
        self.foodExists = True
    
    def drawFood(self):
        """Draws the food on the screen."""
        pygame.draw.rect(self.display, self.blue, [self.food.x, self.food.y, self.GRID_SIZE, self.GRID_SIZE])

    def checkEat(self) -> bool:
        """Checks if the snake has eaten the food."""
        if self.head == self.food:
            self.foodExists = False
            self.full = True
            self.score += 1
            self.curMove = 0
            self.spawnFood()
            return True

    def checkCollisionWithSelf(self):
        """Checks if the snake has collided with itself."""
        for segment in self.snake_list[1:]:
            if self.snake_list[0] == segment:
                self.game_over = True

    def checkOutOfBound(self):
        """Checks if the snake has gone out of bounds."""
        if self.head.x >= self.width or self.head.y >= self.height or self.head.x < 0 or self.head.y < 0:
            self.game_over = True

    def checkCollision(self, pt: Point):
        if pt.x >= self.width or pt.y >= self.height or pt.x < 0 or pt.y < 0:
            return True

        for segment in self.snake_list[1:]:
            if pt == segment:
                return True
        return False

    def decodeOneHotDir(self, dir: list[int]) -> str:
        if dir == [1, 0, 0, 0]:
            return "LEFT"
        if dir == [0, 1, 0, 0]:
            return "RIGHT"
        if dir == [0, 0, 1, 0]:
            return "UP"
        if dir == [0, 0, 0, 1]:
            return "DOWN"

    def playStep(self, dir: str) -> tuple[int, bool, int]:
        self.walk(dir)
        reward = 0

        if self.checkCollision(self.head) or self.curMove > 100:
            self.game_over = True
            reward = -10
            return reward, self.game_over, self.score

        if self.checkEat():
            reward = 10

        self.clock.tick(self.speed)

        return reward, self.game_over, self.score

if __name__ == '__main__':
    game = SnakeGame(400, 400)
    game.gameLoop()
