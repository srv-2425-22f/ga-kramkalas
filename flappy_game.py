import pygame
import random
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font("arial.ttf", 25)

SPEED = 50
PLAYER_SIZE = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 50)

Point = namedtuple("Point", "x, y, h")

class FlappyGameAI:

    def __init__(self, width=600, height=400):
        self.width = width
        self.height = height
        # Initialize display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Bird
        self.bird = Point(2 * self.width / 3, self.height / 2, PLAYER_SIZE)
        self.gravity = 0.2
        self.velocity = 0

        # Pipes
        self.pipes = []
        self.pipe_distance = 160
        self.pipe_gap = 80
        self.pipe_width = 40
        self.pipe_velocity = -2
        for pipe in range(3):
            height = random.randint(self.pipe_gap, self.height - self.pipe_gap)
            pipe = Point(500 + pipe * (self.pipe_distance + self.pipe_width), 0, height)
            self.pipes.append(pipe)

        self.score = 0
        self.frame_iteration = 0

    def play_step(self, action):
        self.frame_iteration += 1
        # Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # # User input
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_SPACE and allow_jump:
            #         action = [1]
            #         allow_jump = False
            # if event.type == pygame.KEYUP:
            #     if event.key == pygame.K_SPACE:
            #         allow_jump = True
        
        # Move
        self._move_bird(action)
        self._move_pipes()

        # Check game over
        reward = 0
        game_over = False
        if self.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # Check if passed through pipe
        for pipe in self.pipes:
            if self.bird.x == pipe.x + self.pipe_width - self.pipe_velocity:
                self.score += 1
                reward = 10           
        
        # Update UI and clock
        self.clock.tick(SPEED)
        self._update_ui()

        # Return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.bird

        # Hits top or bottom
        if (pt.y < 0 or pt.y > self.height - PLAYER_SIZE):
            return True
        
        # Hits pipe
        for index, pipe in enumerate(self.pipes):          
            bottom_pipe = self.bottom_pipes[index]
            if pt.x + pt.h >= pipe.x and pt.x <= pipe.x:
                if pt.y <= pipe.h or pt.y+pt.h >= bottom_pipe.y:
                    return True
                
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        bird = self.bird
        pygame.draw.rect(
            self.display, YELLOW, pygame.Rect(bird.x, bird.y, bird.h, bird.h)
        )

        for index, pipe in enumerate(self.pipes):
            bottom_pipe = self.bottom_pipes[index]
            pygame.draw.rect(
                self.display, GREEN, pygame.Rect(pipe.x, pipe.y, pipe.h, pipe.h)
            )
            pygame.draw.rect(
                self.display, GREEN, pygame.Rect(bottom_pipe.x, bottom_pipe.y, bottom_pipe.h, bottom_pipe.h)
            )
        
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0,0])
        pygame.display.flip()

    def _move_bird(self, action):
        if np.array_equal(action, [1]):
            self.velocity = -6
        else:
            self.velocity -= self.gravity

        x = self.bird.x
        y = self.bird.y
        y += self.velocity

        self.bird = Point(x, y)

    def _move_pipes(self):
        self.bottom_pipes = []

        for pipe in self.pipes:
            # Check if pipe is off-screen
            if pipe.x < -self.pipe_width:
                height = random.randint(self.pipe_gap, self.height - self.pipe_gap)
                pipe = Point(640, 0, height)

            # Move pipe
            pipe.x += self.pipe_velocity
            
            # Add bottom pipe to list
            x = pipe.x
            y = pipe.h + self.pipe_gap
            h = self.height - y
            bottom_pipe = Point(x, y, h)
            self.bottom_pipes.append(bottom_pipe)