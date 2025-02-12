import pygame
import random

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# General game values
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
run = True
score = 0
x = 0
y = 1
width = 2
height = 3
font = pygame.font.Font("arial.ttf", 25)

# Initiate player
player_size = 50
allow_jump = True
velocity = 0
gravity = 0.2
player = pygame.Rect(300, 250, player_size, player_size)

# Initiate pipes
pipes = []
pipe_width = 80
pipe_gap = 200
pipe_distance = 400
number_of_pipes = 3
pipe_velocity = -2
for pipe in range(number_of_pipes):
    pipe_height = random.randint(100, 300)
    pipe = pygame.Rect(600+pipe*pipe_distance, 0, pipe_width, pipe_height)
    pipes.append(pipe)


while run:
    screen.fill((0))  

    jump = False       

    # Event handler
    for event in pygame.event.get():   
        if event.type == pygame.QUIT:
            run = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and allow_jump:
                jump = True
                allow_jump = False

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                allow_jump = True
        
    # Player
    if jump:
        velocity = -6    
    player.move_ip(0, velocity)
    velocity += gravity
    pygame.draw.rect(screen, (255, 255, 0), player)
    if player[y] < 0 or player[y]+player_size > SCREEN_HEIGHT:
        run = False

    # Pipe
    for pipe in pipes:
        if pipe[x] < -pipe_width:
            pipe_height = random.randint(100, 300)
            pipe[x] = 1100
            pipe[height] = pipe_height
        pipe.move_ip(pipe_velocity, 0)

        bottom_pipe = (pipe[x], pipe[height]+pipe_gap, pipe[width], SCREEN_HEIGHT-(pipe[height]+pipe_gap))
        pygame.draw.rect(screen, (0, 255, 0), pipe)
        pygame.draw.rect(screen, (0, 255, 0), bottom_pipe)    

        if player[x]  + player_size >= pipe[x] and player[x] <= pipe[x]:
            if player[y] <= pipe[height] or player[y]+player_size >= bottom_pipe[y]:
                run = False       

        if player[x] == pipe[0] + pipe_width - pipe_velocity:
            score += 1
            print(f"Score: {score}")

    # Update game
    text = font.render(f"Score: {score}", True, (255, 255, 255))
    textRect = text.get_rect()
    screen.blit(text, textRect)    
    clock.tick(FPS)

    pygame.display.update()

pygame.quit()