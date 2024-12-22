""" Original Flappy Bird game by `sourahbhv`.

Copy of the code in the "FlapPyBird" repository on GitHub
(https://github.com/sourabhv/FlapPyBird) by `sourahbhv`. Minor alterations were
made on the code in order to improve readability.
"""

from itertools import cycle
import random
import sys

import pygame
from pygame.locals import *
import numpy as np

ASSETS_DIR = "./adrian/lib/site-packages/flappy_bird_gymnasium/assets"

FPS = 30
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
PIPE_GAP_SIZE = 100  # gap between upper and lower part of pipe
BASE_Y = SCREEN_HEIGHT * 0.79

# image, sound and hit-mask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        ASSETS_DIR + '/sprites/redbird-upflap.png',
        ASSETS_DIR + '/sprites/redbird-midflap.png',
        ASSETS_DIR + '/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        ASSETS_DIR + '/sprites/bluebird-upflap.png',
        ASSETS_DIR + '/sprites/bluebird-midflap.png',
        ASSETS_DIR + '/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        ASSETS_DIR + '/sprites/yellowbird-upflap.png',
        ASSETS_DIR + '/sprites/yellowbird-midflap.png',
        ASSETS_DIR + '/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    ASSETS_DIR + '/sprites/background-day.png',
    ASSETS_DIR + '/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    ASSETS_DIR + '/sprites/pipe-green.png',
    ASSETS_DIR + '/sprites/pipe-red.png',
)

class FlappyBirdAI:

    def __init__(self):
        print("Init")
        global SCREEN, FPSCLOCK
        pygame.init()
        FPSCLOCK = pygame.time.Clock()
        SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Bird')

        # numbers sprites for score display
        IMAGES['numbers'] = (
            pygame.image.load(ASSETS_DIR + '/sprites/0.png').convert_alpha(),
            pygame.image.load(ASSETS_DIR + '/sprites/1.png').convert_alpha(),
            pygame.image.load(ASSETS_DIR + '/sprites/2.png').convert_alpha(),
            pygame.image.load(ASSETS_DIR + '/sprites/3.png').convert_alpha(),
            pygame.image.load(ASSETS_DIR + '/sprites/4.png').convert_alpha(),
            pygame.image.load(ASSETS_DIR + '/sprites/5.png').convert_alpha(),
            pygame.image.load(ASSETS_DIR + '/sprites/6.png').convert_alpha(),
            pygame.image.load(ASSETS_DIR + '/sprites/7.png').convert_alpha(),
            pygame.image.load(ASSETS_DIR + '/sprites/8.png').convert_alpha(),
            pygame.image.load(ASSETS_DIR + '/sprites/9.png').convert_alpha()
        )

        # game over sprite
        IMAGES['gameover'] = pygame.image.load(ASSETS_DIR + '/sprites/gameover.png').convert_alpha()
        # message sprite for welcome screen
        IMAGES['message'] = pygame.image.load(ASSETS_DIR + '/sprites/message.png').convert_alpha()
        # base (ground) sprite
        IMAGES['base'] = pygame.image.load(ASSETS_DIR + '/sprites/base.png').convert_alpha()

        # sounds
        if 'win' in sys.platform:
            soundExt = '.wav'
        else:
            soundExt = '.ogg'

        SOUNDS['die'] = pygame.mixer.Sound(ASSETS_DIR + '/audio/die' + soundExt)
        SOUNDS['hit'] = pygame.mixer.Sound(ASSETS_DIR + '/audio/hit' + soundExt)
        SOUNDS['point'] = pygame.mixer.Sound(ASSETS_DIR + '/audio/point' + soundExt)
        SOUNDS['swoosh'] = pygame.mixer.Sound(ASSETS_DIR + '/audio/swoosh' + soundExt)
        SOUNDS['wing'] = pygame.mixer.Sound(ASSETS_DIR + '/audio/wing' + soundExt)

        # while True:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipe_index = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.flip(
                pygame.image.load(PIPES_LIST[pipe_index]).convert_alpha(), False, True),
            pygame.image.load(PIPES_LIST[pipe_index]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            self.get_hitmask(IMAGES['pipe'][0]),
            self.get_hitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            self.get_hitmask(IMAGES['player'][0]),
            self.get_hitmask(IMAGES['player'][1]),
            self.get_hitmask(IMAGES['player'][2]),
        )

        self.movement_info = self.show_welcome_animation()

    def show_welcome_animation(self):
        """ Shows welcome screen animation of flappy bird. """
        # print("Welcome anim")
        # index of player to blit on screen
        self.player_index = 0
        self.player_index_gen = cycle([0, 1, 2, 1])
        # iterator used to change player_index after every 5th iteration
        self.loop_iter = 0

        self.player_x = int(SCREEN_WIDTH * 0.2)
        self.player_y = int((SCREEN_HEIGHT - IMAGES['player'][0].get_height()) / 2)

        self.message_x = int((SCREEN_WIDTH - IMAGES['message'].get_width()) / 2)
        self.message_y = int(SCREEN_HEIGHT * 0.12)

        self.base_x = 0
        # amount by which base can maximum shift to left
        self.base_shift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

        # player shm for up-down motion on welcome screen
        self.player_shm_vals = {'val': 0, 'dir': 1}

        while True:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                    # make first flap sound and return values for main_game
                    SOUNDS['wing'].play()
                    return {
                        'player_y': self.player_y + self.player_shm_vals['val'],
                        'base_x': self.base_x,
                        'player_index_gen': self.player_index_gen,
                    }

            # adjust player_y, player_index, base_x
            if (self.loop_iter + 1) % 5 == 0:
                self.player_index = next(self.player_index_gen)
            self.loop_iter = (self.loop_iter + 1) % 30
            self.base_x = -((-self.base_x + 4) % self.base_shift)
            self.playerShm(self.player_shm_vals)

            # draw sprites
            SCREEN.blit(IMAGES['background'], (0,0))
            SCREEN.blit(IMAGES['player'][self.player_index],
                        (self.player_x, self.player_y + self.player_shm_vals['val']))
            SCREEN.blit(IMAGES['message'], (self.message_x, self.message_y))
            SCREEN.blit(IMAGES['base'], (self.base_x, BASE_Y))

            pygame.display.update()
            FPSCLOCK.tick(FPS)

    def reset(self, movement_info):
        print("Reset")
        ###########  S E T U P   F O R   G A M E  ###########
        self.score = 0
        self.player_index = 0
        self.loop_iter = 0
        self.player_index_gen = movement_info['player_index_gen']
        self.player_x, self.player_y = int(SCREEN_WIDTH * 0.2), movement_info['player_y']

        self.base_x = movement_info['base_x']
        self.base_shift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

        # get 2 new pipes to add to upper_pipes lower_pipes list
        new_pipe1 = self.get_random_pipe()
        new_pipe2 = self.get_random_pipe()

        # list of upper pipes
        self.upper_pipes = [
            {'x': SCREEN_WIDTH + 200, 'y': new_pipe1[0]['y']},
            {'x': SCREEN_WIDTH + 200 + (SCREEN_WIDTH / 2), 'y': new_pipe2[0]['y']},
        ]

        # list of lower pipes
        self.lower_pipes = [
            {'x': SCREEN_WIDTH + 200, 'y': new_pipe1[1]['y']},
            {'x': SCREEN_WIDTH + 200 + (SCREEN_WIDTH / 2), 'y': new_pipe2[1]['y']},
        ]

        self.pipe_vel_x = -4

        # player velocity, max velocity, downward acceleration, acceleration on flap
        self.player_vel_y = -9  # player's velocity along Y, default same as player_flapped
        self.player_max_vel_y = 10   # max vel along Y, max descend speed
        self.player_min_vel_y = -8   # min vel along Y, max ascend speed
        self.player_acc_y = 1   # players downward accleration
        self.player_rot = 45   # player's rotation
        self.player_vel_rot = 3   # angular speed
        self.player_rot_thr = 20   # rotation threshold
        self.player_flap_acc = -9   # players speed on flapping
        self.player_flapped = False  # True when player flaps       

    def play_step(self, action):
            ###########  G A M E   L O O P  ###########
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()
            if action:
                if self.player_y > -2 * IMAGES['player'][0].get_height():
                    self.player_vel_y = self.player_flap_acc
                    self.player_flapped = True
                    SOUNDS['wing'].play()

            # check for crash here
            self.crash_test = self.check_crash({'x': self.player_x, 'y': self.player_y, 'index': self.player_index},
                                    self.upper_pipes, self.lower_pipes)
            # print("crash test:", self.crash_test)
            if self.crash_test[0]:
                return {
                    'y': self.player_y,
                    'groundCrash': self.crash_test[1],
                    'base_x': self.base_x,
                    'upper_pipes': self.upper_pipes,
                    'lower_pipes': self.lower_pipes,
                    'score': self.score,
                    'player_vel_y': self.player_vel_y,
                    'player_rot': self.player_rot
                }

            # check for score
            self.player_mid_pos = self.player_x + IMAGES['player'][0].get_width() / 2
            for pipe in self.upper_pipes:
                pipe_mid_pos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
                if pipe_mid_pos <= self.player_mid_pos < pipe_mid_pos + 4:
                    score += 1
                    SOUNDS['point'].play()

            # player_index base_x change
            if (self.loop_iter + 1) % 3 == 0:
                self.player_index = next(self.player_index_gen)
            self.loop_iter = (self.loop_iter + 1) % 30
            self.base_x = -((-self.base_x + 100) % self.base_shift)

            # rotate the player
            if self.player_rot > -90:
                self.player_rot -= self.player_vel_rot

            # player's movement
            if self.player_vel_y < self.player_max_vel_y and not self.player_flapped:
                self.player_vel_y += self.player_acc_y
            if self.player_flapped:
                self.player_flapped = False

                # more rotation to cover the threshold (calculated in visible rotation)
                self.player_rot = 45

            self.player_height = IMAGES['player'][self.player_index].get_height()
            self.player_y += min(self.player_vel_y, BASE_Y - self.player_y - self.player_height)

            # move pipes to left
            for up_pipe, low_pipe in zip(self.upper_pipes, self.lower_pipes):
                up_pipe['x'] += self.pipe_vel_x
                low_pipe['x'] += self.pipe_vel_x

            # add new pipe when first pipe is about to touch left of screen
            if len(self.upper_pipes) > 0 and 0 < self.upper_pipes[0]['x'] < 5:
                newPipe = self.get_random_pipe()
                self.upper_pipes.append(newPipe[0])
                self.lower_pipes.append(newPipe[1])

            # remove first pipe if its out of the screen
            if len(self.upper_pipes) > 0 and self.upper_pipes[0]['x'] < -IMAGES['pipe'][0].get_width():
                self.upper_pipes.pop(0)
                self.lower_pipes.pop(0)

            # draw sprites
            SCREEN.blit(IMAGES['background'], (0,0))

            for up_pipe, low_pipe in zip(self.upper_pipes, self.lower_pipes):
                SCREEN.blit(IMAGES['pipe'][0], (up_pipe['x'], up_pipe['y']))
                SCREEN.blit(IMAGES['pipe'][1], (low_pipe['x'], low_pipe['y']))

            SCREEN.blit(IMAGES['base'], (self.base_x, BASE_Y))
            # print score so player overlaps the score
            self.show_score(self.score)

            # Player rotation has a threshold
            self.visible_rot = self.player_rot_thr
            if self.player_rot <= self.player_rot_thr:
                self.visible_rot = self.player_rot
            
            self.player_surface = pygame.transform.rotate(IMAGES['player'][self.player_index], self.visible_rot)
            SCREEN.blit(self.player_surface, (self.player_x, self.player_y))

            pygame.display.update()
            FPSCLOCK.tick(FPS)

    def show_game_over_screen(self, crash_info):
        """crashes the player down ans shows game over image"""
        print("crash info 1: ", crash_info)
        self.score = crash_info['score']
        self.player_x = SCREEN_WIDTH * 0.2
        self.player_y = crash_info['y']
        self.player_height = IMAGES['player'][0].get_height()
        self.player_vel_y = crash_info['player_vel_y']
        self.player_acc_y = 2
        self.player_rot = crash_info['player_rot']
        self.player_vel_rot = 7

        self.base_x = crash_info['base_x']

        self.upper_pipes, self.lower_pipes = crash_info['upper_pipes'], crash_info['lower_pipes']

        # play hit and die sounds
        SOUNDS['hit'].play()
        if not crash_info['groundCrash']:
            SOUNDS['die'].play()

        while True:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                    if self.player_y + self.player_height >= BASE_Y - 1:
                        return

            # player y shift
            if self.player_y + self.player_height < BASE_Y - 1:
                self.player_y += min(self.player_vel_y, BASE_Y - self.player_y - self.player_height)

            # player velocity change
            if self.player_vel_y < 15:
                self.player_vel_y += self.player_acc_y

            # rotate only when it's a pipe crash
            if not crash_info['groundCrash']:
                if self.player_rot > -90:
                    self.player_rot -= self.player_vel_rot

            # draw sprites
            SCREEN.blit(IMAGES['background'], (0,0))

            for up_pipe, low_pipe in zip(self.upper_pipes, self.lower_pipes):
                SCREEN.blit(IMAGES['pipe'][0], (up_pipe['x'], up_pipe['y']))
                SCREEN.blit(IMAGES['pipe'][1], (low_pipe['x'], low_pipe['y']))

            SCREEN.blit(IMAGES['base'], (self.base_x, BASE_Y))
            self.show_score(self.score)

            self.player_surface = pygame.transform.rotate(IMAGES['player'][1], self.player_rot)
            SCREEN.blit(self.player_surface, (self.player_x, self.player_y))
            SCREEN.blit(IMAGES['gameover'], (50, 180))

            FPSCLOCK.tick(FPS)
            pygame.display.update()

    def playerShm(self, player_shm):
        """ Oscillates the value of player_shm['val'] between 8 and -8. """
        if abs(player_shm['val']) == 8:
            player_shm['dir'] *= -1

        if player_shm['dir'] == 1:
            player_shm['val'] += 1
        else:
            player_shm['val'] -= 1

    def get_random_pipe(self):
        """ Returns a randomly generated pipe. """
        # y of gap between upper and lower pipe
        gap_y = random.randrange(0, int(BASE_Y * 0.6 - PIPE_GAP_SIZE))
        gap_y += int(BASE_Y * 0.2)
        pipe_height = IMAGES['pipe'][0].get_height()
        pipe_x = SCREEN_WIDTH + 10

        return [
            {'x': pipe_x, 'y': gap_y - pipe_height},  # upper pipe
            {'x': pipe_x, 'y': gap_y + PIPE_GAP_SIZE},  # lower pipe
        ]

    def show_score(self, score):
        """ Displays score in center of screen. """
        score_digits = [int(x) for x in list(str(score))]
        total_width = 0  # total width of all numbers to be printed

        for digit in score_digits:
            total_width += IMAGES['numbers'][digit].get_width()

        x_offset = (SCREEN_WIDTH - total_width) / 2

        for digit in score_digits:
            SCREEN.blit(IMAGES['numbers'][digit], (x_offset, SCREEN_HEIGHT * 0.1))
            x_offset += IMAGES['numbers'][digit].get_width()

    def check_crash(self, player, upper_pipes, lower_pipes):
        """ Returns True if player colliders with base or pipes. """
        pi = player['index']
        player['w'] = IMAGES['player'][0].get_width()
        player['h'] = IMAGES['player'][0].get_height()

        # if player crashes into ground
        if player['y'] + player['h'] >= BASE_Y - 1:
            return [True, True]
        else:
            player_rect = pygame.Rect(player['x'], player['y'], player['w'], player['h'])
            pipe_w = IMAGES['pipe'][0].get_width()
            pipe_h = IMAGES['pipe'][0].get_height()

            for up_pipe, low_pipe in zip(upper_pipes, lower_pipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(up_pipe['x'], up_pipe['y'], pipe_w, pipe_h)
                lPipeRect = pygame.Rect(low_pipe['x'], low_pipe['y'], pipe_w, pipe_h)

                # player and upper/lower pipe hitmasks
                p_hitmask = HITMASKS['player'][pi]
                up_hitmask = HITMASKS['pipe'][0]
                low_hitmask = HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                up_collide = self.pixel_collision(player_rect, uPipeRect, p_hitmask, up_hitmask)
                low_collide = self.pixel_collision(player_rect, lPipeRect, p_hitmask, low_hitmask)

                if up_collide or low_collide:
                    return [True, False]

        return [False, False]

    def pixel_collision(self, rect1, rect2, hitmask1, hitmask2):
        """ Checks if two objects collide and not just their rects. """
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                    return True
        return False

    def get_hitmask(self, image):
        """ Returns a hitmask using an image's alpha. """
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x, y))[3]))
        return mask

    def get_state_rgb(self):
        return np.array(pygame.surfarray.array3d(SCREEN))

    # def get_state_values():
    #     # x, y player
    #     # hastighet y-led
    #     # x, y 3 första pipes
    #     player_array = np.array(
    #         [
    #             # player coordinates
    #             player_x,
    #             player_y,

    #             # player velocity
    #             player_vel_y
    #         ],
    #         dtype=np.float32
    #     )
    #     upper_pipes_array = []
    #     for pipe in upper_pipes[:2]:
    #         print(pipe)
    #         upper_pipes_array.append(pipe.x, pipe.y)
    #     upper_pipes_array = np.array(upper_pipes_array, dtype=np.float32)
        
    #     lower_pipes_array = []
    #     for pipe in lower_pipes[:2]:
    #         lower_pipes_array.append(pipe.x, pipe.y)
    #     lower_pipes_array = np.array(lower_pipes_array, dtype=np.float32)

    #     # print(np.concat((player_array, upper_pipes_array, lower_pipes_array),dtype=np.float32, axis=0))

    #     pass

# game = FlappyBirdAI()
# game.reset(game.movement_info)
# for i in range(600):
#     if i % 20 == 0: 
#         game.play_step(1)
#     else:
#         game.play_step(0)