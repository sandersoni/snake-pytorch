import torch
import random
import math
import numpy as np
from collections import deque
from game import Game

GUI = True

game_runs = 100
# size of grid
cells_x, cells_y = 10, 10

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
# learning rate:
LR = 0.001

class Agent:

    def __init__(self):

        self.n_games = 0

        # randomness
        self.epsilon = 0
        # discount rate
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY) # automatically pops left when beyond max
        # TODO: model, trainer


    def get_state(self, game):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self):
        pass

    def get_action(self, state):
        pass

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game(cells_x, cells_y)
    
    while True:
        # get old/current state
        state_old = agent.get_state()
        
        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        # reward, done, score = 

def train_gui():
    import pygame
    cells_x, cells_y = 10, 10
    window_width, window_height = cells_x*20, cells_y*20
    cells_x_pixels, cells_y_pixels = math.floor(window_width/cells_x), math.floor(window_height/cells_y)
    window_x = 960 - math.floor(window_width/2.)
    window_y = 540 - math.floor(window_height/2.)
    pygame.init()
    win=pygame.display.set_mode((window_width, window_height))
    background_fill = (13, 70, 77)
    win.fill(background_fill)
    apple_colour = (28, 212, 77)
    pygame.display.set_mode((window_width + 1, window_height))
    pygame.display.set_mode((window_width, window_height))
    pygame.display.update()
    pygame.display.set_caption('Snake game')
    clock_tick = 60
    def terminate():
        global game_over
        game_over = True
        pygame.quit()
        quit()
    def handle_events(game):
        for event in pygame.event.get():
            # LOGGER.log(5, 'event: {0}'.format(event))
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                terminate()
    def draw_snake_cell(coords, colour):
        pygame.draw.rect(win, colour, [coords[0]*cells_x_pixels + 1, coords[1]*cells_y_pixels + 1, cells_x_pixels - 1, cells_y_pixels - 1])
    def draw_snake_head(coords, colour, direction):
        if direction == (0, -1): # up direction
            pygame.draw.polygon(win, colour, [(coords[0]*cells_x_pixels + 1, coords[1]*cells_y_pixels + cells_y_pixels - 1), (coords[0]*cells_x_pixels + cells_x_pixels - 1, coords[1]*cells_y_pixels + cells_y_pixels - 1), (coords[0]*cells_x_pixels + math.floor(cells_x_pixels/2), coords[1]*cells_y_pixels + 1)])
        if direction == (0, 1): # down direction
            pygame.draw.polygon(win, colour, [(coords[0]*cells_x_pixels + 1, coords[1]*cells_y_pixels + 1), (coords[0]*cells_x_pixels + cells_x_pixels - 1, coords[1]*cells_y_pixels + 1), (coords[0]*cells_x_pixels + math.floor(cells_x_pixels/2), coords[1]*cells_y_pixels + cells_y_pixels - 1)])
        if direction == (-1, 0): # left direction
            pygame.draw.polygon(win, colour, [(coords[0]*cells_x_pixels + cells_x_pixels - 1, coords[1]*cells_y_pixels + 1), (coords[0]*cells_x_pixels + cells_x_pixels - 1, coords[1]*cells_y_pixels + cells_y_pixels - 1), (coords[0]*cells_x_pixels + 1, coords[1]*cells_y_pixels + math.floor(cells_y_pixels/2))])
        if direction == (1, 0): # right direction
            pygame.draw.polygon(win, colour, [(coords[0]*cells_x_pixels + 1, coords[1]*cells_y_pixels + 1), (coords[0]*cells_x_pixels + 1, coords[1]*cells_y_pixels + cells_y_pixels - 1), (coords[0]*cells_x_pixels + cells_x_pixels - 1, coords[1]*cells_y_pixels + math.floor(cells_y_pixels/2))])
    def draw_apple(coords, colour):
        pygame.draw.rect(win, colour, [coords[0]*cells_x_pixels + 1, coords[1]*cells_y_pixels + 1, cells_x_pixels - 1, cells_y_pixels - 1])
    clock = pygame.time.Clock()

    ## GUI RUNNING LOOPS
    game_counter = 0
    while game_counter < game_runs:
        game_run = Game(cells_x, cells_y)
        game_run.make_snake(3, False, "ML")
        game_over = False
        while not game_over:
            handle_events(game_run)
            ##### ML direction needs to be picked here
            game_run.pick_random_direction_no_safe(game_run.ML_snake)



            #####
            game_run.time_step()
            win.fill(background_fill)
            if game_run.player_snake is not False:
                for segment in game_run.player_snake.coords:
                    draw_snake_cell(segment, game_run.player_snake.colour)
            for snake in game_run.AI_snakes.union({game_run.ML_snake}):
                if snake:
                    first = True
                    for segment in snake.coords:
                        if first == True:
                            first = False
                        else:
                            draw_snake_cell(segment, snake.colour)
                    draw_snake_head(snake.coords[0], snake.colour, snake.direction)
            for apple in game_run.apples:
                draw_apple(apple, apple_colour)
            if game_run.check_lose_states():
                game_over = True
            pygame.display.update()
            clock.tick(clock_tick)
        game_counter += 1
        print('games completed: '+str(game_counter))

if __name__ == '__main__':
    if GUI:
        train_gui()
    else:
        train()