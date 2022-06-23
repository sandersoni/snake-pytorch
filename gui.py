# import os
# os.environ["SDL_VIDEODRIVER"]="x11"
# os.environ["SDL_AUDIODRIVER"]="x11"



import math

# window_width, window_height = 300, 300
cells_x, cells_y = 10, 10
window_width, window_height = cells_x*20, cells_y*20

cells_x_pixels, cells_y_pixels = math.floor(window_width/cells_x), math.floor(window_height/cells_y)


# centering position of window to middle of screen
window_x = 960 - math.floor(window_width/2.)
window_y = 540 - math.floor(window_height/2.)

# os.environ['SDL_VIDEO_WINDOW_POS'] = '%d,%d' % (window_x,window_y)

import pygame
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

# def start_game():
#     print('Starting!')
#     global game_started
#     game_started = True

def terminate():
    global game_started
    global game_over
    global close_game
    game_started = True
    game_over = True
    close_game = True

    pygame.quit()
    quit()

from game import Game

clock = pygame.time.Clock()

start_font = pygame.font.Font(None, 30)

close_font = pygame.font.Font(None, 30)

astar_font = pygame.font.Font(None, 15)

def main():
    game_started = False
    start_text_surface = start_font.render('Press Enter To Start', True, (255,255,255))
    start_text_rect = start_text_surface.get_rect(center = (math.floor(window_width/2.),math.floor(window_height/2.)))
    while not game_started:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminate()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    game_started = True
        win.fill(background_fill)
        win.blit(start_text_surface, start_text_rect)
        pygame.display.update()

    game_run = Game(cells_x, cells_y)
    
    # makes a player snake
    # game_run.make_snake(4, True)

    # makes an AI snake
    # game_run.make_snake(15, False, "dumb_v2", (38,20))
    # game_run.make_snake(5, False, "dumb_v2")
    game_run.make_snake(2, False, "custom")
    # game_run.make_snake(1, False, "astar_v2")
    # game_run.make_snake(1, False, "astar_v2")


    # game_run.make_snake(4, False, "astar", (39, 7))


    game_over = False
    while not game_over:

        # handling inputs
        handle_events(game_run)

        # picking direction for AI
        # for snake in game_run.AI_snakes:
        #     game_run.pick_direction(snake)
            # game_run.pick_basic_direction(snake)
            # game_run.pick_basic_direction_v2(snake)
            # if snake.algorithm == "astar":
                # if game_run.closest_apple_manhattan(snake) != None:
                    # print('Attempting astar...')
                    # path = game_run.astar(snake.coords[0], game_run.closest_apple_manhattan(snake), pygame, win, astar_font, window_width, window_height, cells_x, cells_y)
                    # print('path is ', path)

        # processing and moving snakes:
        game_run.time_step()

        win.fill(background_fill)
        if game_run.player_snake is not False:
            for segment in game_run.player_snake.coords:
                draw_snake_cell(segment, game_run.player_snake.colour)

        for snake in game_run.AI_snakes:
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

    close_game = False

    close_text_surface = close_font.render('Game ended!', True, (255,255,255))
    close_text_rect = close_text_surface.get_rect(center = (math.floor(window_width/2.),math.floor(window_height/2.)))

    while not close_game:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminate()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                    close_game = True
        win.fill(background_fill)

        if game_run.player_snake is not False:
            for segment in game_run.player_snake.coords:
                draw_snake_cell(segment, game_run.player_snake.colour)

        for snake in game_run.AI_snakes:
            first = True
            for segment in snake.coords:
                if first == True:
                    first = False
                else:
                    draw_snake_cell(segment, snake.colour)
            draw_snake_head(snake.coords[0], snake.colour, snake.direction)
            

        for apple in game_run.apples:
            draw_apple(apple, apple_colour)

        win.blit(close_text_surface, close_text_rect)

        pygame.display.update()


def handle_events(game):
    for event in pygame.event.get():
        # LOGGER.log(5, 'event: {0}'.format(event))
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            terminate()
        if event.type == pygame.KEYDOWN and (event.key == pygame.K_UP or event.key == pygame.K_w):
            game.move_up()
        if event.type == pygame.KEYDOWN and (event.key == pygame.K_DOWN or event.key == pygame.K_s):
            game.move_down()
        if event.type == pygame.KEYDOWN and (event.key == pygame.K_LEFT or event.key == pygame.K_a):
            game.move_left()
        if event.type == pygame.KEYDOWN and (event.key == pygame.K_RIGHT or event.key == pygame.K_d):
            game.move_right()

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

if __name__ == '__main__':
    main()