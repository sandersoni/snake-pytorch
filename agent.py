import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import math
import numpy as np
from collections import deque
from game import Game
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('detected torch device is ', device)
if device == 'cpu':
    print('a GPU is highly recommended.')


GUI = False
PLOT = True

game_runs = 1000000
# size of grid
cells_x, cells_y = 10, 10

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
# learning rate:
LR = 0.001


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
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


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()





class Agent:

    def __init__(self):

        self.n_games = 0

        # randomness
        self.epsilon = 0
        # discount rate
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY) # automatically pops left when beyond max
        self.model = Linear_QNet(cells_x*cells_y, 256, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)



    def get_state(self, game):
        # head is 1, body is 2, and apple(s) are 3
        state = np.zeros((game.cells_x, game.cells_y), dtype=int)
        if 0 <= game.ML_snake.coords[0][0] < game.cells_x and 0 <= game.ML_snake.coords[0][1] < game.cells_y:
            state[game.ML_snake.coords[0][0], game.ML_snake.coords[0][1]] = 1
        for segment in game.ML_snake.coords[1:]:
            state[segment[0], segment[1]] = 2
        for apple in game.apples:
            state[apple[0], apple[1]] = 3
        state = np.concatenate(state, axis=None)
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(np.array(states), actions, rewards, np.array(next_states), dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, game):
        self.epsilon = 1000 - self.n_games
        final_move = [0,0,0,0]
        if random.randint(0,2000) < self.epsilon:
            directions = [(1,0), (-1,0), (0,1), (0,-1)]
            # prevents going directly backwards            
            # directions.remove((-game.ML_snake.direction[0], -game.ML_snake.direction[1]))
            choice = random.choice([direction for direction in directions if direction != (-game.ML_snake.direction[0], -game.ML_snake.direction[1])])
            final_move[directions.index(choice)] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
            

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    agent = Agent()
    game_counter = 0
    while game_counter < game_runs:
        # New game:
        game_run = Game(cells_x, cells_y)
        old_snake_length = 3
        game_run.make_snake(old_snake_length, False, "ML")
        game_over = False
        step_count = 0
        while not game_over:

            state_old = agent.get_state(game_run)
            reward = 0
            final_move = agent.get_action(state_old, game_run)
            game_run.ML_snake.direction = directions[final_move.index(1)]
            step_count =+ 1
            game_run.time_step()
            new_snake_length = len(game_run.ML_snake.coords)

            if new_snake_length > old_snake_length:
                reward = 100
                old_snake_length = new_snake_length
            if game_run.check_lose_states() or step_count > 100*new_snake_length:
                reward = -100
                game_over = True
            else:
                reward =+ 1
            
            state_new = agent.get_state(game_run)

            agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

            agent.remember(state_old, final_move, reward, state_new, game_over)
        
        # Game over, process results:

        game_counter += 1
        agent.train_long_memory()
        score = len(game_run.ML_snake.coords)
        if score > record:
            record = score
            agent.model.save()
        print('length: '+str(score)+', game count: '+str(game_counter)+', record: '+str(record))
        if PLOT:
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / game_counter
            plot_mean_scores.append(mean_score)
            plt.clf()
            plt.plot(plot_scores)
            plt.plot(plot_mean_scores)
            plt.show(block=False)
            plt.pause(0.001)

     

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

    agent = Agent()
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    ## GUI RUNNING LOOPS
    game_counter = 0
    while game_counter < game_runs:

        # New game:

        game_run = Game(cells_x, cells_y)
        old_snake_length = 3
        game_run.make_snake(old_snake_length, False, "ML")
        game_over = False
        step_count = 0

        while not game_over:
            handle_events(game_run)
            ################## ML stuff needs to be placed here
            state_old = agent.get_state(game_run)
            reward = 0

            final_move = agent.get_action(state_old, game_run)
            game_run.ML_snake.direction = directions[final_move.index(1)]

            # game_run.pick_random_direction_no_safe(game_run.ML_snake)

            step_count =+ 1
            game_run.time_step()

            new_snake_length = len(game_run.ML_snake.coords)
            if new_snake_length > old_snake_length:
                reward = 100
                old_snake_length = new_snake_length
            if game_run.check_lose_states() or step_count > 100*new_snake_length:
                reward = -100
                game_over = True
            else:
                reward =+ 1
            
            state_new = agent.get_state(game_run)

            agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

            agent.remember(state_old, final_move, reward, state_new, game_over)

            ##################
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
            pygame.display.update()
            clock.tick(clock_tick)

        # Game over, process results:

        game_counter += 1
        agent.train_long_memory()
        score = len(game_run.ML_snake.coords)
        if score > record:#plot_mean_scores[-1]:
            record = score
            agent.model.save()
        print('length: '+str(score)+', game count: '+str(game_counter)+', record: '+str(record))
        if PLOT:
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / game_counter
            plot_mean_scores.append(mean_score)
            plt.clf()
            plt.plot(plot_scores)
            plt.plot(plot_mean_scores)
            plt.show(block=False)
            plt.pause(0.001)
            

if __name__ == '__main__':
    if GUI:
        train_gui()
    else:
        train()