import random
import math
import numpy as np

random.seed()

class Game:
    def __init__(self, cells_x, cells_y, apple_count = 1):
        self.cells_x = cells_x
        self.cells_y = cells_y
        self.apple_count = 1
        self.whole_space = set(
            (i, j) for i in range(self.cells_x) for j in range(self.cells_y)
        )
        self.edge_spaces = set((i,0) for i in range(self.cells_x)) | set((i,self.cells_y - 1) for i in range(self.cells_x)) | set((0,i) for i in range(self.cells_y)) | set((self.cells_x - 1,i) for i in range(self.cells_y))
        
        self.whole_minus_edges = self.whole_space - self.edge_spaces
        
        self.apples = set()
        self.AI_snakes = set()
        self.ML_snake = False
        self.player_snake = False

        self.empty_space_persistent = self.whole_space
        
        
    def apple_spawn_locations(self):
        return self.empty_space_persistent

    def make_apple(self):
        self.apples.add(random.choice(tuple(self.apple_spawn_locations())))

    class Snake:
        pass

    def make_snake(self, length = 4, player = False, algorithm = None, head_coords = None):
        new_snake = self.Snake()
        new_snake.dead = False
        new_snake.algorithm = algorithm
        if head_coords is not None:
            head_coords_try = head_coords
        else:
            head_coords_try = (random.randint(length + 1, self.cells_x - length - 1), random.randint(length + 1, self.cells_y - length - 1))
        x_dist = self.cells_x/2. - head_coords_try[0]
        y_dist = self.cells_y/2. - head_coords_try[1]
        if abs(x_dist) < abs(y_dist):
            if x_dist < 0:
                new_snake.direction = (-1, 0)
            else:
                new_snake.direction = (1, 0)
        else:
            if y_dist < 0:
                new_snake.direction = (0, -1)
            else:
                new_snake.direction = (0, 1)

        new_snake.coords = []
        new_snake.coords.append(head_coords_try)
        while len(new_snake.coords) < length:
            new_snake.coords.append((new_snake.coords[-1][0] - new_snake.direction[0], new_snake.coords[-1][1] - new_snake.direction[1]))
        self.empty_space_persistent = self.empty_space_persistent - set(new_snake.coords)
        new_snake.colour = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

        if player:
            self.player_snake = new_snake
        elif algorithm == 'ML':
            self.ML_snake = new_snake
        else:
            self.AI_snakes.add(new_snake)



    def time_step(self):
        # pick AI direction
        for snake in self.AI_snakes:
            if snake:
                self.pick_direction(snake)


        # move AI snakes
        for snake in self.AI_snakes.union({self.ML_snake}):
            if snake:
                snake.coords.insert(0, (snake.coords[0][0] + snake.direction[0], snake.coords[0][1] + snake.direction[1]))
                self.empty_space_persistent = self.empty_space_persistent - set([snake.coords[0]])
                if snake.coords[0] in self.apples:
                    self.apples.remove(snake.coords[0])
                else:
                    self.empty_space_persistent.add(snake.coords[-1])
                    del snake.coords[-1]

        # move player snake
        if self.player_snake is not False:
            self.player_snake.coords.insert(0, (self.player_snake.coords[0][0] + self.player_snake.direction[0], self.player_snake.coords[0][1] + self.player_snake.direction[1]))
            self.player_snake.previous_direction = self.player_snake.direction
            if self.player_snake.coords[0] in self.apples:
                self.apples.remove(self.player_snake.coords[0])
            else:
                del self.player_snake.coords[-1]

        # check for AI collisions
        for snake in self.AI_snakes.union({self.ML_snake}):
            if snake:
                if snake.coords[0][0] < 0 or snake.coords[0][0] >= self.cells_x or snake.coords[0][1] < 0 or snake.coords[0][1] >= self.cells_y:
                    snake.dead = True

                for other_snake in self.AI_snakes.union({self.ML_snake}):
                    if other_snake:
                        if snake != other_snake:
                            if snake.coords[0] in other_snake.coords:
                                snake.dead = True

                        else:
                            if snake.coords[0] in snake.coords[1:]:
                                snake.dead = True


                if self.player_snake is not False:
                    if snake[0] in self.player_snake.coords:
                        snake.dead = True

        # check for player collisions
        if self.player_snake is not False:
            if self.player_snake.coords[0][0] < 0 or self.player_snake.coords[0][0] > self.cells_x or self.player_snake.coords[0][1] < 0 or self.player_snake.coords[0][1] > self.cells_y:
                self.player_snake.dead = True

            if self.player_snake.coords[0] in self.player_snake.coords[1:]:
                self.player_snake.dead = True


        # check for repopulating apples
        if len(self.apples) < self.apple_count:
            self.make_apple()


    def move_up(self):
        if self.player_snake is not False:
            if self.player_snake.previous_direction != (0, 1):
                self.player_snake.direction = (0, -1)

    def move_down(self):
        if self.player_snake is not False:
            if self.player_snake.previous_direction != (0, -1):
                self.player_snake.direction = (0, 1)

    def move_left(self):
        if self.player_snake is not False:
            if self.player_snake.previous_direction != (1, 0):
                self.player_snake.direction = (-1, 0)

    def move_right(self):
        if self.player_snake is not False:
            if self.player_snake.previous_direction != (-1, 0):
                self.player_snake.direction = (1, 0) 

    def check_lose_states(self):
        if self.player_snake is not False:
            if self.player_snake.dead == True:
                return True
        for snake in self.AI_snakes.union({self.ML_snake}):
            if snake:
                if snake.dead == True:
                    return True

    def pick_random_direction(self, snake):
        possible_moves = []
        empty_spaces = self.empty_space_persistent
        directions = [(1,0), (-1,0), (0,1), (0,-1)]

        for direction in directions:
            new_spot = (snake.coords[0][0] + direction[0], snake.coords[0][1] + direction[1])
            if new_spot in empty_spaces:
                possible_moves.append(direction)

        if possible_moves != []:
            snake.direction = random.choice(possible_moves)

    # This chooses a random direction even if it means running into a wall or self
    def pick_random_direction_no_safe(self, snake):
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        # prevents going directly backwards
        directions.remove((-snake.direction[0], -snake.direction[1]))
        snake.direction = random.choice(directions)


    def pick_basic_direction(self, snake):
        possible_moves = []
        shortest_distance = self.cells_x*self.cells_y
        empty_spaces = self.empty_space()
        directions = [(1,0), (-1,0), (0,1), (0,-1)]

        for direction in directions:
            new_spot = (snake.coords[0][0] + direction[0], snake.coords[0][1] + direction[1])
            if new_spot in empty_spaces | self.apples:
                for apple in self.apples:
                    if self.distance_to_apple_manhattan(new_spot, apple) == shortest_distance:
                        possible_moves.append(direction)
                    elif self.distance_to_apple_manhattan(new_spot, apple) < shortest_distance:
                        possible_moves = []
                        possible_moves.append(direction)
                        shortest_distance = self.distance_to_apple_manhattan(new_spot, apple)
        if possible_moves != []:
            snake.direction = random.choice(possible_moves)

    def pick_basic_direction_v2(self, snake):
        possible_moves = []
        shortest_distance = self.cells_x*self.cells_y
        empty_spaces = self.empty_space_avoid_other_heads(snake)
        directions = [(1,0), (-1,0), (0,1), (0,-1)]

        for direction in directions:
            new_spot = (snake.coords[0][0] + direction[0], snake.coords[0][1] + direction[1])
            if new_spot in empty_spaces | self.apples:
                for apple in self.apples:
                    if self.distance_to_apple_manhattan(new_spot, apple) == shortest_distance:
                        possible_moves.append(direction)
                    elif self.distance_to_apple_manhattan(new_spot, apple) < shortest_distance:
                        possible_moves = []
                        possible_moves.append(direction)
                        shortest_distance = self.distance_to_apple_manhattan(new_spot, apple)
        if possible_moves != []:
            snake.direction = random.choice(possible_moves)
        else:
            print('No possible moves in v2, trying near heads (v1):')
            self.pick_basic_direction(snake)



    def distance_to_apple_manhattan(self, location, apple):
        return abs(apple[0] - location[0]) + abs(apple[1] - location[1])


    def closest_apple_manhattan(self, snake):
        shortest_distance = self.cells_x*self.cells_y
        found_apple = None
        for apple in self.apples:
            distance = self.distance_to_apple_manhattan(snake.coords[0], apple)
            if distance < shortest_distance:
                found_apple = apple

        return found_apple

    def m_dist(pointA, pointB):
        return abs(pointA[0] - pointB[0]) + abs(pointA[1] - pointB[1])

    def get_direction_to_apple(self, snake):
        # print('starting custom path, head at ', snake.coords[0])
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        # set untested cells to 0
        cost_grid = np.full((self.cells_x, self.cells_y), -1)
        # edge_set are spaces that still need expanding around
        edge_set = set()
        # tested_set are spaces that have already been expanded around
        tested_set = set()

        # just making snake head for testing
        cost_grid[snake.coords[0][0], snake.coords[0][1]] = -100

        # This does initial check of adjacent cells
        found_initial = False
        found_path_cell = False
        for direction in directions:
            new_spot = (snake.coords[0][0] + direction[0], snake.coords[0][1] + direction[1])
            if new_spot not in self.empty_space_persistent:
                if 0 <= new_spot[0] < self.cells_x and 0 <= new_spot[1] < self.cells_y:
                    # if impassible, then set to -1
                    cost_grid[new_spot[0], new_spot[1]] = -1
                    # don't want to expand around impassible, since we cannot path through it
                    tested_set.add(new_spot)
            else:
                if new_spot in self.apples:
                    # Apple is right next to snake head, take it!
                    # print('apple is next door!')
                    found_path_cell = new_spot

                found_initial = True
                # cost is how many moves to get there; adjacent is hence cost 1
                cost_grid[new_spot[0], new_spot[1]] = 1
                edge_set.add(new_spot)
                # if new_spot in snake.coords:
                #     print('Error: new spot is in snake')

        # finished expanding around the head
        tested_set.add(snake.coords[0])

        if not found_initial:
        #    print('No possible initial moves!')
           return False

        # We have found initial moves that do not contain the apple, begin searching...

        target_found = False
        target_apple = False

        while not target_found:
            # dummy set to iterate through 
            iterating_set = edge_set.copy()

            #expand around each edge cell
            # if this is false, could not find any new cells, and so cannot find path
            found_new_cell = False
            for cell in iterating_set:
                if cell in tested_set:
                    continue
                for direction in directions:
                    new_spot = (cell[0] + direction[0], cell[1] + direction[1])

                    # checking if new spot is target
                    if new_spot in self.apples:
                        target_found = True
                        target_apple = new_spot
                        continue
                    # checking for if we have already checked this site
                    if new_spot in tested_set or new_spot in edge_set:
                        continue
                    # checking if spot is passable or not
                    elif new_spot not in self.empty_space_persistent:
                        if 0 <= new_spot[0] < self.cells_x and 0 <= new_spot[1] < self.cells_y:
                            # if impassible, then set to -1
                            cost_grid[new_spot[0], new_spot[1]] = -1
                    # seems walkable, set the value:
                    else:
                        cost_grid[new_spot[0], new_spot[1]] = cost_grid[cell[0], cell[1]] + 1
                        edge_set.add(new_spot)
                        # if new_spot in snake.coords:
                            # print('Error: new spot is in snake 2')
                edge_set.remove(cell)
                tested_set.add(cell)
                found_new_cell = True
            if not found_new_cell:
                # print('ran out of cells to expand to.')
                return False

        # target now found, now to trace back path
        # print('Apple found at ', target_apple)

        # print('cost_grid is ', cost_grid.transpose())

        # look at cells around target apple
        # debug_counter = 0
        centre_cell = target_apple
        while not found_path_cell:
            # debug_counter += 1
            # if debug_counter > 20:
            #     break
            # print('while not found_path_cell:')
            adjacent_spots = []
            lowest_cost = False
            # print('cost_grid is ', cost_grid.transpose())

            for direction in directions:
                new_spot = (centre_cell[0] + direction[0], centre_cell[1] + direction[1])
                if new_spot == snake.coords[0]:
                    # print('found_path_cell:', found_path_cell)
                    found_path_cell = centre_cell
                    break
                if new_spot in self.empty_space_persistent:
                    new_cost = cost_grid[new_spot[0], new_spot[1]]
                    if new_cost >= 0:
                        # print('if new_cost >= 0:')
                        if lowest_cost == False:
                            lowest_cost = new_cost
                            adjacent_spots = [new_spot]
                        elif new_cost < lowest_cost:
                            lowest_cost = new_cost
                            adjacent_spots = [new_spot]
                        elif new_cost == lowest_cost:
                            adjacent_spots.append(new_spot)

            if not found_path_cell and len(adjacent_spots) > 0:
                # print('adjacent_spots:', adjacent_spots)
                centre_cell = random.choice(adjacent_spots)
                # print('picked new centre cell: ', centre_cell)


        found_direction = (found_path_cell[0] - snake.coords[0][0], found_path_cell[1] - snake.coords[0][1])
        # print('found direction:', found_direction)
        snake.direction = found_direction
        return True
        



    def pick_custom_direction(self, snake):
        if len(self.apples) > 0:
            if not self.get_direction_to_apple(snake):
                # print('Could not use algorithm, using random...')
                self.pick_random_direction(snake)

    def pick_ML_direction(self, snake):
        self.pick_random_direction_no_safe(snake)

    def pick_direction(self, snake):
        # if snake.algorithm == 'dumb_v2':
        #     self.pick_basic_direction_v2(snake)
        # elif snake.algorithm == 'astar':
        #     self.pick_astar_direction(snake)
        # elif snake.algorithm == 'astar_v2':
        #     self.pick_astar_direction_v2(snake)
        
        if snake.algorithm == 'ML':
            self.pick_ML_direction(snake)
        elif snake.algorithm == 'custom':
            self.pick_custom_direction(snake)

cells_x, cells_y = 10, 10

def run_game_no_gui():
    new_game = Game(cells_x,cells_y)
    new_game.make_snake(2, False, "custom")
    game_over = False
    step_count = 0
    while not game_over:
        # This is the working game loop
        new_game.time_step()
        step_count += 1

        if new_game.check_lose_states():
            game_over = True
            for snake in new_game.AI_snakes.union({new_game.ML_snake}):
                if snake:
                    snake_length = len(snake.coords)
                    print('finished game with length:', snake_length)
                    return snake_length

def run_game_no_gui_ML():
    new_game = Game(cells_x,cells_y)
    new_game.make_snake(2, False, "ML")
    game_over = False
    step_count = 0
    while not game_over:
        # This is the working game loop
        new_game.time_step()
        step_count += 1

        # To prevent infinite looping
        if step_count > 100*len(new_game.ML_snake.coords):
            game_over = True
            print('Too many repetitions without eating!')
            snake_length = len(new_game.ML_snake.coords)
            print('finished game with length:', snake_length)
            return snake_length

        if new_game.check_lose_states():
            game_over = True
            snake_length = len(new_game.ML_snake.coords)
            print('finished game with length:', snake_length)
            return snake_length



number_of_games = 100
def main():
    lengths = []
    for i in range(number_of_games):
        lengths.append(run_game_no_gui_ML())
    print('Average length over '+str(number_of_games)+' games:', sum(lengths)/len(lengths))


if __name__ == '__main__':
    main()