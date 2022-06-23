# import logging
# LOGGER = logging.getLogger(__name__)

from multiprocessing import parent_process
import random
import math
import numpy as np
from sympy import numbered_symbols

random.seed()

class Game:
    def __init__(self, cells_x, cells_y, apple_count = 1):
        # print('Initialising')
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

        self.player_snake = False

        self.empty_space_persistent = self.whole_space
        # self.empty_space_persistent_avoid_heads = self.whole_space
        

    # def empty_space(self):
    #     empty_space_result = self.whole_space
    #     for snake in self.AI_snakes:
    #         empty_space_result = empty_space_result - set(snake.coords)
    #     if self.player_snake is not False:
    #         empty_space_result = empty_space_result - set(self.player_snake.coords)
    #     return empty_space_result

    # def empty_space_avoid_other_heads(self, snake):
    #     empty_space_result = self.empty_space_persistent
    #     for other_snake in self.AI_snakes:
    #         if snake != other_snake:
    #             empty_space_result = empty_space_result - set([(other_snake.coords[0][0] + 1, other_snake.coords[0][1]), (other_snake.coords[0][0] - 1, other_snake.coords[0][1]), (other_snake.coords[0][0], other_snake.coords[0][1] + 1), (other_snake.coords[0][0], other_snake.coords[0][1] - 1)])
    #     if self.player_snake is not False:
    #         empty_space_result = empty_space_result - set([(self.player_snake.coords[0][0] + 1, self.player_snake.coords[0][1]), (self.player_snake.coords[0][0] - 1, self.player_snake.coords[0][1]), (self.player_snake.coords[0][0], self.player_snake.coords[0][1] + 1), (self.player_snake.coords[0][0], self.player_snake.coords[0][1] - 1)])

    #     return empty_space_result
        
    def apple_spawn_locations(self):
        empty_space_result = self.whole_minus_edges - self.apples
        for snake in self.AI_snakes:
            empty_space_result = empty_space_result - set(snake.coords)
        if self.player_snake is not False:
            empty_space_result = empty_space_result - set(self.player_snake.coords)
        return empty_space_result

    def make_apple(self):
        self.apples.add(random.choice(tuple(self.apple_spawn_locations())))

    class Snake:
        pass

    def make_snake(self, length = 4, player = False, algorithm = None, head_coords = None):
        # print('Making a snake')
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
                # print('going negative x')
                new_snake.direction = (-1, 0)
            else:
                # print('going positive x')
                new_snake.direction = (1, 0)
        else:
            if y_dist < 0:
                # print('going negative y')
                new_snake.direction = (0, -1)
            else:
                # print('going positive y')
                new_snake.direction = (0, 1)

        new_snake.coords = []
        new_snake.coords.append(head_coords_try)
        while len(new_snake.coords) < length:
            new_snake.coords.append((new_snake.coords[-1][0] - new_snake.direction[0], new_snake.coords[-1][1] - new_snake.direction[1]))
            # print('adding segment at ', new_snake.coords[-1])

        # self.empty_space_persistent = self.empty_space_persistent - set(new_snake.coords + [(new_snake.coords[0][0] + 1, new_snake.coords[0][1]), (new_snake.coords[0][0] - 1, new_snake.coords[0][1]), (new_snake.coords[0][0], new_snake.coords[0][1] + 1), (new_snake.coords[0][0], new_snake.coords[0][1] - 1)])

        self.empty_space_persistent = self.empty_space_persistent - set(new_snake.coords)

        # print('New empty space avoid heads is ', self.empty_space_persistent_avoid_heads)


        new_snake.colour = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

        if player:
            self.player_snake = new_snake
        else:
            self.AI_snakes.add(new_snake)



    def time_step(self):
        # pick AI direction
        for snake in self.AI_snakes:
            self.pick_direction(snake)


        # move AI snakes
        for snake in self.AI_snakes:
            snake.coords.insert(0, (snake.coords[0][0] + snake.direction[0], snake.coords[0][1] + snake.direction[1]))
            self.empty_space_persistent = self.empty_space_persistent - set([snake.coords[0]])
            if snake.coords[0] in self.apples:
                # print('snake eating apple at ', snake.coords[0], ', new length: ', len(snake.coords))
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
        for snake in self.AI_snakes:
            if snake.coords[0][0] < 0 or snake.coords[0][0] >= self.cells_x or snake.coords[0][1] < 0 or snake.coords[0][1] >= self.cells_y:
                # print('AI wall collision!')
                snake.dead = True

            for other_snake in self.AI_snakes:
                if snake != other_snake:
                    if snake.coords[0] in other_snake.coords:
                        # print('AI snake collision!')
                        snake.dead = True

                else:
                    if snake.coords[0] in snake.coords[1:]:
                        # print('AI collided with self!')
                        snake.dead = True


            if self.player_snake is not False:
                if snake[0] in self.player_snake.coords:
                    # print('AI collided with player!')
                    snake.dead = True

        # check for player collisions
        if self.player_snake is not False:
            if self.player_snake.coords[0][0] < 0 or self.player_snake.coords[0][0] > self.cells_x or self.player_snake.coords[0][1] < 0 or self.player_snake.coords[0][1] > self.cells_y:
                # print('Player wall collision!')
                self.player_snake.dead = True

            if self.player_snake.coords[0] in self.player_snake.coords[1:]:
                # print('Player collided with self!')
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
        for snake in self.AI_snakes:
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
        # else:
        #     print('No possible moves!')

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
        # else:
        #     print('No possible moves!')
            
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
        # print('head at', snake.coords[0], ', possible moves: ', possible_moves)
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

    # A-star stuff now

    def draw_value(self, pygame, window, font, coords, value, window_width, window_height, cells_x, cells_y):
        cells_x_pixels, cells_y_pixels = math.floor(window_width/cells_x), math.floor(window_height/cells_y)
        text_surface = font.render(value, True, (255,255,255))
        text_rect = text_surface.get_rect(center = (math.floor(coords[0]*cells_x_pixels + cells_x_pixels/2.), math.floor(coords[1]*cells_y_pixels + cells_y_pixels/2.)))
        window.blit(text_surface, text_rect)
        pygame.display.update()

    class Node():
        def __init__(self, parent=None, coords=None):
            self.parent = parent
            self.coords = coords
            self.g = 0
            self.h = 0
            self.f = 0

        def __eq__(self, other):
            return self.coords == other.coords

    def astar(self, start, end):#, pygame, window, font, window_width, window_height, cells_x, cells_y):
        start_node = self.Node(None, start)
        end_node = self.Node(None, end)
        open_list = []
        closed_list = []
        open_list.append(start_node)

        while len(open_list) > 0:
            # print('while len(open_list) > 0 running')
            current_node = open_list[0]
            current_index = 0

            for index,item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index
            
            open_list.pop(current_index)
            closed_list.append(current_node)

            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.coords)
                    current = current.parent
                return path[::-1]
        
            children = []
            for new_position in [(1,0), (-1,0), (0,1), (0,-1)]:
                node_position = (current_node.coords[0] + new_position[0], current_node.coords[1] + new_position[1])

                # if node_position[0] < 0 or node_position[0] > self.cells_x - 1 or node_position[1] < 0 or node_position[1] > self.cells_y - 1:
                #     continue

                if node_position not in self.empty_space():
                    continue
                    
                new_node = self.Node(current_node, node_position)

                children.append(new_node)

            for child in children:
                skip = False
                for closed_child in closed_list:
                    if child == closed_child:
                        skip = True
                        # print('child ', child.coords, 'found in closed list')
                        continue
                if skip:
                    continue

                child.g = current_node.g + 1
                child.h = ((child.coords[0] - end_node.coords[0]) ** 2) + ((child.coords[1] - end_node.coords[1]) ** 2)
                child.f = child.g + child.h

                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                open_list.append(child)
                # print('appending child ', child.coords)
                # self.draw_value(pygame, window, font, child.coords, str(child.f), window_width, window_height, cells_x, cells_y)

    def astar_avoid_heads(self, snake, end):#, pygame, window, font, window_width, window_height, cells_x, cells_y):
        start = snake.coords[0]
        start_node = self.Node(None, start)
        end_node = self.Node(None, end)
        open_list = []
        closed_list = []
        open_list.append(start_node)
        empty_space = self.empty_space()
        empty_space_avoid_heads = self.empty_space_avoid_other_heads(snake)

        if end not in empty_space_avoid_heads:
            possible_random_directions = []
            for direction in [(1,0), (-1,0), (0,1), (0,-1)]:
                if (snake.coords[0][0] + direction[0], snake.coords[0][1] + direction[1]) in empty_space_avoid_heads:
                    possible_random_directions.append(direction)

            if possible_random_directions != []:
                snake.direction = random.choice(possible_random_directions)
            
            return None

        while len(open_list) > 0:
            # print('while len(open_list) > 0 running')
            current_node = open_list[0]
            current_index = 0

            for index,item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index
            
            open_list.pop(current_index)
            closed_list.append(current_node)

            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.coords)
                    current = current.parent
                return path[::-1]
        
            children = []
            for new_position in [(1,0), (-1,0), (0,1), (0,-1)]:
                node_position = (current_node.coords[0] + new_position[0], current_node.coords[1] + new_position[1])

                # if node_position[0] < 0 or node_position[0] > self.cells_x - 1 or node_position[1] < 0 or node_position[1] > self.cells_y - 1:
                #     continue

                if current_node.coords == snake.coords[0]:
                    print('snake head is ', snake.coords[0], ', node position is ', node_position)
                    if node_position not in empty_space_avoid_heads:
                        print('node not in empty space avoid heads, skipping')
                        continue
                elif node_position not in empty_space:
                    continue
                    
                new_node = self.Node(current_node, node_position)

                children.append(new_node)

            for child in children:
                skip = False
                for closed_child in closed_list:
                    if child == closed_child:
                        skip = True
                        # print('child ', child.coords, 'found in closed list')
                if skip:
                    continue

                child.g = current_node.g + 1
                child.h = ((child.coords[0] - end_node.coords[0]) ** 2) + ((child.coords[1] - end_node.coords[1]) ** 2)
                child.f = child.g + child.h

                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                open_list.append(child)
                # print('appending child ', child.coords)
                # self.draw_value(pygame, window, font, child.coords, str(child.f), window_width, window_height, cells_x, cells_y)
    
    def pick_astar_direction(self, snake):
        target_apple = self.closest_apple_manhattan(snake)
        if target_apple is not None:
            path = self.astar(snake.coords[0], target_apple)
            # print('snake head is ', snake.coords[0], ', path[0] is ', path[0])
            snake.direction = (path[1][0] - snake.coords[0][0], path[1][1] - snake.coords[0][1])

    def pick_astar_direction_v2(self, snake):
        target_apple = self.closest_apple_manhattan(snake)
        if target_apple is not None:
            path = self.astar_avoid_heads(snake, target_apple)
            # print('snake head is ', snake.coords[0], ', path[0] is ', path[0])
            if path is not None:
                snake.direction = (path[1][0] - snake.coords[0][0], path[1][1] - snake.coords[0][1])
            else:
                print('astar path not found!')
                self.pick_random_direction(snake)


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


    def pick_direction(self, snake):
        if snake.algorithm == 'dumb_v2':
            self.pick_basic_direction_v2(snake)
        elif snake.algorithm == 'astar':
            self.pick_astar_direction(snake)
        elif snake.algorithm == 'astar_v2':
            self.pick_astar_direction_v2(snake)
        elif snake.algorithm == 'custom':
            self.pick_custom_direction(snake)

cells_x, cells_y = 10, 10

def run_game_no_gui():
    new_game = Game(cells_x,cells_y)
    new_game.make_snake(2, False, "custom")
    game_over = False
    while not game_over:
        new_game.time_step()
        if new_game.check_lose_states():
            game_over = True
            for snake in new_game.AI_snakes:
                snake_length = len(snake.coords)
                print('finished game with length:', snake_length)
                return snake_length


number_of_games = 100
def main():
    lengths = []
    for i in range(number_of_games):
        lengths.append(run_game_no_gui())
    print(sum(lengths)/len(lengths))


if __name__ == '__main__':
    main()