###############################################################################################################
#                                             Mena S.A. Kamel
#                           3547-014: Intelligent Agents and Reinforcement Learning
#                                              Assignment #1
#                                             February 20, 2022                                   
###############################################################################################################

from stringprep import b3_exceptions
import numpy as np
import environment
import random

class BeelineAgent:
    def __init__(self, game):
        '''
        Initializes a Naive Agent
        '''
        self.game = game
        # Initial agent heading and location (belief state)
        self.home_location = (0, 0)
        self.location = (0, 0)
        self.heading = "right"
        self.has_gold = False
        self.safe_locations = set([self.location])
        
        self.infinity_cost = 999999
        self.heurist_matrix = self.compute_heuristic_matrix()
    
    def compute_heuristic_matrix(self):
        heuristic_matrix = np.zeros((self.game.height, self.game.width))
        for x in range(self.game.width):
            for y in range(self.game.height):
                curr_location = np.array([y, x])
                heuristic_matrix[y, x] = np.linalg.norm(curr_location - self.home_location)
        return heuristic_matrix

    def compute_next_action(self, percepts):
        '''
        Computes the next best action to take give the percepts. 
        Percepts not used for now, picks the action at random
        '''
        if percepts["glitter"] and not self.has_gold:
            action = "grab"
            self.has_gold = True
        elif self.location == (0,0) and self.has_gold:
            action = "climb"
        elif self.location != (0,0) and self.has_gold:
            action = self.navigate_back()
        else:
            actions_to_choose_from = game.actions.copy()
            actions_to_choose_from.remove('grab')
            actions_to_choose_from.remove('climb') 
            action = random.choice(actions_to_choose_from)
        return action
    
    def update_location(self, action):
        '''
        Updates the agent believed location given an action
        '''
        agent_x, agent_y = self.location
        if action == "forward":
            # Only update the location if this is a forward motion (right/left motion don't change the agent location)
            if self.heading == "right":
                new_x = agent_x + 1
                new_y = agent_y
                if new_x > (self.game.width - 1):
                    new_x = agent_x
            elif self.heading == "left":
                new_x = agent_x - 1
                new_y = agent_y
                if new_x < 0:
                    new_x = agent_x
            elif self.heading == "up":
                new_x = agent_x
                new_y = agent_y + 1
                if new_y > (self.game.height - 1):
                    new_y = agent_y
            else:
                # heading down
                new_x = agent_x
                new_y = agent_y - 1
                if new_y < 0:
                    new_y = agent_y
            # Computing the new agent location
            self.location = (new_x, new_y)
        
    def update_heading(self, action):
        '''
        Updates the heading action after taking a right or left turn
        '''
        new_heading = self.heading
        if action == "turn right":
            # Use headings list to see what is the next heading if agent turns right
            headings = ["right", "down", "left", "up"]
            if self.heading == "up":
                # Going back to beggining of list
                new_heading = "right"
            else:
                new_heading = headings[headings.index(self.heading) + 1]
        elif action == "turn left":
            # Use headings list to see what is the next heading if agent turns left
            headings = ["up", "left", "down", "right"]
            if self.heading == "right":
                # Going back to beggining of list
                new_heading = "up"
            else:
                new_heading = headings[headings.index(self.heading) + 1]
        self.heading = new_heading
    
    def update_safe_locations(self):
        self.safe_locations.add(self.location)

    def update_belief_state(self, action):
        self.update_location(action)
        self.update_heading(action)
        self.update_safe_locations()
    
    def get_adjacent_squares(self):
        '''
        Gets the 4 adjacent squares to the current agent location
        '''
        headings = ["left", "right", "down", "up"]
        agent_x, agent_y = self.location
        left_square = (agent_x - 1, agent_y)
        right_square = (agent_x + 1, agent_y)
        bottom_square = (agent_x, agent_y - 1)
        top_square = (agent_x, agent_y + 1)
        adjacent_squares = [left_square, right_square, bottom_square, top_square]
        return adjacent_squares, headings

    def navigate_back(self):
        heuristic_matrix = self.heurist_matrix.copy()
        adjacent_squares, headings = self.get_adjacent_squares()
        actions = []
        square_costs = np.zeros(4)
        for i, adjacent_square in enumerate(adjacent_squares):
            x, y = adjacent_square
            if 0 <= x < self.game.width and 0 <= y < self.game.height:
                heuristic_cost = heuristic_matrix[adjacent_square]
                cost = 1 if adjacent_square in self.safe_locations else self.infinity_cost
                square_costs[i] = cost + heuristic_cost
            else:
                square_costs[i] = self.infinity_cost
        best_next_box_ix = np.argmin(square_costs)
        desired_heading = headings[best_next_box_ix]
        # Need to go to the best_next_box: First change heading to point there, then go forward
        if self.heading != desired_heading:
            action = "turn right"
        else:
            action = "forward"
        return action
    
        
        
        
        # Need to get the cost of going to the adjacent boxes
            # if adjacent box has Wumpus or Pit, cost = 10000
            # else, cost = 0

    def play_game(self, visualize=False, verbose=False):
        '''
        Plays WumpusWorld
        Visualize(bool) : Specifies whether to display game canvas or not
        Verbose(bool) : Specifies whether to print the actions and percepts to command line
        '''
        if visualize:
            print('======================== GAME CANVAS ========================')
            self.game.visualize_game_canvas()
            print('======================== START GAME ========================\n\n')
    
        while not self.game.terminate_game:
            # Agent queries the environment for percepts
            self.game.get_percepts()
            percepts = self.game.percepts
            print("PERCEPTS: ", percepts) if verbose else ""
            action = agent.compute_next_action(game.percepts)
            print("ACTION: ", action) if verbose else ""
            game.step(action)
            self.update_belief_state(action)
            if self.has_gold:
                print("NAVIGATING BACK")
            game.visualize_game_canvas()
            print('\n\n') if verbose else ""
        print("FINAL PERCEPTS: ", percepts) if verbose else ""



width = 4
height = 4
allow_climb_without_gold = True
pit_prob = 0.2
game = environment.WumpusWorld(width, height, allow_climb_without_gold, pit_prob)
agent = BeelineAgent(game)
agent.play_game(visualize=True, verbose=True)

# # Code to test heading computation and bumps
# game.get_percepts()
# print("PERCEPTS: ", game.percepts)
# action = "turn right"
# print("ACTION: ", action)
# game.step(action)
# game.visualize_game_canvas()

# game.get_percepts()
# print("PERCEPTS: ", game.percepts)
# action = "forward"
# print("ACTION: ", action)
# game.step(action)
# game.visualize_game_canvas()

# game.get_percepts()
# print("PERCEPTS: ", game.percepts)
# action = "turn right"
# print("ACTION: ", action)
# game.step(action)
# game.visualize_game_canvas()

# game.get_percepts()
# print("PERCEPTS: ", game.percepts)
# action = "turn left"
# print("ACTION: ", action)
# game.step(action)
# game.visualize_game_canvas()

# game.get_percepts()
# print("PERCEPTS: ", game.percepts)
# action = "forward"
# print("ACTION: ", action)
# game.step(action)
# game.visualize_game_canvas()

# game.get_percepts()
# print("PERCEPTS: ", game.percepts)

