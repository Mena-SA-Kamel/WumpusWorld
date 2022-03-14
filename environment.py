###############################################################################################################
#                                             Mena S.A. Kamel
#                           3547-014: Intelligent Agents and Reinforcement Learning
#                                              Assignment #1
#                                             February 9, 2022                                   
###############################################################################################################

import numpy as np
import os
import random

class WumpusWorld:
    def __init__(self, width, height, allow_climb_without_gold, pit_prob):
        '''
        Initializes a WumpusWorld game instance
        width (int): Grid width
        height (int): Grid height
        allow_climb_without_gold (bool): Specifies whether or not the agent is allowed to climb without the gold
        pit_prob (float <= 1): Pit probability 
        '''
        self.height = height
        self.width = width
        self.pit_probability = pit_prob
        self.allow_climb_wout_gold = allow_climb_without_gold
        # Set of allowable actions
        self.actions = ["forward", "turn left", "turn right", "grab", "shoot", "climb"]
        # Percepts vector. Initially set as empty at beginning of game
        self.percepts = {}
        # Default action penalty
        self.action_penalty = -1
        # Start location in code coordinate system
        self.start_location = (0, 0)
        # Agent location in code coordinate system
        self.agent_location = (0, 0)
        # Agent location in formal game convention -- (1, 1) start location
        self.standardized_agent_location = (self.agent_location[0] + 1, self.agent_location[1] + 1)
        # Initial agent heading
        self.agent_heading = "right"
        # Intializing agent state values
        self.agent_alive = True
        self.agent_has_gold = False
        self.agent_has_arrow = True
        self.agent_state = {
            "location": self.agent_location,
            "heading": self.agent_heading,
            "alive": self.agent_alive,
            "has_gold": self.agent_has_gold,
            "has_arrow": self.agent_has_arrow
        }
        # Setting initial Wumpus life
        self.wumpus_alive = True
        # Setting climb to False at the beggining of the game
        self.climb = False
        # Flag specifying if game ended or not
        self.terminate_game = False
        # Number of actions taken so far
        self.num_actions = 0
        # Creating the game canvas
        self.create_game_canvas()
        # Flag used to make sure we penalize arrow shoot once
        self.arrow_penalized = False
    
    def create_game_canvas(self):
        '''
        Creates the game canvas
        '''
        # Canvas is set as an empty numpy array of size[height, width]
        self.canvas = np.tile("", (self.height, self.width)).tolist()
        # Choosing a random Wumpus location
        wumpus_x = random.randint(0,self.height-1)
        wumpus_y = random.randint(0,self.width-1)
        self.wumpus_location = (wumpus_x, wumpus_y)

        # Choosing a random gold location
        gold_x = random.randint(0,self.height-1)
        gold_y = random.randint(0,self.width-1)
        self.gold_location = (gold_x, gold_y)

        # Recreate canvas if Wumpus or Gold at the initial start location
        if self.wumpus_location == self.start_location or self.gold_location == self.start_location:
            self.create_game_canvas()
        
        # Getting pit locations. Locations are stored in self.pit_locations
        self.pit_locations = []
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) == self.start_location:
                    # Skip the initial start location
                    continue
                # Setting pits with probability defined by self.pit_probability
                create_pit = random.random() <= self.pit_probability
                if create_pit:
                    self.pit_locations.append((x, y))

    def visualize_game_canvas(self):
        '''
        Prints the game canvas to command line
        (A*): Agent + heading (<>^˅)
        P: Pit
        G: Gold
        W: Wumpus
        '''
        print("-----------------------------------------------------------------")
        self.canvas = np.tile("", (self.height, self.width)).tolist()
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) == self.agent_location:
                    self.canvas[y][x] += ("(A{})".format(self.get_heading_symbol()))
                if (x, y) == self.start_location:
                    self.canvas[y][x] += "S"
                    continue
                if (x, y) == self.wumpus_location:
                    self.canvas[y][x] += "W"
                if (x, y) == self.gold_location:
                    self.canvas[y][x] += "G"
                if (x, y) in self.pit_locations:
                    self.canvas[y][x] += "P"
        for k in range(self.height-1, -1, -1):
            row = "|\t"
            for j in range(self.width):
                row += (self.canvas[k][j] + "\t|\t")
            print(row)
        print("-----------------------------------------------------------------")
    
    def get_heading_symbol(self):
        '''
        Getting a heading symbol from agent heading (Used for visualizing game canvas)
        '''
        if self.agent_heading == "up":
            return "˄"
        elif self.agent_heading == "down":
                return "˅"
        elif self.agent_heading == "right":
                return ">"
        else: 
            return "<"

    def get_adjacent_squares(self):
        '''
        Gets the 4 adjacent squares to the current agent location
        '''
        agent_x, agent_y = self.agent_location
        left_square = (agent_x - 1, agent_y)
        right_square = (agent_x + 1, agent_y)
        bottom_square = (agent_x, agent_y - 1)
        top_square = (agent_x, agent_y + 1)
        adjacent_squares = [left_square, right_square, bottom_square, top_square]
        return adjacent_squares
        
    def sense_stench(self):
        '''
        Senses stench by checking if a Wumpus is present in the adjacent squares or in the current agent location
        '''
        adjacent_squares = self.get_adjacent_squares() + [self.agent_location]
        stench = self.wumpus_location in adjacent_squares
        return stench

    def sense_breeze(self):
        '''
        Senses breeze by checking if a Pit is present in the adjacent squares or in the current agent location
        '''
        adjacent_squares = self.get_adjacent_squares() + [self.agent_location]
        # Have atleast one square in common
        breeze = len(set(self.pit_locations) & set(adjacent_squares)) > 0
        return breeze
    
    def sense_glitter(self):
        '''
        Senses Gold by checking if the agent is located where the gold is
        '''
        return self.agent_location == self.gold_location

    def sense_scream(self):
        '''
        Senses if the Wumpus is screaming, if agent kills wumpus, wumpus dies a woeful death and screams for the rest of the episode
        '''
        return not self.wumpus_alive

    def sense_bump(self):
        '''
        Senses bump. Logic is handled in the step function and update_location function
        '''
        if "bump" not in self.percepts.keys():
            return False
        else:
            return self.percepts["bump"]

    def get_percepts(self):
        '''
        Populates the Percepts vector by calling all the sensing functions and the reward function
        '''
        stench = self.sense_stench()
        breeze = self.sense_breeze()
        glitter = self.sense_glitter()
        bump = self.sense_bump()
        scream = self.sense_scream()
        reward = self.get_reward()
        self.percepts = {"stench": stench,
                        "breeze": breeze,
                        "glitter": glitter,
                        "bump": bump,
                        "scream": scream,
                        "reward": reward}
        return self.percepts
    
    def check_wumpus_life(self):
        '''
        Checking if the Wumpus is still alive after an arrow is shot
        '''
        wumpus_x, wumpus_y = self.wumpus_location
        agent_x, agent_y = self.agent_location
        if self.agent_heading == "right" and agent_y == wumpus_y and agent_x < wumpus_x:
            # (A>)  W
            self.wumpus_alive = False
            self.percepts["scream"] = True 
        elif self.agent_heading == "left" and agent_y == wumpus_y and agent_x > wumpus_x:
            # W  (<A)
            self.wumpus_alive = False
            self.percepts["scream"] = True
        elif self.agent_heading == "down" and agent_x == wumpus_x and agent_y > wumpus_y:
            # (A˅)
            #  W
            self.wumpus_alive = False
            self.percepts["scream"] = True
        elif self.agent_heading == "up" and agent_x == wumpus_x and agent_y < wumpus_y:
            #  W
            # (A^)
            self.wumpus_alive = False
            self.percepts["scream"] = True
        else:
            # If all conditions not met, then the Wumpus is still alive and hence, not screaming
            self.wumpus_alive = True
            self.percepts["scream"] = False

    def update_location(self, action):
        '''
        Updates the agent location given an action
        '''
        # Handling cases where the agent walks into a wall
        # BUMP PERCEPT IS HANDLED HERE
        bump = False
        agent_x, agent_y = self.agent_location
        if action == "forward":
            # Only update the location if this is a forward motion (right/left motion don't change the agent location)
            if self.agent_heading == "right":
                new_x = agent_x + 1
                new_y = agent_y
                if new_x > (self.width - 1):
                    new_x = agent_x
                    bump = True
            elif self.agent_heading == "left":
                new_x = agent_x - 1
                new_y = agent_y
                if new_x < 0:
                    new_x = agent_x
                    bump = True
            elif self.agent_heading == "up":
                new_x = agent_x
                new_y = agent_y + 1
                if new_y > (self.height - 1):
                    new_y = agent_y
                    bump = True
            else:
                # heading down
                new_x = agent_x
                new_y = agent_y - 1
                if new_y < 0:
                    new_y = agent_y
                    bump = True
            # Computing the new agent location
            self.agent_location = (new_x, new_y)
            # Updating the Bump percept
            self.percepts["bump"] = bump
            # Checking if agent is still alive after walking into a new box
            self.agent_alive = (self.agent_location != self.wumpus_location) and (self.agent_location not in self.pit_locations)
        
    def update_heading(self, action):
        '''
        Updates the heading action after taking a right or left turn
        '''
        if action == "turn right":
            # Use headings list to see what is the next heading if agent turns right
            headings = ["right", "down", "left", "up"]
            if self.agent_heading == "up":
                # Going back to beggining of list
                new_heading = "right"
            else:
                new_heading = headings[headings.index(self.agent_heading) + 1]
        elif action == "turn left":
            # Use headings list to see what is the next heading if agent turns left
            headings = ["up", "left", "down", "right"]
            if self.agent_heading == "right":
                # Going back to beggining of list
                new_heading = "up"
            else:
                new_heading = headings[headings.index(self.agent_heading) + 1]
        self.agent_heading = new_heading
 
    def get_reward(self):
        '''
        Computes the current reward. 
        '''
        reward = 0
        if self.agent_has_gold and self.climb and self.agent_alive:
            reward = 1000
        if not self.agent_alive:
            reward = -1000
        # Penalizing shooting the arrow
        if not(self.agent_has_arrow) and not(self.arrow_penalized):
            reward += -10
            self.arrow_penalized = True
        # Action penalty
        reward += -1
        return reward

    def step(self, action):
        '''
        Main step function that defines how the percepts and agent location/headning update given an action
        '''
        self.num_actions += 1 
        if action == "forward":
            self.update_location(action)
        elif action in ["turn right", "turn left"]:
            self.update_heading(action)
            self.percepts["bump"] = False
        elif action == "grab":
            self.agent_has_gold = (self.agent_location == self.gold_location)
            self.percepts["bump"] = False
        elif action == "shoot":
            self.agent_has_arrow = False
            self.percepts["bump"] = False
            self.check_wumpus_life()
        elif action == "climb":
            if self.agent_location == self.start_location:
                # Check if agent is in the start location
                if self.allow_climb_wout_gold or self.agent_has_gold:
                    # Check if the agent has the gold or if they are allowed to climb without the gold
                    self.climb = True
            self.percepts["bump"] = False
        
        # Moving gold with agent if the agent grabbed it
        if self.agent_has_gold:
            self.gold_location = self.agent_location

        if not self.agent_alive or self.climb:
            # Terminate the game if agent died or climbed
            self.terminate_game = True
        # Updating the new agent state
        self.agent_state["location"] = self.agent_location
        self.agent_state["heading"] = self.agent_heading
        self.agent_state["alive"] = self.agent_alive
        self.agent_state["has_gold"] = self.agent_has_gold
        self.agent_state["has_arrow"] = self.agent_has_arrow
    