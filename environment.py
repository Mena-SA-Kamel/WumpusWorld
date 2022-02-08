import numpy as np
import os
import random

class WumpusWorld:
    def __init__(self, width, height, allow_climb_without_gold, pit_prob):
        self.height = height
        self.width = width
        self.pit_probability = pit_prob
        self.allow_climb_wout_gold = allow_climb_without_gold
        self.actions = ["forward", "turn left", "turn right", "grab", "shoot", "climb"]
        self.percepts = ["stench", "breeze", "glitter", "bump", "scream"]
        self.action_penalty = -1
        self.start_location = (0, 0)
        self.agent_location = (0, 0)
        self.agent_heading = "right"
        self.agent_alive = True
        self.agent_has_gold = False
        self.agent_has_arrow = True
        self.agent_state = {
            "location": self.agent_location,
            "heading": self.agent_heading,
            "alive": self.agent_alive,
            "has_gold": self.agent_has_gold,
            "has_arrow": self.agent_has_gold
        }
        self.wumpus_alive = True
        self.climb = False
        self.terminate_game = False
        self.num_actions = 0
        self.create_game_canvas()
        self.visualize_game_canvas()
    
    def create_game_canvas(self):
        self.canvas = np.tile("", (self.height, self.width)).tolist()
        wumpus_x = random.randint(0,self.height-1)
        wumpus_y = random.randint(0,self.width-1)
        self.wumpus_location = (wumpus_x, wumpus_y)

        gold_x = random.randint(0,self.height-1)
        gold_y = random.randint(0,self.width-1)
        self.gold_location = (gold_x, gold_y)

        if self.wumpus_location == self.start_location or self.gold_location == self.start_location:
            self.create_game_canvas()
        
        self.pit_locations = []
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) == self.start_location:
                    continue
                create_pit = random.random() <= self.pit_probability
                if create_pit:
                    self.pit_locations.append((x, y))

    def visualize_game_canvas(self):
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) == self.start_location:
                    self.canvas[y][x] = "S"
                    continue
                if (x, y) == self.wumpus_location:
                    self.canvas[y][x] += "W"
                if (x, y) == self.gold_location:
                    self.canvas[y][x] += "G"
                if (x, y) in self.pit_locations:
                    self.canvas[y][x] += "P"
        for y in range(self.height-1, -1, -1):
            row = "|\t"
            for x in range(self.width):
                row += (self.canvas[y][x] + "\t|\t")
            print(row)

    def get_adjacent_squares(self):
        agent_x, agent_y = self.agent_location
        left_square = (agent_x - 1, agent_y)
        right_square = (agent_x + 1, agent_y)
        bottom_square = (agent_x, agent_y - 1)
        top_square = (agent_x, agent_y + 1)
        adjacent_squares = [left_square, right_square, bottom_square, top_square]
        return adjacent_squares
        
    def sense_stench(self):
        adjacent_squares = self.get_adjacent_squares() + [self.agent_location]
        stench = self.wumpus_location in adjacent_squares
        return stench

    def sense_breeze(self):
        adjacent_squares = self.get_adjacent_squares() + [self.agent_location]
        # Have atleast one square in common
        breeze = len(set(self.pit_locations) & set(adjacent_squares)) > 0
        return breeze
    
    def sense_glitter(self):
        return self.agent_location == self.gold_location

    def sense_scream(self):
        return not self.wumpus_alive

    def sense_bump(self):
        # False by default unless forward motion of agent causes collision (logic handled in update_location)
        return False

    def get_percepts(self, action = ""):
        stench = self.sense_stench()
        breeze = self.sense_breeze()
        glitter = self.sense_glitter()
        bump = self.sense_bump(action)
        scream = self.sense_scream()
        reward = self.get_reward()
        self.percepts = {"stench": stench,
                        "breeze": breeze,
                        "glitter": glitter,
                        "bump": bump,
                        "scream": scream,
                        "reward": reward}
        return self.percepts
        # Returns the percepts + the reward
        # import code; code.interact(local=dict(globals(), **locals()))
    
    def get_reward(self):
        reward = 0
        if self.agent_has_gold and self.climb and self.agent_alive:
            reward = 1000
        if not self.agent_alive:
            reward = -1000
        action_penalty = -1 * self.num_actions
        arrow_penalty = -10 * int(not(self.agent_has_arrow))
        final_reward = reward + action_penalty + arrow_penalty
        return final_reward
    
    def update_location(self, action):
        bump = False
        agent_x, agent_y = self.agent_location
        if action == "forward":
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
            self.agent_location = (new_x, new_y)
            self.percepts["bump"] = bump
            self.agent_alive = (self.agent_location != self.wumpus_location) and (self.agent_location not in self.pit_locations)
        
    def update_heading(self, action):
        if action == "turn right":
            headings = ["right", "down", "left", "up"]
            if self.agent_heading == "up":
                new_heading = "right"
            else:
                new_heading = headings[headings.index(self.agent_heading) + 1]
            self.agent_heading = new_heading
        elif action == "turn left":
            headings = ["up", "left", "down", "right"]
            if self.agent_heading == "right":
                new_heading = "up"
            else:
                new_heading = headings[headings.index(self.agent_heading) + 1]
            self.agent_heading = new_heading

    def check_wumpus_life(self):
        wumpus_x, wumpus_y = self.wumpus_location
        agent_x, agent_y = self.agent_location
        if self.agent_heading == "right" and agent_y == wumpus_y and agent_x < wumpus_x:
            self.wumpus_alive = False
            self.percepts["scream"] = True 
        elif self.agent_heading == "left" and agent_y == wumpus_y and agent_x > wumpus_x:
            self.wumpus_alive = False
            self.percepts["scream"] = True
        elif self.agent_heading == "down" and agent_x == wumpus_x and agent_y > wumpus_y:
            self.wumpus_alive = False
            self.percepts["scream"] = True
        elif self.agent_heading == "up" and agent_x == wumpus_x and agent_y < wumpus_y:
            self.wumpus_alive = False
            self.percepts["scream"] = True
        else:
            self.wumpus_alive = True
            self.percepts["scream"] = False

    def step(self, action):
        self.num_actions += 1 
        if action == "forward":
            self.update_location(action)
        elif action in ["turn right", "turn left"]:
            self.update_heading(action)
        elif action == "grab":
            self.agent_has_gold = (self.agent_location == self.gold_location)
        elif action == "shoot":
            self.agent_has_arrow = False
            self.check_wumpus_life()
        elif action == "climb":
            if self.agent_location == self.start_location:
                if self.allow_climb_wout_gold or self.agent_has_gold:
                    self.climb = True
        
        if not self.agent_alive or self.climb:
            self.terminate_game = True
        self.agent_state["location"] = self.agent_location
        self.agent_state["heading"] = self.agent_heading
        self.agent_state["alive"] = self.agent_alive
        self.agent_state["has_gold"] = self.agent_has_gold
        self.agent_state["has_arrow"] = self.agent_has_gold
    


# width = 4
# height = 4
# allow_climb_without_gold = True
# pit_prob = 0.2
# game = WumpusWorld(width, height, allow_climb_without_gold, pit_prob)
# for i in range(50):
#     game.create_game_canvas()
#     game.visualize_game_canvas()
#     print("\n========================================================================")