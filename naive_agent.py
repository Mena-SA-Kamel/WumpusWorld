###############################################################################################################
#                                             Mena S.A. Kamel
#                           3547-014: Intelligent Agents and Reinforcement Learning
#                                              Assignment #1
#                                             February 9, 2022                                   
###############################################################################################################

import numpy as np
import environment
import random

class NaiveAgent:
    def __init__(self, game):
        '''
        Initializes a Naive Agent
        '''
        self.game = game
        # Initial agent heading and location (belief state)
        self.location = (0, 0)
        self.heading = "right"
    
    def compute_next_action(self, percepts):
        '''
        Computes the next best action to take give the percepts. 
        Percepts not used for now, picks the action at random
        '''
        return random.choice(game.actions)

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
            game.visualize_game_canvas()
            print('\n\n') if verbose else ""
        print("FINAL PERCEPTS: ", percepts) if verbose else ""



width = 4
height = 4
allow_climb_without_gold = True
pit_prob = 0.2
game = environment.WumpusWorld(width, height, allow_climb_without_gold, pit_prob)
agent = NaiveAgent(game)
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

