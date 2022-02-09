import numpy as np
import environment
import random

class NaiveAgent:
    def __init__(self, game):
        self.game = game
        self.location = (0, 0)
        self.heading = "right"
    
    def compute_next_action(self, percepts):
        # Percepts not used for now
        return random.choice(game.actions)

width = 4
height = 4
allow_climb_without_gold = True
pit_prob = 0.2
game = environment.WumpusWorld(width, height, allow_climb_without_gold, pit_prob)
print('=================== GAME CANVAS ===================')
game.visualize_game_canvas()
print('=================== START GAME ===================\n\n')

agent = NaiveAgent(game)
while not game.terminate_game:
    game.get_percepts()
    print("PERCEPTS: ", game.percepts)
    action = agent.compute_next_action(game.percepts)
    print("ACTION: ", action)
    game.step(action)
    game.visualize_game_canvas()
    print('\n\n')

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

