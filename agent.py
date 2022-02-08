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
game.visualize_game_canvas()
print('\n\n')
agent = NaiveAgent(game)
while not game.terminate_game:
    print(game.percepts)
    action = agent.compute_next_action(game.percepts)
    print(action, game.agent_location)
    game.step(action)
    game.visualize_game_canvas()
    print('\n\n')


