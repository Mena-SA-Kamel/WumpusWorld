import numpy as np
import os

class WumpusWorld:
    def __init__(self, grid_size, pit_probability):
        self.grid_size = grid_size
        self.pit_probability = pit_probability
        self.actions = ["forward", "turn left", "turn right", "grab", "shoot", "climb"]
        self.percepts = ["stench", "breeze", "glitter", "bump", "scream"]
        
        import code; code.interact(local=dict(globals(), **locals()))


game = WumpusWorld(4, 0.2)
