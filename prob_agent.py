###############################################################################################################
#                                             Mena S.A. Kamel
#                           3547-014: Intelligent Agents and Reinforcement Learning
#                                              Assignment #2
#                                             February 23, 2022
# Beeline agent uses an A* algorithm to navigate back to the home location when the Gold is picked up
###############################################################################################################

from stringprep import b3_exceptions
import numpy as np
import environment
import random
from pomegranate import *
import itertools

class ProbAgent:
    def __init__(self, game):
        '''
        Initializes a Probability-Based Agent
        '''
        self.game = game
        self.home_location = (0, 0)

        # Initializing Agent belief state
        self.location = self.home_location
        self.heading = "right"
        self.has_gold = False
        self.has_arrow = True
        self.safe_locations = set([self.location])
        self.breezes = {}
        self.stenches = {}

        self.stench_wumpus_model, self.stench_wumpus_states = self.create_stench_wumpus_model()

        self.breeze_pit_model, self.breeze_pit_states = self.create_breeze_pit_model()

        self.risk_thresh = 0.49

        # Defining A* algorithm parameters
        # Defining infinity cost as an integer approximation
        self.infinity_cost = 999999

        # Heuristic used is the Euclidean distance. heurist_matrix is a matrix storing
        # the Euclidean distance from each box to the home_location
        self.heurist_matrix = self.compute_heuristic_matrix()

        # Representing headings as angles
        self.heading_angles = {"up": 0, "right": 90, "down": 180, "left": 270}

    def create_breeze_pit_model(self):
        '''
        Creates the breeze-pit model
        '''
        # Setting the pit locations to have a discrete disctribution with p(pit=True) = 0.2
        pits = {}
        pit_probability = 0.2
        for x in list(range(1,5)):
            for y in list(range(1,5)):
                if [x,y] == [1,1]:
                    continue
                pits["pit_{}_{}".format(x,y)] = DiscreteDistribution({"True": pit_probability, "False": 1-pit_probability})

        # Setting the breezes to be dependent on the pits
        breezes = {}
        edges = []
        # Looping through all box locations
        for x in list(range(1,5)):
            for y in list(range(1,5)):
                # Getting the adjacent boxes and adding the current location to it
                affected_boxes = self.get_surrounding_boxes([x, y])
                affected_boxes.append((x, y))
                state_changes = []
                name = "{}_{}".format(x,y)
                source_pits = []
                source_pit_distribution = []
                # Looping through the adjacent boxes -- boxes that might have a pit that might cause a breeze in the current box
                for affected_box in affected_boxes:
                    i, j = affected_box
                    # Skipping adjacent boxes out of game canvas bounds
                    if not 1<=i<=4 or not 1<=j<=4 or [i,j] == [1,1]:
                        continue
                    source_pits.append((i,j))
                    source_pit_name = "pit_{}_{}".format(i,j)
                    source_pit_distribution.append(pits[source_pit_name])
                    # Adding the edge to the list of edges [source pit, breeze at current location]
                    edges.append([source_pit_name, "breeze_{}_{}".format(x,y)])
                state_changes = []

                # Getting all the possible combinations of pits
                pit_combos = list(itertools.product(["True","False"], repeat=len(source_pits)))
                for pit_combo in pit_combos:
                    if 'True' in pit_combo:
                        state_changes.append(list(pit_combo) + ["True", 1.0])
                        state_changes.append(list(pit_combo) + ["False", 0.0])
                    else:
                        state_changes.append(list(pit_combo) + ["False", 1.0])
                        state_changes.append(list(pit_combo) + ["True", 0.0])
                # Defining the Conditional Probability Table
                breezes["breeze_{}_{}".format(x,y)] = ConditionalProbabilityTable(state_changes, source_pit_distribution)
        # Creating the pit states for the Bayesian model
        pit_state_names = []
        pit_states = {}
        for pit in pits.keys():
            pit_states[pit] = State(pits[pit], name=pit)
            pit_state_names.append(pit)

        #  Creating the breeze states for the Bayesian model
        breeze_state_names = []
        breeze_states = {}
        for breeze in breezes.keys():
            breeze_states[breeze] = State(breezes[breeze], name=breeze)
            breeze_state_names.append(breeze)

        # Creating a bayesian network
        pit_breeze_model = BayesianNetwork("Pit_Breeze")
        states = list(breeze_states.values()) + list(pit_states.values())
        state_names = breeze_state_names + pit_state_names
        # Adding the states to the model
        pit_breeze_model.add_states(*states)
        # Defining the edges of the model
        for edge in edges:
            pit_state = pit_states[edge[0]]
            breeze_state = breeze_states[edge[1]]
            pit_breeze_model.add_edge(pit_state, breeze_state)
        # Baking the model
        pit_breeze_model.bake()
        return pit_breeze_model, state_names

    def create_stench_wumpus_model(self):
        '''
        Creates the wumpus-stench model
        '''
        # Defining the initial wumpus probability
        num_boxes = 15.
        wumpus = DiscreteDistribution({
        "wumpus_1_2": 1/num_boxes, "wumpus_1_3": 1/num_boxes, "wumpus_1_4": 1/num_boxes,
        "wumpus_2_1": 1/num_boxes, "wumpus_2_2": 1/num_boxes, "wumpus_2_3": 1/num_boxes, "wumpus_2_4": 1/num_boxes,
        "wumpus_3_1": 1/num_boxes, "wumpus_3_2": 1/num_boxes, "wumpus_3_3": 1/num_boxes, "wumpus_3_4": 1/num_boxes,
        "wumpus_4_1": 1/num_boxes, "wumpus_4_2": 1/num_boxes, "wumpus_4_3": 1/num_boxes, "wumpus_4_4": 1/num_boxes
        })

        # Defining the relationship between stenches and wumpus
        stenches = {}
        edges = []
        # Looping through all the boxes
        for x in list(range(1,5)):
            for y in list(range(1,5)):
                affected_boxes = self.get_surrounding_boxes([x, y])
                affected_boxes.append((x, y))
                state_changes = []
                # Adding an edge to relate the surrounding possible wumpus' to the current box
                edges.append(["wumpus", "stench_{}_{}".format(x,y)])
                for i in list(range(1,5)):
                    for j in list(range(1,5)):
                        if [i,j] == [1,1]:
                            continue
                        box_name = "wumpus_{}_{}".format(i,j)
                        if (i,j) in affected_boxes:
                            state_changes.append([box_name, "True", 1.0])
                            state_changes.append([box_name, "False", 0.0])
                        else:
                            state_changes.append([box_name, "True", 0.0])
                            state_changes.append([box_name, "False", 1.0])
                # Creating the conditional probability table
                stenches["stench_{}_{}".format(x,y)] = ConditionalProbabilityTable(state_changes, [wumpus])
        # Defining the model states
        stench_state_names = []
        stench_states = {}
        for stench in stenches.keys():
            stench_states[stench] = State(stenches[stench], name=stench)
            stench_state_names.append(stench)
        stench_state_names.append("wumpus")
        wumpus_state = State(wumpus, name="wumpus")

        # Defining the model as a Bayesian Network
        stench_wumpus_model = BayesianNetwork("Wumpus_Stench")
        states = list(stench_states.values()) + [wumpus_state]

        # Adding the states to the model
        stench_wumpus_model.add_states(*states)
        for edge in edges:
            stench_wumpus_model.add_edge(wumpus_state, stench_states[edge[1]])

        # Baking the model
        stench_wumpus_model.bake()
        return stench_wumpus_model, stench_state_names

    def compute_heuristic_matrix(self):
        '''
        Computes the Euclidean distance from each box in the grid to the home location
        '''
        # Initializing the matrix to zeros
        heuristic_matrix = np.zeros((self.game.height, self.game.width))

        # Looping through all the boxes
        for x in range(self.game.width):
            for y in range(self.game.height):
                # Defining the current box we are in
                curr_location = np.array([y, x])

                # Computing the norm / Eulidean distance from the current location to the home location
                heuristic_matrix[y, x] = np.linalg.norm(curr_location - self.home_location)
        return heuristic_matrix

    def compute_next_action(self, percepts):
        '''
        Computes the next best action to take give the percepts.
        Percepts not used for now, picks the action at random
        '''
        # If agent senses glitter and does not have the gold, grab the gold and update the belief state
        if percepts["glitter"] and not self.has_gold:
            action = "grab"
            self.has_gold = True

         # If agent is at the home location and has the gold, climb
        elif self.location == (0,0) and self.has_gold:
            action = "climb"

        # If agent is not at the home location, but has the gold, navigate back using A*
        elif self.location != (0,0) and self.has_gold:
            action = self.navigate_back()

        # Otherwise, pick an action according to the probabilistic model
        else:
            actions_to_choose_from = game.actions.copy()
            actions_to_choose_from.remove('grab')
            actions_to_choose_from.remove('climb')

            # Storing the observed stench and breeze percepts
            x, y = self.location
            if self.percepts['stench']:
                self.stenches["stench_{}_{}".format(x+1, y+1)] = "True"
            if self.percepts['breeze']:
                self.breezes["breeze_{}_{}".format(x+1, y+1)] = "True"

            # Getting the pit and wumpus probabilities using the probabilistic models
            pit_probabilities = self.compute_pit_probability()
            wumpus_probabilities = self.compute_wumpus_probability()

            # Getting the 4 surrounding squares
            adjacent_squares, headings = self.get_adjacent_squares()

            # Creating lists to store the boxes that are safe (risk less than 50%)
            safe_boxes = []
            # Creating lists to store the box total risks (wumpus probability + pit probability)
            total_box_risks = []
            # Creating lists to store the safe box headings
            safe_boxes_headings = []
            p_wumpus = []
            p_pit = []

            # Looping through all the adjacent squares
            for ix, adjacent_square in enumerate(adjacent_squares):
                i, j = adjacent_square
                # Confirming adjacent boxes are within game canvas
                if 0 <= i < self.game.width and 0 <= j < self.game.height:
                    if i == 0 and j == 0:
                        # Hard coding pit and wumpus probability to zero at home location
                        pit_prob = 0
                        wumpus_prob = 0
                    else:
                        # Getting the pit and wumpus probability for the given box
                        pit_prob = pit_probabilities['pit_{}_{}'.format(i+1,j+1)]
                        wumpus_prob = wumpus_probabilities['wumpus_{}_{}'.format(i+1,j+1)]
                    # If the p(Wumpus) < 50 and p(Pit) < 50 and we have not visited the box before (To prevent unnecessarily visiting an old box again)
                    if pit_prob < self.risk_thresh and wumpus_prob < self.risk_thresh and (i, j) not in self.safe_locations:
                        safe_boxes.append((i+1, j+1))
                        total_box_risks.append(pit_prob + wumpus_prob)
                        safe_boxes_headings.append(headings[ix])
                    p_wumpus.append(wumpus_prob)
                    p_pit.append(pit_prob)
            # If no safe boxes have been identified
            if len(safe_boxes) == 0:
                # Check if the arrow is still there
                if self.has_arrow:
                    # Point in the direction of the box with the highest Wumpus probability and fire
                    wumpus_heading = headings[np.argmax(wumpus_probabilities)]
                    if self.heading_angles[self.heading] - self.heading_angles[wumpus_heading] > 0:
                        action = "turn left"
                    elif self.heading_angles[self.heading] - self.heading_angles[wumpus_heading] < 0:
                        action = "turn right"
                    else:
                        print("SHOT ARROW!")
                        action = "shoot"
                        self.has_arrow = False
                # If the arrow has been used, we navigate back
                else:
                    action = self.navigate_back()
            else:
                # If safe locations have been found, go to the safe location
                safest_location = safe_boxes[np.argmin(total_box_risks)]
                safest_heading = safe_boxes_headings[np.argmin(total_box_risks)]
                # Need to go to safest_location
                if self.heading_angles[self.heading] - self.heading_angles[safest_heading] > 0:
                    action = "turn left"
                elif self.heading_angles[self.heading] - self.heading_angles[safest_heading] < 0:
                    action = "turn right"
                else:
                    action = "forward"
        return action

    def compute_pit_probability(self):
        '''
        Computes pit probabilities using self.breeze_pit_model
        '''
        pit_probs = self.breeze_pit_model.predict_proba([self.breezes])
        prob = {}
        for i, entry in enumerate(pit_probs[0]):
            state_name = self.breeze_pit_states[i]
            if 'breeze' in state_name:
                continue
            prob[state_name] = entry.to_dict()["parameters"][0]["True"]
        return prob

    def compute_wumpus_probability(self):
        '''
        Computes Wumpus probabilities using self.stench_wumpus_model
        '''
        stench_probs = self.stench_wumpus_model.predict_proba([self.stenches])
        prob = {}
        for i, entry in enumerate(stench_probs[0]):
            state_name = self.stench_wumpus_states[i]
            if 'stench' in state_name:
                continue
            prob[state_name] = entry.to_dict()["parameters"]
        return prob['wumpus'][0]


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
        '''
        Adding current location to the list of safe locations
        '''
        self.safe_locations.add(self.location)

    def update_belief_state(self, action):
        '''
        Updates the agent belief state
        '''
        # Updating believed location
        self.update_location(action)

        # Updating believed heading
        self.update_heading(action)

        # Updating believed safe locations
        self.update_safe_locations()

    def get_surrounding_boxes(self, location):
        '''
        Gets the 4 adjacent squares to the current agent location
        '''
        agent_x, agent_y = location
        left_square = (agent_x - 1, agent_y)
        right_square = (agent_x + 1, agent_y)
        bottom_square = (agent_x, agent_y - 1)
        top_square = (agent_x, agent_y + 1)
        adjacent_squares = [left_square, right_square, bottom_square, top_square]
        return adjacent_squares

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
        '''
        Function that determines the next action to take in order to navigate
        back to the home location
        '''
        # Getting the heuristic matrix
        heuristic_matrix = self.heurist_matrix.copy()

        # Getting the 4 adjacent boxes to where the agent is in
        adjacent_squares, headings = self.get_adjacent_squares()

        # Defining a list to store the costs of going to the adjacent boxes
        square_costs = np.zeros(4)

        # Looping through the adjacent squares
        for i, adjacent_square in enumerate(adjacent_squares):
            x, y = adjacent_square
            # If the adjacent box is within the grid
            if 0 <= x < self.game.width and 0 <= y < self.game.height:
                # Total cost is the cost of going to the adjacent box PLUS the heuristic cost (distance to home)

                # Getting the heuristic cost
                heuristic_cost = heuristic_matrix[adjacent_square]

                # Getting the cost of going to the adjacent box (1 if it is safe, infinity if it has a Wumpus or Pit)
                cost = 1 if adjacent_square in self.safe_locations else self.infinity_cost
                square_costs[i] = cost + heuristic_cost

            # If the adjacent box is outside the grid, we give a very large cost to stop the agent from going there
            else:
                square_costs[i] = 2*self.infinity_cost

        # Choosing the best box to go to next as the one with the minimum cost
        best_next_box_ix = np.argmin(square_costs)
        # Getting the heading required to go to that desired box
        desired_heading = headings[best_next_box_ix]

        # Need to go to the best_next_box: First change heading to point there, then go forward
        if self.heading_angles[self.heading] - self.heading_angles[desired_heading] > 0:
            action = "turn left"
        elif self.heading_angles[self.heading] - self.heading_angles[desired_heading] < 0:
            action = "turn right"
        else:
            action = "forward"
        print(action)
        return action

    def play_game(self, visualize=False, verbose=False):
        '''
        Plays WumpusWorld
        Visualize(bool) : Specifies whether to display game canvas or not
        Verbose(bool) : Specifies whether to print the actions and percepts to command line
        '''
        num_steps = 0
        if visualize:
            print('======================== GAME CANVAS ========================')
            self.game.visualize_game_canvas()
            print('======================== START GAME ========================\n\n')
        while not self.game.terminate_game:
            # Agent queries the environment for percepts
            self.game.get_percepts()
            self.percepts = self.game.percepts
            # print("PERCEPTS: ", self.percepts) if verbose else ""
            action = agent.compute_next_action(game.percepts)
            # print("ACTION: ", action) if verbose else ""
            game.step(action)
            self.update_belief_state(action)
            if self.has_gold:
                print("NAVIGATING BACK")
            if num_steps >100:
                break
            num_steps += 1
            game.visualize_game_canvas()
            # print('\n\n') if verbose else ""
        # print("TERMINATED GAME")
        # print("FINAL PERCEPTS: ", self.percepts) if verbose else ""
        if self.has_gold and self.location==(0,0):
            return True
        else:
            return False



width = 4
height = 4
allow_climb_without_gold = True
pit_prob = 0.2

total_score = 0
payoff = 1000
num_games = 1000
for k in range(num_games):
    game = environment.WumpusWorld(width, height, allow_climb_without_gold, pit_prob)
    agent = ProbAgent(game)
    # Test logic by setting agent location to gold location at first
    win = agent.play_game(visualize=True, verbose=True)
    if win:
        total_score += payoff
print("TOTAL PAYOFF AFTER 1000 EPISODES: {}".format(total_score))

#TOTAL PAYOFF AFTER 1000 EPISODES: 162000
