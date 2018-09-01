# QLearning Class
# Responsible for creating a QMatrix, updating it, and deciding moves
import numpy as np
import random as r


class QLearner(object):
    def __init__(self, epsilon, alpha, gamma, states, actions, initial_state):
        self.epsilon = epsilon  # Random Action Rate
        self.alpha = alpha  # Learning Rate
        self.gamma = gamma  # Discount Rate
        self.states = states  # Number of states possible in the game
        self.actions = actions  # Number of actions possible per state
        self.q = np.zeros((states, actions), dtype=float)  # QTable
        self.state = initial_state  # State that each learner starts on
        self.action = None  # Action that the learner will update

    #       QLearning Equation:
    #       Q(s,a) <- Q(s,a) + lr[r'+(dr)maxQ(s',a') - Q(s,a)]
    #       Q(s,a) = Current QTable Values
    #       lr = Learning Rate
    #       r' = Reward earned from move
    #       dr = Discount Rate
    #       maxQ(s',a') = Total future reward based on actions in the next state

    def update(self, s1, s2, a, r):
        # QLearning Equation (Not used)
        self.q[s1][a] = self.q[s1][a] + (
                self.alpha * ((r + (self.gamma * np.max(self.q[s2]))) - self.q[s1][a]))
        self.epsilon *= .999999

    def query(self, s2, r):
        # QLearning Equation (Reformatted)
        self.q[self.state][self.action] = self.q[self.state][self.action] + (
                self.alpha * (
                    (r + (self.gamma * np.max(self.q[s2]))) - self.q[self.state][self.action]))  # QLearning Equation
        self.epsilon = max(self.epsilon * .999999, .2)  # Decays Random Action Rate
        self.alpha = max(self.alpha * .999999, .4)
        self.state = s2  # Sets the current state on deck to update

    def reset(self, initial_state):
        # Resets both the initial state and action for every game
        self.state = initial_state
        self.action = None

    def choose_move(self, s):
        # Chooses a move given a state
        if (self.epsilon > r.random()):  # Compare random decimal with random action rate for EXPLORATION
            move = r.randint(0, self.actions - 1)
        else:  # Otherwise EXPLOITATION occurs which picks the best calculated action according to the matrix
            move = self.greedy_move(s)  # Sets the move equal to the move taken form the greedy move
        self.action = move  # Sets the action equal to the move so it can be put in the query function
        return move  # Returns the move

    def greedy_move(self, s):
        # Chooses a greedy move
        move = np.argmax(self.q[s])  # Sets move equal to best calculated action
        return move  # Returns the move
