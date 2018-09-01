import numpy as np
import random as r


class RandomPlayer(object):
    # Player Class
    # Responsible for creating a real player, retrieving moves from the player, and keeping their statistics
    def __init__(self, name):
        # Keeps track of the player statistics
        self.name = name  # Name
        self.wins = 0  # Wins

    def choose_move(self, board):
        # Chooses a move randomly based off of empty board spaces
        aboard = np.empty(0)  # Creates an empty array to hold potential actions. Action Board
        oboard = board.reshape(9)  # One-Dimensional board
        for i in range (0, 9):  # Scrolls through the board for open spaces and adds the coordinate of an open space
            # to the action board
            if oboard[i] == '':
                aboard = np.append(aboard, i)
        move = aboard[r.randint(0,np.ma.size(aboard)-1)]  # Returns a random move from the action board
        return int(move)
