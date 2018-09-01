import numpy as np
import QLearning as rq


class QPlayer(object):
    # QPlayer Class
    # Responsible for creating a reinforcement agent player, keeping its statistics,
    # and rearranging boards in a usable format for the QLearning class
    def __init__(self, epsilon, alpha, gamma, states, actions, initial_state):
        # Creates a QLearning agent from the QLearning class
        self.q = rq.QLearner(epsilon, alpha, gamma, states, actions, initial_state)
        self.name = 'QLearner'  # Name
        self.wins = 0  # Wins

    def board_to_int(self, board):
        # This method is responsible for making a unique identification number for each state possible in the game. This
        # is done by assigning each space on the board with a 0, 1, or 2 (empty, X, or O) and getting the sum of all the
        # spaces after each space's value is multiplied by 3 to the power of the space's position.
        # ex.  [ 'X', 'O', 'O' ]               [ '1*(3^0)', '2*(3^1)', '2*(3^2)' ]
        # ID = [ ' ', 'X', ' ' ] = The Sum of: [ '0*(3^3)', '1*(3^4)', '0*(3^5)' ] = 1+6+18+0+81+0+0+0+6561 = 6667
        #      [ ' ', ' ', 'X' ]               [ '0*(3^6)', '0*(3^7)', '1*(3^8)' ]
        lboard = board.reshape(9)  # Converts the board into a "linear board" (lboard)
        sboard = np.zeros(9, dtype=int)  # Creates an empty "sum board" (sboard) that will store character values
        sum_id = 0  # Identification number for each state based off of the spaces and characters on the board
        for i in range(0, np.ma.size(lboard)):
            # Sets sum board values equal to integers that correspond with the alphabetical equivalents
            if lboard[i] == '':
                sboard[i] = 0
            if lboard[i] == 'X':
                sboard[i] = 1
            if lboard[i] == 'O':
                sboard[i] = 2
        for i in range(0, np.ma.size(sboard)):
            # Multiplies all of the board values by 3^position
            sboard[i] = sboard[i] * (3 ** i)
        for i in range(0, np.ma.size(sboard)):
            # Adds all of the board values
            sum_id += sboard[i]
        return sum_id  # Returns the ID

    def choose_move(self, board):
        # Configures board in a way so the QLearner can choose a move, then returns the move
        oboard = board.reshape(9)  # One-Dimensional board
        return self.q.choose_move(self.board_to_int(oboard))  # Returns single integer which will later be turned into a move

    def greedy_move(self, board):
        # Configures board in a way so the QLearner can choose a move, then returns the move
        oboard = board.reshape(9)  # One-Dimensional board
        return self.q.greedy_move(self.board_to_int(oboard))  # Returns single integer which will later be turned into a move
