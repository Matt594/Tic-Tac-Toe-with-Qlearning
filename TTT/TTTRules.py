import numpy as np


class TTTRules(object):
    def __init__(self):
        # Keeps track of the current board and last board in one game's history
        self.board = np.zeros((3, 3), dtype=str)  # Current Board
        self.last_board = np.zeros((3, 3), dtype=str)  # Last Board

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

    def illegal_move_check3(self, a, b):
        # This method checks the remaining empty spaces in a board and compares it to another board. If they number of
        # empty spaces are equal, then it signals that an illegal move has been made
        lbec = 0  # "Last Board Empty Count"
        cbec = 0  # "Current Board Empty Count"
        llboard = np.copy(a.reshape(9))  # "Last Linear Board", makes last board 1D
        clboard = np.copy(b.reshape(9))  # "Current Linear Board", makes current board 1D
        for i in range(0, 9):
            # Counts the number of empty spaces for the last board and current board
            if llboard[i] == '':
                lbec += 1
            if clboard[i] == '':
                cbec += 1
        if (lbec == cbec):
            # If the counts are equal, then an illegal move has been made so it returns true
            return True
        else:
            # Otherwise nothing is wrong and it returns false
            return False

    def mark_board_x(self, coordinates):
        # This marks the board with an X permanently
        self.last_board = self.board.copy()  # The last board becomes the current board
        self.board = self.board.reshape(9)  # The current board is reshaped linearly to accommodate a single digit move
        self.board[coordinates] = 'X'  # The current board is marked with an X at the coordinate
        self.board = self.board.reshape(3, 3)  # The current board is reshaped to a 3*3 grid for viewing pleasure

    def mark_board_o(self, coordinates):
        # This marks the board with an O permanently
        self.last_board = self.board.copy()  # The last board becomes the current board
        self.board = self.board.reshape(9)  # The current board is reshaped linearly to accommodate a single digit move
        self.board[coordinates] = 'O'  # The current board is marked with an O at the coordinate
        self.board = self.board.reshape(3, 3)  # The current board is reshaped to a 3*3 grid for viewing pleasure

    def mark_board_x_loop(self, coordinates):
        # This marks a mock board with an X and returns boards a and b to be checked for illegal moves. This is needed
        # to expose the learner to more states by having it redo moves that are illegal
        a = self.board.copy()  # A copy of the current board is made to act as a new last board
        b = self.board.copy().reshape(9)  # A copy of the current board is reshaped linearly to accommodate a single digit move
        b[coordinates] = 'X'  # The linear copy is marked with an X at the coordinate
        b = b.reshape(3, 3)  # The linear copy is reshaped to a 3*3 grid
        return a, b  # Boards a and b are returned to check for illegal moves later

    def mark_board_o_loop(self, coordinates):
        # This marks a mock board with an X and returns boards a and b to be checked for illegal moves. This is needed
        # to expose the learner to more states by having it redo moves that are illegal
        a = self.board.copy()  # A copy of the current board is made to act as a new last board
        b = self.board.copy().reshape(9)  # A copy of the current board is reshaped linearly to accommodate a single digit move
        b[coordinates] = 'O'  # The linear copy is marked with an O at the coordinate
        b = b.reshape(3, 3)  # The linear copy is reshaped to a 3*3 grid
        return a, b  # Boards a and b are returned to check for illegal moves later

    def reset(self):
        # This resets the boards for a new game
        self.board = np.zeros((3, 3), dtype=str)  # The board is wiped clean of markers
        self.last_board = np.zeros((3, 3), dtype=str)  # The last board is also wiped clean of markers

    def check_win(self, winner):
        # Combines all of the methods that check for a win/tie, prints the winner, and returns true that the game is over
        if self.check_win_vertical('O') or self.check_win_horizontal('O') or self.check_win_diagnol('O'):
            # Check for vertical, horizontal, or diagnol wins by the O player
            print(winner + ' has won!')  # Prints the name of the winner
            return True  # Returns true that the game is over
        if self.check_win_vertical('X') or self.check_win_horizontal('X') or self.check_win_diagnol('X'):
            # Check for vertical, horizontal, or diagnol wins by the X player
            print(winner + ' has won!')  # Prints the name of the winner
            return True  # Returns true that the game is over
        if self.check_tie():
            # Checks if the game resulted in a tie
            print('The match is a draw!')  # Prints the outcome of the game
            return True  # Returns true that the game is over

    def training_check_win(self):
        # Combines all of the methods check for a win and returns true if the game is over
        if self.check_win_vertical('O') or self.check_win_horizontal('O') or self.check_win_diagnol('O'):
            # Check for vertical, horizontal, or diagnol wins by the O player
            return True  # Returns true that the game is over
        if self.check_win_vertical('X') or self.check_win_horizontal('X') or self.check_win_diagnol('X'):
            # Check for vertical, horizontal, or diagnol wins by the X player
            return True  # Returns true that the game is over

    def check_win_vertical(self, a):
        # Checks for a win condition in the board's columns
        for i in range(0, 3):
            if np.array_equal(self.board[i, :], [a, a, a]):
                return True

    def check_win_horizontal(self, a):
        # Checks for a win condition in the board's rows
        for i in range(0, 3):
            if np.array_equal(self.board[:, i], [a, a, a]):
                return True

    def check_win_diagnol(self, a):
        # Checks for a win condition in the board's diagnols
        if ((self.board[0, 0] == a and self.board[1, 1] == a and self.board[2, 2] == a) or
                (self.board[0, 2] == a and self.board[1, 1] == a and self.board[2, 0] == a)):
            return True

    def check_tie(self):
        # Checks for a tie game
        if np.all(self.board != ''):
            return True



