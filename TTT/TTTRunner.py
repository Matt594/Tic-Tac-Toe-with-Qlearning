import QPlayer
import Player
import TTTRules
import RandomPlayer
import random as r
import time
import pickle
from matplotlib import pyplot as plt
import numpy as np


class TTTRunner:
    # TTT Runner
    # Responsible for assembling all of the classes to execute a game and train
    def __init__(self, epsilon_test=None, alpha_test=None, gamma_test=None):
        # Default hyperparameter value arrays for testing
        if gamma_test is None:
            gamma_test = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
        if alpha_test is None:
            alpha_test = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
        if epsilon_test is None:
            epsilon_test = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
        self.qp1 = QPlayer.QPlayer(.999, .9, .9, (3 ** 9) + 1, 9, 0)  # Creates a QLearning Agent
        self.qp2 = QPlayer.QPlayer(.999, .9, .9, (3 ** 9) + 1, 9, 1)  # Creates a QLearning Agent
        self.rp = RandomPlayer.RandomPlayer('Random')  # Creates a Random Agent
        self.ttt = TTTRules.TTTRules()  # Applies the tic-tac-toe rules
        self.p = Player.Player('Player')  # Creates a real player
        # Creates hyperparameter test arrays
        self.e = epsilon_test
        self.a = alpha_test
        self.g = gamma_test

    # def main():

    def play(self):
        # Sets up a game against a real player and a QLearning agent
        print('Starting game!')
        self.ttt.reset()  # Resets the game
        if r.randint(0, 1) == 0:  # Makes a random integer to determine if the player or agent goes first
            for i in range(0, 9):  # Makes a loop that runs 9 times which corresponds to the max number of moves
                if i % 2 == 0:  # Separates the turns based off of odd and even integers of the turn number
                    print('Go, ' + self.p.name + '!')  # Prints whose turn it is
                    action = self.player_move()  # Sets the action equal to the player's coordinate
                    self.ttt.mark_board_x(action)  # Marks an X on the board at the player's desired coordinate
                    print('========= Boards =========')
                    print(self.ttt.last_board)  # Prints the last board
                    print(self.ttt.board)  # Prints the current board
                    if self.ttt.check_win(self.p.name):  # If the move was a winning move, the game ends
                        self.p.wins += 1
                        break
                if i % 2 == 1:  # Separates the turns based off of odd and even integers of the turn number
                    print('Go, QLearner!')  # Prints whose turn it is
                    action = self.qp2.greedy_move(self.ttt.board)  # Sets the action equal to the agent's coordinate
                    print(self.qp2.q.q[self.ttt.board_to_int(self.ttt.board)],
                          action)  # Prints the QMatrix at the state
                    while self.ttt.illegal_move_check3(self.ttt.mark_board_o_loop(action)[0],
                                                       self.ttt.mark_board_o_loop(action)[1]):  # While the move
                        # chosen is illegal, it loops through other moves until it makes a move
                        action = self.qp2.choose_move(self.ttt.board)  # Sets the action equal to the agent's coordinate
                        self.ttt.mark_board_o_loop(action)  # Marks an X on the board at the agent's desired coordinate
                    self.ttt.mark_board_o(action)  # Marks an O on the board at the agent's desired coordinate
                    print('========= Boards =========')
                    print(self.ttt.last_board)  # Prints the last board
                    print(self.ttt.board)  # Prints the current board
                    if self.ttt.check_win('QLearner'):  # If the move was a winning move, the game ends
                        self.qp2.wins += 1
                        break
        else:  # This portion of the code does everything the previous code does, except the agent goes first
            for i in range(0, 9):  # Makes a loop that runs 9 times which corresponds to the max number of moves
                if i % 2 == 0:  # Separates the turns based off of odd and even integers of the turn number
                    print('Go, QLearner!')  # Prints whose turn it is
                    action = self.qp1.greedy_move(self.ttt.board)  # Sets the action equal to the agent's coordinate
                    print(self.qp1.q.q[self.ttt.board_to_int(self.ttt.board)],
                          action)  # Prints the QMatrix at the state
                    while self.ttt.illegal_move_check3(self.ttt.mark_board_x_loop(action)[0],
                                                       self.ttt.mark_board_x_loop(action)[1]):  # While the move
                        # chosen is illegal, it loops through other moves until it makes a move
                        action = self.qp1.choose_move(self.ttt.board)  # Sets the action equal to the agent's coordinate
                        self.ttt.mark_board_x_loop(action)  # Sets the action equal to the agent's coordinate again
                    self.ttt.mark_board_x(action)  # Marks an X on the board at the agent's desired coordinate
                    print('========= Boards =========')
                    print(self.ttt.last_board)  # Prints the last board
                    print(self.ttt.board)  # Prints the current board
                    if self.ttt.check_win('QLearner'):  # If the move was a winning move, the game ends
                        self.qp2.wins += 1
                        break
                if i % 2 == 1:
                    print('Go, ' + self.p.name + '!')  # Prints whose turn it is
                    action = self.player_move()  # Sets the action equal to the player's coordinate
                    self.ttt.mark_board_o(action)  # Marks an O on the board at the agent's desired coordinate
                    print('========= Boards =========')
                    print(self.ttt.last_board)  # Prints the last board
                    print(self.ttt.board)  # Prints the current board
                    if self.ttt.check_win(self.p.name):  # If the move was a winning move, the game ends
                        self.p.wins += 1
                        break
        print(self.p.name + ' has won ' + str(self.p.wins) + ' time(s)!')
        print('QLearner has won ' + str(self.qp1.wins+self.qp2.wins) + ' time(s)!')
        print('Do you want to play again?')
        if input('y/n: ') == 'y':
            #  Inquiries the player whether or not they want to play again or not
            self.play()

    def load(self): # Loads the previous QMatrices made from the last training session
        self.qp1.q.q = pickle.load(open('qp1matrix', 'rb'))
        self.qp2.q.q = pickle.load(open('qp2matrix', 'rb'))

    def player_move(self):
        # Queries the player for a move and checks if it is a valid move or not
        lboard = self.ttt.board.reshape(9)  # Converts the board into a "linear board" (lboard)
        while True:  # Loop that executes until broken
            try:  # Tries the next lines of code
                action = self.p.choose_move()  # Sets the action equal to the player's coordinate
                if lboard[action] == '':  # If the space at the coordinate is empty, the action is returned as valid
                    return action
                else:  # If the space is not empty, the player is notified and is queried for another move
                    print('That spot has been taken already.')
                    continue
            except(ValueError):  # If there is a value error (not in range 0-8 or mathematical infeasible),
                # the player is notified and is queried for another move
                print('The coordinate you entered was invalid.')
                continue

    def train(self, episodes):
        # Ascribe values to ties ****
        # Update on separate turns after the first move of each learner before the second move ****
        qvalues = []
        qvalues2 = []
        wins = []
        # Trains two QLearning agents
        print('Training QLearner...')
        start = time.time()  # Records the start time
        for i in range(episodes):  # Trains the agents for x amount of episodes
            if i % 1000 == 0:  # Every 1000 episodes, the number of episodes is printed
                print('Episode Number:', i)
            self.ttt.reset()  # Resets the game
            self.qp1.q.reset(0)
            for j in range(0, 9):  # Makes a loop that runs 9 times which corresponds to the max number of moves
                r1 = 0  # Sets Q1's reward equal to zero
                r2 = 0  # Sets Q2's reward equal to zero
                if j % 2 == 0:  # Separates the turns based off of odd and even integers of the turn number
                    if j > 1:
                        self.qp1.q.query(self.ttt.board_to_int(self.ttt.board), 0)  # Q1 is updated first
                    action = self.qp1.choose_move(self.ttt.board)  # Sets the action equal to the agent's coordinate
                    while self.ttt.illegal_move_check3(self.ttt.mark_board_x_loop(action)[0],
                                                       self.ttt.mark_board_x_loop(action)[1]):  # While the move
                        # chosen is illegal, it loops through other moves until it makes a move
                        action = self.qp1.choose_move(self.ttt.board)  # Sets the action equal to the agent's coordinate
                        self.ttt.mark_board_x_loop(action)  # Sets the action equal to the agent's coordinate again
                    self.ttt.mark_board_x(action)  # Marks an X on the board at the agent's desired coordinate
                    if j == 0:  # Initial state of Q2 can only be determined before turn 2
                        self.qp2.q.state = self.ttt.board_to_int(self.ttt.board)  # Sets the state being updated by Q2
                        qp2_initial_state = self.qp2.q.state
                    # equal to the state determined by Q1
                    if self.ttt.training_check_win():  # If the move is a winning move, reward for Q1 is set to 100
                        r1 = 100
                        r2 = -100
                        wins.append(1)
                        self.qp1.q.query(self.ttt.board_to_int(self.ttt.last_board), r1)
                        self.qp2.q.query(self.ttt.board_to_int(self.ttt.board), r2)
                        break
                if j % 2 != 0:  # Separates the turns based off of odd and even integers of the turn number
                    if j > 1:
                        self.qp2.q.query(self.ttt.board_to_int(self.ttt.board), 0)
                    action = self.qp2.choose_move(self.ttt.board)  # Sets the action equal to the agent's coordinate
                    while self.ttt.illegal_move_check3(self.ttt.mark_board_o_loop(action)[0],
                                                       self.ttt.mark_board_o_loop(action)[1]):  # While the move
                        # chosen is illegal, it loops through other moves until it makes a move
                        action = self.qp2.choose_move(self.ttt.board)  # Sets the action equal to the agent's coordinate
                        self.ttt.mark_board_o_loop(action)  # Marks an X on the board at the agent's desired coordinate
                    self.ttt.mark_board_o(action)  # Marks an O on the board at the agent's desired coordinate
                    if self.ttt.training_check_win():  # If the move is a winning move, reward for Q2 is set to 100
                        r2 = 100
                        r1 = -100
                        wins.append(0)
                        self.qp2.q.query(self.ttt.board_to_int(self.ttt.last_board), r2)
                        self.qp1.q.query(self.ttt.board_to_int(self.ttt.board), r1)
                        break
                if (r1 == 0 or r2 == 0) and j == 8:
                    self.qp2.q.query(self.ttt.board_to_int(self.ttt.last_board), 10)
                    self.qp1.q.query(self.ttt.board_to_int(self.ttt.board), 10)
                    break
            qvalues.append(np.max(self.qp1.q.q[0]))
            qvalues2.append(np.max(self.qp2.q.q[qp2_initial_state]))
        print('Training complete')
        end = time.time()  # Records the end time
        print('========= QP1 =========')
        print(self.qp1.q.q)  # Prints the Q1 QMatrix
        with open('qp1matrix', 'wb') as f:  # Stores the Q1 matrix in qp1matrix
            pickle.dump(self.qp1.q.q, f)
            f.close()
        #print(pickle.load(open('qp1matrix', 'rb')))
        print('========= QP2 =========')
        print(self.qp2.q.q)  # Prints the Q2 QMatrix
        with open('qp2matrix', 'wb') as f:  # Stores the Q2 matrix in qp2matrix
            pickle.dump(self.qp2.q.q, f)
            f.close()
        #print(pickle.load(open('qp2matrix', 'rb')))
        plt.plot(qvalues)
        plt.plot(qvalues2)
        plt.show()
        print('Minutes elapsed: ' + str((end - start) / 60))  # Prints the time elapsed in minutes

    def train_optimize(self, episodes, mode):
        # Ascribe values to ties ****
        # Update on separate turns after the first move of each learner before the second move ****
        # Initial Q Value arrays are initiated to plot
        qvalues = []
        qvalues2 = []
        # Trains two QLearning agents
        print('Training QLearner...')
        start = time.time()  # Records the start time
        # Hyperparameters are tested given the default value arrays
        if mode == 1:
            hyperparameters = self.e  # Hyperparameters array is set to epsilon array
            type = 'Epsilon '  # Epsilon is used in all the text
        if mode == 2:
            hyperparameters = self.a  # Hyperparameters array is set to alpha array
            type = 'Alpha '  # Alpha is used in all the text
        if mode == 3:
            hyperparameters = self.g  # Hyperparameters array is set to gamma array
            type = 'Gamma '  # Gamma is used in all the text
        for y in range(len(hyperparameters)):  # QValues are made into 2D arrays to graph different arrays on plots
            qvalues.append([])
            qvalues2.append([])
        for x in range(len(hyperparameters)):  # Cycles through all of the hyperparameters in the array to test
            print(type + str(x+1) + ' out of ' + str(len(hyperparameters)))
            if mode == 1:  # Sets the agent's epsilon equal to the loop's corresponding hyperparameter
                self.qp1 = QPlayer.QPlayer(hyperparameters[x], .9, .9, (3 ** 9) + 1, 9, 0)
                self.qp2 = QPlayer.QPlayer(hyperparameters[x], .9, .9, (3 ** 9) + 1, 9, 1)
            if mode == 2:  # Sets the agent's alpha equal to the loop's corresponding hyperparameter
                self.qp1 = QPlayer.QPlayer(.9, hyperparameters[x], .9, (3 ** 9) + 1, 9, 0)
                self.qp2 = QPlayer.QPlayer(.9, hyperparameters[x], .9, (3 ** 9) + 1, 9, 1)
            if mode == 3:  # Sets the agent's gamma equal to the loop's corresponding hyperparameter
                self.qp1 = QPlayer.QPlayer(.9, .9, hyperparameters[x], (3 ** 9) + 1, 9, 0)
                self.qp2 = QPlayer.QPlayer(.9, .9, hyperparameters[x], (3 ** 9) + 1, 9, 1)
            for i in range(episodes):  # Trains the agents for x amount of episodes
                if i % 1000 == 0:  # Every 1000 episodes, the number of episodes is printed
                    print('Episode Number:', i)
                self.ttt.reset()  # Resets the game
                self.qp1.q.reset(0)
                for j in range(0,
                               9):  # Makes a loop that runs 9 times which corresponds to the max number of moves
                    r1 = 0  # Sets Q1's reward equal to zero
                    r2 = 0  # Sets Q2's reward equal to zero
                    if j % 2 == 0:  # Separates the turns based off of odd and even integers of the turn number
                        if j > 1:
                            self.qp1.q.query(self.ttt.board_to_int(self.ttt.board), 0)  # Q1 is updated first
                        action = self.qp1.choose_move(
                            self.ttt.board)  # Sets the action equal to the agent's coordinate
                        while self.ttt.illegal_move_check3(self.ttt.mark_board_x_loop(action)[0],
                                                           self.ttt.mark_board_x_loop(action)[
                                                               1]):  # While the move
                            # chosen is illegal, it loops through other moves until it makes a move
                            action = self.qp1.choose_move(self.ttt.board)  # Sets the action equal to the agent's
                            # coordinate
                            self.ttt.mark_board_x_loop(action)  # Sets the action equal to the agent's coordinate again
                        self.ttt.mark_board_x(action)  # Marks an X on the board at the agent's desired coordinate
                        if j == 0:  # Initial state of Q2 can only be determined before turn 2
                            self.qp2.q.state = self.ttt.board_to_int(self.ttt.board)  # Sets the state being updated
                            # by Q2
                            qp2_initial_state = self.qp2.q.state
                        # equal to the state determined by Q1
                        if self.ttt.training_check_win():  # If the move is a winning move, reward for Q1 is set to 100
                            r1 = 100
                            r2 = -100
                            self.qp1.q.query(self.ttt.board_to_int(self.ttt.last_board), r1)
                            self.qp2.q.query(self.ttt.board_to_int(self.ttt.board), r2)
                            break
                    if j % 2 != 0:  # Separates the turns based off of odd and even integers of the turn number
                        if j > 1:
                            self.qp2.q.query(self.ttt.board_to_int(self.ttt.board), 0)
                        action = self.qp2.choose_move(self.ttt.board)  # Sets the action equal to the agent's coordinate
                        while self.ttt.illegal_move_check3(self.ttt.mark_board_o_loop(action)[0],
                                                           self.ttt.mark_board_o_loop(action)[1]):  # While the move
                            # chosen is illegal, it loops through other moves until it makes a move
                            action = self.qp2.choose_move(self.ttt.board)  # Sets the action equal to the agent's
                            # coordinate
                            self.ttt.mark_board_o_loop(action)  # Marks an X on the board at the agent's desired
                            # coordinate
                        self.ttt.mark_board_o(action)  # Marks an O on the board at the agent's desired coordinate
                        if self.ttt.training_check_win():  # If the move is a winning move, reward for Q2 is set to 100
                            r2 = 100
                            r1 = -100
                            self.qp2.q.query(self.ttt.board_to_int(self.ttt.last_board), r2)
                            self.qp1.q.query(self.ttt.board_to_int(self.ttt.board), r1)
                            break
                    if (r1 == 0 or r2 == 0) and j == 8:
                        self.qp2.q.query(self.ttt.board_to_int(self.ttt.last_board), 10)
                        self.qp1.q.query(self.ttt.board_to_int(self.ttt.board), 10)
                        break
                qvalues[x].append(np.max(self.qp1.q.q[0]))
                qvalues2[x].append(np.max(self.qp2.q.q[qp2_initial_state]))
            print('Training complete')
            end = time.time()  # Records the end time
            print('========= QP1 =========')
            print(self.qp1.q.q)  # Prints the Q1 QMatrix
            with open('qp1matrix', 'wb') as f:  # Stores the Q1 matrix in qp1matrix
                pickle.dump(self.qp1.q.q, f)
                f.close()
            # print(pickle.load(open('qp1matrix', 'rb')))
            print('========= QP2 =========')
            print(self.qp2.q.q)  # Prints the Q2 QMatrix
            with open('qp2matrix', 'wb') as f:  # Stores the Q2 matrix in qp2matrix
                pickle.dump(self.qp2.q.q, f)
                f.close()
            # print(pickle.load(open('qp2matrix', 'rb')))
            # Organizes the save file to different directories to save the data later
            if mode == 1:
                dr = '/Users/josephescobar/Desktop/QLearning Test Data/Epsilon Tests/Epsilon Tests ' + str(x)
            if mode == 2:
                dr = '/Users/josephescobar/Desktop/QLearning Test Data/Alpha Tests/Alpha Tests ' + str(x)
            if mode == 3:
                dr = '/Users/josephescobar/Desktop/QLearning Test Data/Gamma Tests/Gamma Tests ' + str(x)
            plt.title(type + str(hyperparameters[x]))  # Sets the plot title to the type and hyperparameter value
            plt.xlabel('Episodes')  # Sets the x axis equal to episodes
            plt.ylabel('Initial Q Value')  # Sets the y axis equal to initial Q value
            plt.ylim(0, 1000)  # Sets the y limit to 1000
            plt.plot(qvalues[x])  # Plots the q values of agent 1
            plt.plot(qvalues2[x])  # Plots the q values of agent 2
            plt.savefig(dr)  # Saves the plot into a directory
            plt.gcf().clear()  # Clears the plot
            print('Minutes elapsed: ' + str((end - start) / 60))  # Prints the time elapsed in minutes

    def random_train(self, episodes):
        wins = []
        # Trains two QLearning agents
        print('Training QLearner...')
        start = time.time()  # Records the start time
        for i in range(episodes):  # Trains the agents for x amount of episodes
            if i % 1000 == 0:  # Every 1000 episodes, the number of episodes is printed
                print('Episode Number:', i)
            self.ttt.reset()  # Resets the game
            self.qp1.q.reset(0)  # Resets the initial state and action of the agent
            coin = r.randint(0, 1)
            if coin == 0:
                for j in range(0, 9):  # Makes a loop that runs 9 times which corresponds to the max number of moves
                    r1 = 0  # Sets Q1's reward equal to zero
                    r2 = 0  # Sets Q2's reward equal to zero
                    if j % 2 == 0:  # Separates the turns based off of odd and even integers of the turn number
                        action = self.rp.choose_move(self.ttt.board)
                        self.ttt.mark_board_x(action)  # Marks an X on the board at the agent's desired coordinate
                        if j < 2:  # Initial state of Q2 can only be determined before turn 2
                            self.qp2.q.state = self.ttt.board_to_int(
                                self.ttt.board)  # Sets the state being updated by Q2
                        # equal to the state determined by Q1
                        if self.ttt.training_check_win():  # If the move is a winning move, reward for Q1 is set to 100
                            r1 = 100
                            r2 = -100
                            wins.append(1)
                    if j % 2 != 0:  # Separates the turns based off of odd and even integers of the turn number
                        action = self.qp2.choose_move(
                            self.ttt.board)  # Sets the action equal to the agent's coordinate
                        while self.ttt.illegal_move_check3(self.ttt.mark_board_o_loop(action)[0],
                                                           self.ttt.mark_board_o_loop(action)[1]):  # While the move
                            # chosen is illegal, it loops through other moves until it makes a move
                            action = self.qp2.choose_move(self.ttt.board)  # Sets the action equal to the agent's
                            # coordinate
                            self.ttt.mark_board_o_loop(
                                action)  # Marks an O on the board at the agent's desired coordinate
                        self.ttt.mark_board_o(action)  # Marks an O on the board at the agent's desired coordinate
                        if self.ttt.training_check_win():  # If the move is a winning move, reward for Q2 is set to 100
                            r2 = 100
                            r1 = -100
                            wins.append(0)
                    if j > 2:  # Q2 can only be updated after Q1 because it always goes after
                        self.qp2.q.query(self.ttt.board_to_int(self.ttt.last_board), r2)
                    if r1 == 100 or r2 == 100:  # If either of the rewards are 100, the game breaks and restarts
                        break
            else:
                for j in range(0, 9):  # Makes a loop that runs 9 times which corresponds to the max number of moves
                    r1 = 0  # Sets Q1's reward equal to zero
                    r2 = 0  # Sets Q2's reward equal to zero
                    if j % 2 == 0:  # Separates the turns based off of odd and even integers of the turn number
                        action = self.qp1.choose_move(
                            self.ttt.board)  # Sets the action equal to the agent's coordinate
                        while self.ttt.illegal_move_check3(self.ttt.mark_board_x_loop(action)[0],
                                                           self.ttt.mark_board_x_loop(action)[1]):  # While the move
                            # chosen is illegal, it loops through other moves until it makes a move
                            action = self.qp1.choose_move(self.ttt.board)  # Sets the action equal to the agent's
                            # coordinate
                            self.ttt.mark_board_x_loop(action)  # Sets the action equal to the agent's coordinate again
                        self.ttt.mark_board_x(action)  # Marks an X on the board at the agent's desired coordinate
                        if self.ttt.training_check_win():  # If the move is a winning move, reward for Q1 is set to 100
                            r1 = 100
                            r2 = -100
                            wins.append(1)
                    if j % 2 != 0:  # Separates the turns based off of odd and even integers of the turn number
                        action = self.rp.choose_move(self.ttt.board)
                        self.ttt.mark_board_o(action)  # Marks an O on the board at the agent's desired coordinate
                        if self.ttt.training_check_win():  # If the move is a winning move, reward for Q2 is set to 100
                            r2 = 100
                            r1 = -100
                            wins.append(0)
                    self.qp1.q.query(self.ttt.board_to_int(self.ttt.board), r1)  # Q1 is updated first
                    if r1 == 100 or r2 == 100:  # If either of the rewards are 100, the game breaks and restarts
                        break
        print('Training complete')
        end = time.time()  # Records the end time
        print('========= QP1 =========')
        print(self.qp1.q.q)  # Prints the Q1 QMatrix
        with open('qp1matrix', 'wb') as f:  # Stores the Q1 matrix in qp1matrix
            pickle.dump(self.qp1.q.q, f)
            f.close()
        # print(pickle.load(open('qp1matrix', 'rb')))
        print('========= QP2 =========')
        print(self.qp2.q.q)  # Prints the Q2 QMatrix
        with open('qp2matrix', 'wb') as f:  # Stores the Q2 matrix in qp2matrix
            pickle.dump(self.qp2.q.q, f)
            f.close()
        # print(pickle.load(open('qp2matrix', 'rb')))
        plt.plot(wins)
        plt.show()
        print('Minutes elapsed: ' + str((end - start) / 60))  # Prints the time elapsed in minutes



ttt = TTTRunner()  # Creates a TTT Runner
ttt.train(70000)  # Trains two QLearners for x amount of games
#ttt.train_optimize(60000, 3)

#ttt.load()  # Loads the previous QMatrices made from the last training session
ttt.play()  # Makes a game where the player fights the two agents
