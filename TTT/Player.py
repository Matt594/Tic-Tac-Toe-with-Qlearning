class Player(object):
    # Player Class
    # Responsible for creating a real player, retrieving moves from the player, and keeping their statistics
    def __init__(self, name):
        # Keeps track of the player statistics
        self.name = name  # Name
        self.wins = 0  # Wins

    def choose_move(self):
        # Chooses a move
        coordinate = eval(input('Enter box number(0-8):'))  # Sets the move equal to a coordinate chosen by the player
        return coordinate  # Returns the move
