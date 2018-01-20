import numpy as np

# define a class for the game
class connect4:
    def __init__ (self):
        self.width = 10
        self.height = 10

        self.board = np.zeros((self.height, self.width), dtype=np.int)
        self.players_turn = np.random.randint(2)

        self.previous_players_moves = np.array([-1, -2])


    def AddMove(self, column):
        # Track who has been trying to make moves
        # if the same person has made 3 invalid moves in a row
        # they lose the game
        self.previous_players_moves = np.roll(self.previous_players_moves, 1)
        self.previous_players_moves[0] = self.players_turn

        player_same_turn_count = 0
        for i in range(len(self.previous_players_moves)-1):
            if self.previous_players_moves[i] == self.previous_players_moves[i+1]:
                player_same_turn_count += 1

        if player_same_turn_count == len(self.previous_players_moves)-1:
            # flip winning players turn
            self.players_turn ^= 1
            # return that the move was added to the board
            # return that a player won and the game is complete
            return (True, True, True)


        # if the move is invalid
        if (column < 0) or (column > (self.height-1)):
            # return that the move was not added to the board
            # and there is no winner and the game is not complete
            return (False, False, False)

        # search from the bottom to the top of the board for an open spot
        for row in range(self.height-1, -1, -1):

            # if an open spot is found
            if self.board[row, column] == 0:
                # add the move to the board
                self.board[row, column] = 1 if (self.players_turn == 0) else 2

                # check if a player won on the most recent move
                if self.checkWin():
                    # return that the move was added to the board
                    # return that a player won and the game is complete
                    return (True, True, True)
                elif self.checkTie():
                    # return that the move was added to the board
                    # return that no player won and the game is complete
                    return (True, False, True)

                # switch the players_turn
                self.players_turn ^= 1

                # return move was added to board
                # return that no player won and the game is not complete
                return (True, False, False)

        # return that the move was not added to the board
        # return that no player has won and the game is not complete
        return (False, False, False)

    def checkWin(self):
        boardHeight = self.height
        boardWidth = self.width
        tile = 1 if (self.players_turn == 0) else 2
        board = np.transpose(self.board)

        # check horizontal spaces
        for y in range(boardHeight):
            for x in range(boardWidth - 3):
                if board[x, y] == tile and board[x+1, y] == tile and board[x+2, y] == tile and board[x+3, y] == tile:
                    return True

        # check vertical spaces
        for x in range(boardWidth):
            for y in range(boardHeight - 3):
                if board[x, y] == tile and board[x, y+1] == tile and board[x, y+2] == tile and board[x, y+3] == tile:
                    return True

        # check / diagonal spaces
        for x in range(boardWidth - 3):
            for y in range(3, boardHeight):
                if board[x, y] == tile and board[x+1, y-1] == tile and board[x+2, y-2] == tile and board[x+3, y-3] == tile:
                    return True

        # check \ diagonal spaces
        for x in range(boardWidth - 3):
            for y in range(boardHeight - 3):
                if board[x, y] == tile and board[x+1, y+1] == tile and board[x+2, y+2] == tile and board[x+3, y+3] == tile:
                    return True

        return False


    def checkTie(self):
        col_full = 0

        # check tie
        for x in range(self.width):
            if self.board[0, x] != 0:
                col_full += 1

        if col_full >= self.width:
            return True
        else:
            return False















def dr_doolittle():
    pass
