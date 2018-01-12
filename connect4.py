import numpy as np

# define a class for the game
class connect4:
    def __init__ (self):
        self.width = 5
        self.height = 5

        self.board = np.zeros((self.height, self.width), dtype=np.int)
        self.players_turn = np.random.randint(2)

        self.game_complete = False

    def AddMove(self, column):
        # if the move is invalid
        if (column < 0) or (column > (self.height-1)):
            # return that the move was not added to the board
            # and there is no winner
            return (False, None)

        # search from the bottom to the top of the board for an open spot
        for row in range(self.height-1, -1, -1):

            # if an open spot is found
            if self.board[row, column] == 0:
                # add the move to the board
                self.board[row, column] = 1 if (self.players_turn == 1) else -1

                # check if a player won on the most recent move
                if self.checkWin():
                    # return that the move was added to the board
                    # return that a player won
                    return (True, True)

                # switch the players_turn
                self.players_turn ^= 1

                # return move was added to board
                # return that no player won
                return (True, None)

        # return that the move was not added to the board
        # return that no player has won
        return (False, None)

    def checkWin(self):
        boardHeight = self.height
        boardWidth = self.width
        tile = 1 if (self.players_turn == 1) else -1
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
