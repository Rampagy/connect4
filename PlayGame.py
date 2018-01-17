import connect4 as c4
import HumanPlayer as hp
import numpy as np
import Connect4AI as c4a
import tensorflow as tf
import Connect4Classifier as c4c
import math

# Call this function to play a game of connect4
# Returns a game log of the winning players moves
# the winning players pieces are always represented as a 1
def PlayGame():
    # create instance of the game
    game = c4.connect4()
    game_complete = False

    while not (game_complete):

        valid_input = False
        players_turn = game.players_turn
        board_state = np.copy(game.board)

        while not (valid_input):
            if players_turn == 0:
                # player 0's pieces are represented as -1, so multiply by -1
                # so that from every players perspective their pieces are +1
                move = c4a.ComputerPlayer(np.copy(game.board)*-1)
                #move = hp.HumanAI(np.copy(game.board)*-1)
            else:
                # player 1's pieces are represented as 11, so multiply by 1
                # so that from every players perspective their pieces are +1
                #move = c4a.ComputerPlayer(np.copy(game.board))
                move = hp.HumanAI(np.copy(game.board))

            valid_input, player_won, game_complete = game.AddMove(move)

    return player_won


# play a game against the computer
player_winner = PlayGame()



















def ace_ventura():
    pass
