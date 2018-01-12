import connect4 as c4
import HumanPlayer as hp
import numpy as np

def PlayGame():
    # create instance of the game
    game = c4.connect4()
    game_complete = False
    game_log = []

    while not (game_complete):

        valid_input = False
        players_turn = game.players_turn
        board_state = np.copy(game.board)

        while not (valid_input):
            if players_turn == 0:
                # player 0's pieves are represented as -1, so multiply by -1
                # so that from every players perspective their pieces are 1
                move = hp.HumanAI(np.copy(game.board)*-1)
            else:
                move = hp.HumanAI(np.copy(game.board))

            valid_input, game_complete = game.AddMove(move)

        game_log.append((players_turn, board_state, move))

    # filter out the non-winning players moves
    winning_game_log = []

    # if the winning player is player 0
    if game.players_turn == 0:
        # add a multiplication factor of -1 so that the log shows
        # the winning players moves as +1 no matter the player that won
        mul_factor = -1
    else:
        mul_factor = 1

    for player, game_state, move in game_log:
        if player == game.players_turn:
            winning_game_log.append((game_state*mul_factor, move))

    return winning_game_log

print(PlayGame())
