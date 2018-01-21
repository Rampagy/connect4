import connect4 as c4
import HumanPlayer as hp
import numpy as np
import Connect4AI as c4a
import tensorflow as tf
import math
import time

# Call this function to play a game of connect4
# Returns a game log of the winning players moves
# the winning players pieces are always represented as a 1
def TrainBot(aiBot):
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
                # player 0's pieces are represented as 1, so don't do anything
                # because his pieces are already from his perspective
                move = aiBot.predict_move(np.copy(game.board))
                #move = hp.HumanAI(np.copy(game.board)*-1)
            else:
                # player 1's pieces are represented as 2, so flip his pieces
                # so that from every players perspective their pieces are 1
                move = aiBot.predict_move(FlipPlayers(np.copy(game.board)))
                #move = hp.HumanAI(FlipPlayers(np.copy(game.board)))

            valid_input, player_won, game_complete = game.AddMove(move)

        game_log.append((players_turn, board_state, move))

    # filter out the non-winning players moves
    winning_game_log = []

    # if the winning player is player 0
    if game.players_turn == 0:
        # flag that the players values do not need to be flipped
        flip_player_vals = False
    else:
        # normalize the players persepective such that it is always
        # player 0 (which is a val of 1 on the board)
        flip_player_vals = True

    for player, game_state, move in game_log:

        if flip_player_vals:
            game_state = FlipPlayers(np.copy(game_state))

        if player == game.players_turn:
            winning_game_log.append((game_state, move))

    return (player_won, winning_game_log)



def FlipPlayers(board_state):
    # normalize the game such that the perspective is
    # always from player 0 (value of 1)
    for y in range(board_state.shape[0]):
        for x in range(board_state.shape[1]):
            if board_state[y, x] == 1:
                board_state[y, x] = 2
            elif board_state[y, x] == 2:
                board_state[y, x] = 1

    return board_state




# create AI class
aiBot = c4a.Connect4Bot()

for game_num in range(10):
    game_time_start = time.process_time()
    player_winner, c4game_log = TrainBot(aiBot)
    game_time = time.process_time() - game_time_start

    # do back propagation if a player won
    if player_winner:
        # init array for game states
        game_states = np.zeros(shape=(2*len(c4game_log), 100), dtype=np.float32)

        # init array for labels
        truth_labels = np.zeros(shape=(2*len(c4game_log)), dtype=np.float32)

        # go through each board state and add it to the appropriate arrays
        for i in range(2*len(c4game_log)):
            game = c4game_log[math.floor(i/2)][0]
            truth = c4game_log[math.floor(i/2)][1]

            # if odd add the flipped game state
            if i%2 == 1:
                game_states[i, :] = np.fliplr(game).flatten()
                truth_labels[i] = 9-truth
            # else (if even) add to the original game state
            else:
                game_states[i, :] = game.flatten()
                truth_labels[i] = truth

        #for game_disp, truth_disp in zip(game_states, truth_labels):
        #    print("{}\n{}\n\n".format(np.reshape(game_disp, (10, 10)), truth_disp))

        train_time_start = time.process_time()
        aiBot.train_game(game_states, truth_labels)
        train_time = time.process_time() - train_time_start

        print("{} took {} seconds for the game and {} seconds for the training".format(game_num, game_time, train_time))

















def ace_ventura():
    pass
