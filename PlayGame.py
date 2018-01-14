import connect4 as c4
import HumanPlayer as hp
import numpy as np
import Connect4AI as c4a
import tensorflow as tf
import Connect4Classifier as c4c

# Call this function to play a game of connect4
# Returns a game log of the winning players moves
# the winning players pieces are always represented as a 1
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
                # player 0's pieces are represented as -1, so multiply by -1
                # so that from every players perspective their pieces are +1
                move = c4a.ComputerPlayer(np.copy(game.board)*-1)
                #move = hp.HumanAI(np.copy(game.board)*-1)
            else:
                # player 1's pieces are represented as 11, so multiply by 1
                # so that from every players perspective their pieces are +1
                move = c4a.ComputerPlayer(np.copy(game.board))
                #move = hp.HumanAI(np.copy(game.board))

            valid_input, player_won, game_complete = game.AddMove(move)

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
            winning_game_log.append((game_state.flatten()*mul_factor, move))

    return (player_won, winning_game_log)



for _ in range(100):
    player_winner, c4game_log = PlayGame()

    # do back propagation if a player won
    if player_winner:
        # Create the Estimator
        connect4_classifier = tf.estimator.Estimator(
            model_fn=c4c.cnn_model_fn, model_dir="Connect4_model/")

        # go through each board state and train off of it
        for c4_game_state, truth_label in c4game_log:
            truth_label = np.array(truth_label, dtype=np.float32, ndmin=1)
            c4_game_state = np.array(c4_game_state, dtype=np.float32, ndmin=2)

            # Train the model
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": c4_game_state},
                y=truth_label,
                batch_size=1,
                num_epochs=None,
                shuffle=True)
            connect4_classifier.train(
                input_fn=train_input_fn,
                steps=1)


















def ace_ventura():
    pass
