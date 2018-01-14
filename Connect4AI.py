import Connect4Classifier as c4c
import numpy as np
import tensorflow as tf


def ComputerPlayer(board_state):
    # use a trained model if possible
    if len(tf.train.latest_checkpoint('Connect4_model/')) != 0:
        # Create the Estimator
        connect4_classifier = tf.estimator.Estimator(model_fn=c4c.cnn_model_fn, model_dir="Connect4_model/")

        # reshape the board_state
        board_state = np.array(board_state.flatten(), dtype=np.float32, ndmin=2)

        # input the data
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": board_state}, num_epochs=1, shuffle=False)

        # run the prediction
        eval_results = connect4_classifier.predict(input_fn=pred_input_fn)

        # grab the probabilities
        position_probabilities = next(eval_results)['probabilities']

        # find the highest probability that is also valid
        max_prob = -1
        col_count = 0
        for prob in position_probabilities:
            col_count += 1
            if (prob > max_prob) and (board_state[0, col_count] == 0):
                max_prob = prob
                move = col_count

        return move

    # else use random rows
    else:
        print("rando bot")
        move = np.random.randint(board_state.shape[1])
        while (board_state[0, move] != 0):
            move = np.random.randint(board_state.shape[1])
        return move
