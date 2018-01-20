import Connect4Classifier as c4c
import numpy as np
import tensorflow as tf


def ComputerPlayer(board_state):
    # use a trained model if possible
    if tf.train.latest_checkpoint('Connect4_model/') != None:
        # Create the Estimator
        connect4_classifier = tf.estimator.Estimator(model_fn=c4c.cnn_model_fn, model_dir="Connect4_model/")

        # reshape the board_state
        board_state = np.array(board_state, dtype=np.float32, ndmin=2)

        # input the data
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": board_state.flatten()}, num_epochs=1, shuffle=False)

        # run the prediction
        eval_results = connect4_classifier.predict(input_fn=pred_input_fn)

        # grab the probabilities
        position_probabilities = next(eval_results)['probabilities']
        #print(position_probabilities)
        move = 0
        prev_prob = 0
        col_count = 0

        # pick a random number between 0 and 1
        uniform_num = np.random.uniform(0, 1)

        # check to see which 'window' the random number is in
        # if the value is in an invalid window (aka invalid move)
        # pick a new random number and try again until a valid move is chosen
        for column_prob in position_probabilities:
            if ((prev_prob <= uniform_num) and \
                (uniform_num <= prev_prob + column_prob)):
                    move = col_count
                    break

            prev_prob += column_prob
            col_count += 1

        return move

    # else pick randomly
    else:
        move = np.random.randint(board_state.shape[1])
        while (board_state[0, move] != 0):
            move = np.random.randint(board_state.shape[1])
        return move










































def ace_ventura():
    pass
