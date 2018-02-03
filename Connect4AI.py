import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, reshape
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical
import os

class Connect4Bot():
    def __init__(self):
        # create current baord state
        self.current_board_state = np.zeros(shape=(1, 100), dtype=np.float32)

        # Building convolutional network
        network = input_data(shape=[None, 100], name='input')
        network = reshape(network, new_shape=[-1, 10, 10, 1])
        network = conv_2d(network, 32, 5, activation='relu')
        network = conv_2d(network, 64, 5, activation='relu')
        network = conv_2d(network, 128, 5, activation='relu')
        network = conv_2d(network, 256, 5, activation='relu')
        network = fully_connected(network, 1024, activation='relu')
        network = dropout(network, 0.4)
        network = fully_connected(network, 1024, activation='relu')
        network = dropout(network, 0.4)
        network = fully_connected(network, 10, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy', name='target')

        self.comp_graph = network

        self.model = tflearn.DNN(self.comp_graph, tensorboard_verbose=0,
                                tensorboard_dir='tb_dir')

        if tf.train.latest_checkpoint('Connect4_model/') != None:
            self.model.load('Connect4_model/model.ckpt')


    def predict_move(self, board_state):
        # use a trained model if possible
        if tf.train.latest_checkpoint('Connect4_model/') != None:
            # reshape the board_state
            self.current_board_state = np.array(board_state.flatten(),
                dtype=np.float32, ndmin=2)

            position_probabilities = self.model.predict(self.current_board_state)
            position_probabilities = np.squeeze(position_probabilities)

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


    def train_game(self, features, labels):
        # Train the model
        labels = to_categorical(y=labels, nb_classes=10)
        self.model.fit({'input': features}, {'target': labels}, n_epoch=2,
                  show_metric=False, batch_size=100, run_id='tensorboard_log',
                  shuffle=True)
        self.model.save('Connect4_model/model.ckpt')

        # remove all previous tensorboard files
        folder = 'tb_dir/tensorboard_log'
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)















def ace_ventura():
    pass
