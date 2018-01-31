import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


class Connect4Bot():
    def __init__(self):
        # create current baord state
        self.current_board_state = np.zeros(shape=(10, 10), dtype=np.float32)

        # create the computational graph
        self.c4_estimator = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="Connect4_model/")

        # create input function
        self.pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": self.current_board_state.flatten()}, num_epochs=1, shuffle=False)

    def predict_move(self, board_state):
        # use a trained model if possible
        if tf.train.latest_checkpoint('Connect4_model/') != None:
            # reshape the board_state
            self.current_board_state = np.array(board_state, dtype=np.float32, ndmin=2)

            # run the prediction
            eval_results = self.c4_estimator.predict(input_fn=self.pred_input_fn)

            # grab the probabilities
            position_probabilities = next(eval_results)['probabilities']

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
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": features},
            y=labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
        self.c4_estimator.train(
            input_fn=train_input_fn,
            steps=1)








"""Convolutional Neural Network Estimator for connect4, built with tf.layers."""

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # connect4 'images' are 10x10 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 10, 10, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 2x2 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 10, 10, 1]
  # Output Tensor Shape: [batch_size, 10, 10, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #2
  # Computes 64 features using a 2x2 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 10, 10, 32]
  # Output Tensor Shape: [batch_size, 10, 10, 64]
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #2
  # Computes 64 features using a 2x2 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 10, 10, 32]
  # Output Tensor Shape: [batch_size, 10, 10, 64]
  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=96,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #2
  # Computes 64 features using a 2x2 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 10, 10, 32]
  # Output Tensor Shape: [batch_size, 10, 10, 64]
  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #2
  # Computes 64 features using a 2x2 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 10, 10, 32]
  # Output Tensor Shape: [batch_size, 10, 10, 64]
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=160,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 10, 10, 64]
  # Output Tensor Shape: [batch_size, 10 * 10 * 64]
  conv2_flat = tf.reshape(conv5, [-1, 10 * 10 * 160])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 10 * 10 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=conv2_flat, units=2048, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
































def ace_ventura():
    pass
