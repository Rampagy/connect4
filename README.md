# connect4

This is a game environment for a 10x10 game of connect4.  The game is larger than the normal game to prevent solving of the game.


## Training the Deep Neural Network

Simply run TrainBot.py.  A trained model has been included but may need additional training.

## Play the Deep Neural Network

Simply run PlayGame.py.  This will whichever model is in Connect4_model/ and if there is none it will select random rows.  The pretrained model has been was trained for 100,000 games.

## Improvements

1. Create input pipeline so model weights and biases aren't releaded on every predict (only when neede will it reload).
2.  Create a better visual.
3.  Create some scoring scheme to test improvements rather then a humans opinion.
