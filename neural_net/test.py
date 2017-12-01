'''
test.py
'''

import neural_net as nn


NEURAL_NET_FILENAME = input('Neural network trained filename: ')
NUM_INPUT, NUM_HIDDEN, NUM_OUTPUT, WEIGHTS = \
    nn.read_neural_net_file(NEURAL_NET_FILENAME)

TRAIN_FILENAME = input('Neural network testing data filename: ')
TRAINING_DATA = nn.read_data_file(TRAIN_FILENAME)

CONFUSION = nn.test(NUM_OUTPUT, WEIGHTS, TRAINING_DATA)

RESULTS_FILENAME = input('Neural network results filename: ')
nn.write_results_file(RESULTS_FILENAME, NUM_OUTPUT, CONFUSION)
