'''
test.py
'''

import neural_net as nn


neural_net_filename = input('Neural network init filename: ')
Ni, Nh, No, weights = nn.read_neural_net_file(neural_net_filename)
train_filename = input('Neural network testing data filename: ')
training_data = nn.read_data_file(train_filename)

nn.test(Ni, Nh, No, weights, training_data)

results_filename = input('Neural network trained filename: ')
nn.write_results_file(results_filename)
