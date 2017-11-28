'''
neural_net.py
'''

import numpy as np
from decimal import Decimal, ROUND_HALF_UP


def read_neural_net_file(filename):
    ''' Reads neural net file'''
    with open(filename) as f:
        line = f.readline()
        nums = [int(num) for num in line.split()]
        Ni, Nh, No = nums

        w1 = np.zeros([Nh, Ni+1])
        for i in range(Nh):
            line = f.readline()
            nums = [float(num) for num in line.split()]
            w1[i, :] = nums

        w2 = np.zeros([No, Nh+1])
        for i in range(No):
            line = f.readline()
            nums = [float(num) for num in line.split()]
            w2[i, :] = nums

    return Ni, Nh, No, [w1, w2]


def read_data_file(filename):
    ''' Reads data file'''
    with open(filename) as f:
        line = f.readline()
        nums = [int(num) for num in line.split()]
        num_obs, Ni, No = nums

        inputs = np.zeros([num_obs, Ni])
        outputs = np.zeros([num_obs, No])

        for i in range(num_obs):
            line = f.readline()
            nums = [float(num) for num in line.split()]
            inputs[i, :] = nums[:Ni]
            outputs[i, :] = nums[Ni:]

    return list(zip(inputs, outputs))


def write_trained_file(filename, Ni, Nh, No, weights):
    ''' Writes trained neural net '''
    with open(filename) as f:
        print(' '.join(str(n) for n in [Ni, Nh, No]), file=f)
        for i in range(Nh):
            print(' '.join('{:.3f}'.format(j) for j in weights[0][i, :]),
                  file=f)
        for i in range(No):
            print(' '.join('{:.3f}'.format(j) for j in weights[1][i, :]),
                  file=f)


def write_results_file(filename):
    ''' Writes results file '''
    with open(filename) as f:
        print('foo', file=f)


def sigmoid(z):
    '''The sigmoid function'''
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    '''Derivative of the sigmoid function'''
    return sigmoid(z) * (1-sigmoid(z))


def train():
    ''' Trains neural network '''
    pass


def test():
    ''' Tests neural network '''
    pass
