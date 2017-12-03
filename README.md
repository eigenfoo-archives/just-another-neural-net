# Just Another Neural Network

An implementation of a simple, multilayer perceptron with one hidden layer.

## Requirements

- A Python 3 interpreter

## Installation and Usage

Using git:

```
$ git clone https://github.com/eigenfoo/just_another_neural_net.git
$ cd just_another_neural_net/neural_net
$ python train.py
$ python test.py
```

Alternatively, you can download the .zip file from my [GitHub
repository](https://github.com/eigenfoo/just_another_neural_net).

Upon running `train.py`, the user will be prompted for:

1. The filename the initial weights and biases
2. The filename of the training data
3. The learning rate to train with
4. The number of epochs to train for
5. The filename to write the trained weights and bias

Upon running `test.py`, the user will be prompted for:

1. The filename of the trained weights and biases
2. The filename of the testing data
3. The filename to write the results of the testing

All file formats are as described in the specification, in the assignment
prompt.

## Description of Data Set and Learning Parameters:

The relevant files are:

- `iris_init.txt`, the neural network initialization file
- `iris_train.txt`, the training data
- `iris_test.txt`, the test data
- `iris_.5_100_trained.txt`, the weights upon training with learning rate = 0.5
  and number of epochs = 100
- `iris_.5_100_results.txt`, the results (i.e. performance statistics) upon
  testing

The dataset is the famous iris flower dataset first introduced by Ronald Fisher
in 1936.  The dataset consists of 50 observations of 3 species of iris flower
(_setosa_, _virginica_ and _versicolor_). Each observation consists of 4
features: petal length, petal width, sepal length and sepal width.

The data was shuffled and split into a training set of 100 observations, and a
testing set of 50 observations. The learning rate was 0.5, and the number of
epochs was 100. The network consisted of 4 input nodes, 5 hidden nodes and 3
output nodes. This produced surprisingly good results, with all performance
statistics upwards of 90%, even in spite of the small amount of data available.

The dataset was collected from scikit-learn's `sklearn.datasets` module, and
manually manipulated into the format described in the specification. The initial
weights were generated using a Gaussian random number generator (mean 0,
standard deviation 0.5), and taking their absolute values (so that all initial
weights would be non-negative).
