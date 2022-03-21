from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from mlp_numpy import *
from matplotlib import pyplot as plt


import argparse
import numpy as np

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10
# GRADIENT_DESCENT_MODE = 'batch'
GRADIENT_DESCENT_MODE = 'batch'


FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    count = 0
    length = len(targets)
    for i in range(length):
        if predictions[i][0]==targets[i][0] and predictions[i][1]==targets[i][1]:
            count += 1
    return count/length

def calculate_accuracy(mlp, inputs, labels):
    predictions = list()
    for input in inputs:
        output = mlp.forward(input)
        if(output[0][0]>=0.5):
            predictions.append(np.array([1,0]))
        else:
            predictions.append(np.array([0,1]))
    return accuracy(predictions,labels)

def train(training_inputs,training_labels,testing_inputs,testing_labels):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    hidden_list = FLAGS.dnn_hidden_units.split(',')
    n_hidden = list(map(int,hidden_list))
    mlp = MLP(2, n_hidden, 2, FLAGS.learning_rate)
    train_accuracy = list()
    test_accuracy = list()
    epoch_list = list()
    descent = FLAGS.gradient_descent_mode
    if FLAGS.gradient_descent_mode == 'batch':
        print("==========This gradient descent mode is batch gradient descent==============")
    else:
        print("==========This gradient descent mode is stochastic gradient descent==============")

    for epoch in range(FLAGS.max_steps):

        for index, training_input in enumerate(training_inputs):
            if descent == 'stochastic':
                mlp.clear_descent()
            # forwarding and backwarding is the same
            training_label = training_labels[index]
            mlp.forward(training_input)
            mlp.lossFunction(training_label)
            mlp.backward()

            if descent == 'stochastic':
                mlp.upadte_parameters()
        
        if descent == 'batch':
            mlp.upadte_parameters(len(training_inputs))
            mlp.clear_descent()

        if epoch%FLAGS.eval_freq == 0:
            epoch_list.append(epoch)
            # calculate accuracy of train
            train_accur_temp = calculate_accuracy(mlp,training_inputs,training_labels)
            train_accuracy.append(train_accur_temp)
            # calculate accuracy of train
            test_accur_temp = calculate_accuracy(mlp,testing_inputs,testing_labels)
            test_accuracy.append(test_accur_temp)
            if(epoch%100 ==0):
                print("Epoch",epoch,"\n","train_accuracy is",train_accur_temp,"\n","test_accuracy is",test_accur_temp) 

    return epoch,train_accuracy,test_accuracy


def generate():
    x, y = datasets.make_moons(n_samples=2000, shuffle=True, noise=None, random_state=None)
    y = np.reshape(y, (2000, -1))
    enc = OneHotEncoder(sparse = False) 
    y_onehot = enc.fit_transform(y)
    training_inputs = x[:1400]
    training_labels = y_onehot[:1400]
    testing_inputs = x[1400:]
    testing_labels = y_onehot[1400:]
    return training_inputs,training_labels,testing_inputs,testing_labels

def plot_picture(epoch,train_accuracy,test_accuracy):
    x_axis = [i for i in range(1, len(train_accuracy) + 1)]
    plt.ylim(0.3,1.0)
    plt.plot(x_axis, train_accuracy, c='red', label='training accuracy')
    plt.plot(x_axis, test_accuracy, c='blue', label='testing accuracy')

    plt.legend(loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.show()
    return

def main():
    """
    Main function
    """
    training_inputs,training_labels,testing_inputs,testing_labels = generate()
    epoch,train_accuracy,test_accuracy = train(training_inputs,training_labels,testing_inputs,testing_labels)
    plot_picture(epoch,train_accuracy,test_accuracy)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    parser.add_argument('--gradient_descent_mode', type=str, default = GRADIENT_DESCENT_MODE,
                          help='gradient descent mode of training')
    FLAGS, unparsed = parser.parse_known_args()
    main()