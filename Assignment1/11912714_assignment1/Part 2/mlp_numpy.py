from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes, rate):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.softmax = SoftMax()
        # hiddenlayer linear
        self.linears = list()
        self.relu = list()
        input_size = self.n_inputs
        for next_size in self.n_hidden:
            self.linears.append(Linear(input_size, next_size))
            self.relu.append(ReLU())
            input_size = next_size
        self.linear_last = Linear(input_size, self.n_classes)
        self.rate = rate

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        x = np.reshape(x,(-1,x.shape[0]))
        for i in range(len(self.linears)):
            x = self.relu[i].forward(self.linears[i].forward(x))
        out = self.softmax.forward(self.linear_last.forward(x))
        self.out = out
        return out

    def lossFunction(self,label):
        self.label = label
        self.cross = CrossEntropy()
        self.loss = self.cross.forward(self.out,label)
        return self.loss

    def backward(self):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        dt = self.softmax.backward(self.cross.backward(self.out,self.label))
        dt = self.linear_last.backward(dt)
        for i in range(len(self.linears)):
            dt = self.linears[len(self.linears)-1-i].backward(self.relu[len(self.linears)-1-i].backward(dt))
        return

    def clear_descent(self):
        self.linear_last.grads['weight'] = np.zeros(self.linear_last.grads['weight'].shape)
        self.linear_last.grads['bias'] = np.zeros(self.linear_last.grads['bias'].shape)
        for linear in self.linears:
            linear.grads['weight'] = np.zeros(linear.grads['weight'].shape)
            linear.grads['bias'] = np.zeros(linear.grads['bias'].shape)
        return

    def upadte_parameters(self, size=1):    
        self.linear_last.params['weight'] = self.linear_last.params['weight'] - self.rate * self.linear_last.grads['weight']/size
        self.linear_last.params['bias'] = self.linear_last.params['bias'] - self.rate * self.linear_last.grads['bias']/size

        for linear in self.linears:
            linear.params['weight'] = linear.params['weight'] - self.rate * linear.grads['weight']/size
            linear.params['bias'] = linear.params['bias'] - self.rate * linear.grads['bias']/size
        return

        
