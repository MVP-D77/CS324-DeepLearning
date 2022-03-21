import numpy as np

class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.params = dict()
        self.params['weight'] = np.random.normal(loc = 0, scale= 0.5,size = (in_features,out_features))
        self.params['bias'] = np.zeros((1,out_features))
        self.x = None
        self.grads = dict()
        self.grads['weight'] = np.zeros((in_features,out_features))
        self.grads['bias'] = np.zeros((1,out_features))
        
        
    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object
        and use them in the backward pass computation. This is true for *all* forward methods of *all* modules in this class
        """
        self.x = x
        out = np.dot(self.x, self.params['weight']) + self.params['bias']
        return out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """
        self.grads['bias'] += dout
        self.grads['weight'] += np.dot(self.x.T, dout)
        dx = np.dot(dout, self.params['weight'].T)
        return dx

class ReLU(object):
    def __init__(self) :
        self.x = None

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        self.x = x
        return np.maximum(x, 0)

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        return np.where(self.x>0, dout ,0)

class SoftMax(object):
    def __init__(self) :
        self.output = None
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
    
        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        
        """
        b = x.max()
        y = np.exp(x-b)
        output = y/y.sum()
        self.output = output
        return output

    def backward(self, dout):
        """
        Backward pass. 
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        return dout

class CrossEntropy(object):
    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            out: cross entropy loss
        """
        loss = -np.sum(y * np.log(x), axis=1)
        return loss

    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            dx: gradient of the loss with respect to the input x.
        """
        # dx = -np.true_divide(y,x)
        dx = np.subtract(x, y)
        return dx
