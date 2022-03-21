import numpy as np

class Perceptron(object):
    def __init__(self, n_inputs, max_epochs=1e2, learning_rate=1e-2):
        """
        Initializes perceptron object.
        Args:
            n_inputs: number of inputs.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
        """
        self.n_inputs = n_inputs
        self.default_size = 2
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight = np.zeros((self.default_size+1,1))
        self.test_accu = []
        
    def forward(self, input):
        """
        Predict label from input 
        Args:
            input: array of dimension equal to n_inputs.
        """
        sum = np.dot(np.append(input,1),self.weight)
        label = np.sign(sum)
        return label
        
    def train(self, training_inputs, training_labels, testing_inputs, testing_labels):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """
        self.test_accu = []
        self.weight = np.zeros(len(training_inputs[0])+1) # bias
        for _ in range(int(self.max_epochs)):
            for index,input in enumerate(training_inputs):
                output = self.forward(input)
                if training_labels[index]*output<=0:
                    self.weight += self.learning_rate*np.append(input,1)*training_labels[index]
            self.test_accu.append(self.prediction_accuracy(testing_inputs,testing_labels))

    def get_test_accu(self):
        return self.test_accu

    def prediction_accuracy(self, testing_inputs, labels):
        total = len(testing_inputs)
        correct = 0
        for index, input in enumerate(testing_inputs):
            output = self.forward(input)
            if output == labels[index]:
                correct +=1
        return correct/total


def generate(m1,s1,m2,s2):
    x1 = np.random.normal(loc=m1, scale=s1,size=(200,2))
    y1 = np.ones((200))
    x2 = np.random.normal(loc=m2, scale=s2,size=(200,2))
    y2 = -np.ones((200))
    return x1,x2,y1,y2