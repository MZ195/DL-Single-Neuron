import random
import time
import math
import numpy as np


class Neuron(object):

    def __init__(self):
        """
        weights
          weights are assigned to each input to the neuron.
          we will start by random weights for simplicity, but there are more fine ways to choose weights.
        
        learning_rate
          the learning rate will determine how far the next step is going to be.
          we choose 0.005 to show the slow progress, in production it's not ideal.
         
        points
          we will generate 2,000 random points and classify them either above or beneath a per-defined line of our choice.
          the neuron will try guessing this line.
        """
        self.weights = [None] * 3
        self.learning_rate = 0.005
        self.points = []

        self.setup()
    
    def predict(self, training_inputs):
        """
        training_inputs
          given (x, y, bias), we will compute and return the weighted sums.
        """
        sum = 0
        for i in range(len(training_inputs)):
            sum = sum + (training_inputs[i] * self.weights[i])
        return self.activate_function(sum)

    def train(self):
        """
        Training the neuron and adjusting the weights with each iteration.
        """
        for pt in self.points:
            training_inputs = [pt['x'], pt['y'], pt['bias']]
            target = pt['label']
            self.train_helper(training_inputs, target)

    def train_helper(self, training_inputs, target):
        """
        training_inputs
          given (x, y, bias), we will get what our current neuron thinks is the right answer.

        target
          the right label for the input data, we will calculate the error then adjust the weights.
        """
        result = self.predict(training_inputs)
        error = target - result

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + (error * training_inputs[i] * self.learning_rate)
    
    def activate_function(self, num):
        """
        Simple activation function that retuens 1 if the weighted sum is positive, and -1 otherwise
        """
        if(num > 0):
            return 1
        else:
            return -1
    
    def training_formula(self, x):
        """
        this is the formula for our line, we will classify the data based on this line
        y = 3x + 2
        """
        return 3*x+2

    def final_forumla(self, x):
        """
        w0x0 + w1x1 + w2*b = 0
        to represent the form y = mx + b
        we need to compute the weights as x1 = -(w2 + w0x0)/w1
        """
        return -(self.weights[2] + self.weights[0]*x)/self.weights[1]

    def setup(self):
        """
        To setup the environment for our neuron
        """
        for i in range(len(self.weights)):
            self.weights[i] = random.uniform(-1, 1)

        for i in range(2000):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            label = 1

            line = self.training_formula(x)

            if (y < line):
                label = -1

            self.points.append({'x': x, 'y': y, 'bias': 1,'label': label})

    
if __name__ == "__main__":
    neuron = Neuron()
    neuron.train()

    print("Train formula result for x=1: %.2f" % neuron.training_formula(1))
    print("Model formula result for x=1: %.2f" % neuron.final_forumla(1))

