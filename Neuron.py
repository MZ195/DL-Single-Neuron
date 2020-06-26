import random
import time
import math
import numpy as np


class Neuron(object):

    def __init__(self):

        self.weights = [None] * 3
        self.learning_rate = 0.01
        self.points = []

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
    

    def guess(self, training_inputs):
        sum = 0
        for i in range(len(training_inputs)):
            sum = sum + (training_inputs[i] * self.weights[i])
        return self.activate_function(sum)

    def train_helper(self, training_inputs, target):
        result = self.guess(training_inputs)
        error = target - result

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + (error * training_inputs[i] * self.learning_rate)

    def train(self):
        for pt in self.points:
            training_inputs = [pt['x'], pt['y'], pt['bias']]
            l = pt['label']
            self.train_helper(training_inputs, l)
    
    def activate_function(self, num):
        return 1 / (1 + np.exp(-num))
    
    def training_formula(self, x):
        return 3*x+2

    def final_forumla(self, x):
        return -(self.weights[2] + self.weights[0]*x)/self.weights[1]

    def get_final_formula(self):
        # y = mx + b
        # w0x0 + w1x1 + w2*b = 0
        
        return (self.weights[0]/self.weights[1], self.weights[2]/self.weights[1])

    def get_weights(self):
        return self.weights

    
if __name__ == "__main__":
    neuron = Neuron()
    neuron.train()

    # print(neuron.get_weights())
    m, b = neuron.get_final_formula()

    print("Train formula: y = 3*x + 2")
    print("Model formula: y = %.2f*x + %.2f" % (m, b))

    print("Train formula result for x=1: %.2f" % neuron.training_formula(1))
    print("Model formula result for x=1: %.2f" % neuron.final_forumla(1))

