# DEEP LEARNING SINGLE NEURON IMPLEMENTATION

## INTRODUCTION
The purpose of this project is to understand how weights change for a single neuron in a neural network.
It was implemented in a naive, not optimized way, for educational purposes.

In general, a perceptron receives inputs. Each input is assigned a weight between -1 and 1.
The output is the weighted sum of each input multiplied by it's weight.

![Perceptron!](https://i.ibb.co/Cnbdfdc/Untitled-Diagram.png)

## PROJECT DESCRIPTION
In this project we will implement [`Linear Regression`](https://en.wikipedia.org/wiki/Linear_regression) using a single neuron, a [`perceptron`](https://www.simplilearn.com/what-is-perceptron-tutorial), to help us find relationships between variables.
The dataset for this project is 2000 randomly generated points classified either above or below a line of our choice.

This is a visual representation of a perceptron learning how to classify the data.

![learning_perceptron](https://i.ibb.co/0yrKkTD/ezgif-4-5d1bf3549507.gif)

## PROJECT STRUCTURE
`predict` Will return the weighted sums given x0, x1, x2.

`train` Will train the neuron and change weights each iteration, also known as [`Stochastic gradient descent`](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

`activate_function` Is a simple activation function. There are better choices for activation functions, we choose this one for simplicity.

`training_formula` Will hold the line our data is classified upon, following the formula: y = mx + b.

`final_forumla` Will return the y value given any x following the formula: y = mx + b.

Note:
Since everythin is randomized in this project, the results will differ with each run.
