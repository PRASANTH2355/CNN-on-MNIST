import numpy as np
from layer import Layer
from activation import Activation


class Sigmoid(Activation):

    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_derivative)


class Softmax(Layer):
    def forward(self, input):
        tm = np.exp(input)
        self.output = tm / np.sum(tm)

        return self.output

    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)

        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
