import numpy as np
from layer import Layer
from activation import Activation


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_derivative(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_derivative)


class Softmax(Layer):
    def forward(self, input):
        tm = np.exp(input)
        self.output = tm / np.sum(tm)

        return self.output

    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)

        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
