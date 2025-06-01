import numpy as np
from layer import Layer
from scipy import signal


class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth

        self.input_shape = input_shape
        self.input_depth = input_depth

        self.output_shape = (
            depth,
            input_height - kernel_size + 1,
            input_width - kernel_size + 1,
        )
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)

        self.kernals = np.random.rand(*self.kernels_shape)
        self.biases = np.random.rand(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(
                    self.input[j], self.kernels[i, j], "valid"
                )
            return self.output
