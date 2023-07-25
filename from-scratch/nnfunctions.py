import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) #creates a matrix of normalized weights with n_inputs rows * n_neurons columns
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases


class reLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)