import numpy as np


inputs = [1,2,3,2.5]
weights = [[0.2, 0.8, -0.5, 1.0], 
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases= [2, 3, 0.5]

output = np.dot(weights, inputs) + biases
print(output)

layer_outputs = [] #output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for neuron_input, weight in zip(inputs, neuron_weights):
        neuron_output += neuron_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)

'''
weights = [[0.2, 0.8, 0.9],
          [0.1, 1.0, 0.5],
          [0.3, 0.4, 0.6]]


inputs = np.array([0.7, 0.4, 0.3])

biases = np.array([1, 0.7, 0.2])

z = np.dot(weights, inputs.T) + biases.T
print(z)
'''