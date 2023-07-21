import numpy as np

def one_output_neuron():
    x = [1.0, -2.0, 3.0] # input values
    w = [-3.0, -1.0, 2.0] # weights
    b = 1.0 # bias

    xw0 = x[0] * w[0] 
    xw1 = x[1] * w[1]
    xw2 = x[2] * w[2]

    z = xw0 + xw1 + xw2 + b

    y = max(z, 0) # ReLU

    # BACKWARD PASS BEGINS

    dvalue = 1.0 # gradient value received from next layer

    # d of ReLU using chain rule
    drelu_dz = dvalue * (1.0 if z > 0 else 0.0)
    print(drelu_dz)

    #partial derivatives of summation wit respect to the inputs times weights, also biases
    #if confused, reference partial derivatives of linear addition function f(x, y, z) = x + y + z
    #consider the variables w no respect = 0 since they're treated as constants

    dsum_dxw0 = 1
    drelu_dxw0 = drelu_dz * dsum_dxw0

    dsum_dxw1 = 1
    drelu_dxw1 = drelu_dz * dsum_dxw1

    dsum_dxw2 = 1
    drelu_dxw2 = drelu_dz * dsum_dxw2

    dsum_db = 1
    drelu_db = drelu_dz * dsum_db
    print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

    #partial derivatives of multiplication with respect to individual inputs and weights
    #if confused, reference partial derivatives of linear multiplication function f(x, y) = x * y
    '''
    dmul_dx0 = w[0]
    drelu_dx0 = drelu_dxw0 * dmul_dx0 

    dmul_dx1 = w[1]
    drelu_dx1 = drelu_dxw0 * dmul_dx1

    dmul_dx2 = w[2]
    drelu_dx2 = drelu_dxw0 * dmul_dx2

    dmul_dw0 = x[0]
    drelu_dw0 = drelu_dxw0 * dmul_dw0

    dmul_dw1 = x[1]
    drelu_dw1 = drelu_dxw0 * dmul_dw1

    dmul_dw2 = x[2]
    drelu_dw2 = drelu_dxw0 * dmul_dw2
    '''
    #drelu_dx0 = drelu_dz * 1 * w[0] #multiplying by 1 is useless, exclude it
    #simplified version of calculating partial derivative of ReLU w respect to each of the inputs and weights
    drelu_dx0 = dvalue * (1.0 if z > 0 else 0.0) * w[0]
    drelu_dx1 = dvalue * (1.0 if z > 0 else 0.0) * w[1]
    drelu_dx2 = dvalue * (1.0 if z > 0 else 0.0) * w[2]
    drelu_dw0 = dvalue * (1.0 if z > 0 else 0.0) * x[0]
    drelu_dw1 = dvalue * (1.0 if z > 0 else 0.0) * x[1]
    drelu_dw2 = dvalue * (1.0 if z > 0 else 0.0) * x[2]

    print(drelu_dx0, drelu_dx1, drelu_dx2, drelu_dw0, drelu_dw1, drelu_dw2)

    dx = [drelu_dx0, drelu_dx1, drelu_dx2] #gradient on inputs
    dw = [drelu_dw0, drelu_dw1, drelu_dw2]
    db = drelu_db

    print(w, b)

    w[0] += -0.001 * dw[0]
    w[1] += -0.001 * dw[1]
    w[2] += -0.001 * dw[2]
    b += -0.001 * db

    print(w, b)

    xw0 = x[0] * w[0]
    xw1 = x[1] * w[1]
    xw2 = x[2] * w[2]

    z = xw0 + xw1 + xw2 + b
    y = max(z, 0)
    print(y)

def multiple_output_neurons():
    dvalues = np.array([[1., 1., 1.],
                        [2., 2., 2.],
                        [3., 3., 3.]])

    #3 sets of weights - one set for each neuron
    #4 inputs, meaning 4 weights in each set. Each set corresponds to the weights of the neurons in the previous layer
    #weights are transposed
    
    inputs = np.array([[1, 2, 3, 2.5],
                       [2., 5., -1., 2],
                       [-1.5, 2.7, 3.3, -0.8]])
    

    weights = np.array([[0.2, 0.8, -0.5, 1],
                        [0.5, -0.91, 0.26, -0.5],
                        [-0.26, -0.27, 0.17, 0.87]]).T
    
    biases = np.array([[2, 3, 0.5]])

    layer_outputs = np.dot(inputs, weights) + biases
    relu_outputs = np.maximum(0, layer_outputs)

    drelu = relu_outputs.copy()
    drelu[layer_outputs <= 0] = 0

    #Dense Layer
    dinputs = np.dot(drelu, weights.T)
    dweights = np.dot(inputs.T, drelu)
    dbiases = np.sum(drelu, axis=0, keepdims=True)

    #update parameters
    weights += -0.001 * dweights
    biases += -0.001 * dbiases
    
    print(weights)
    print(biases)
    
    #sum the weights of given input and multiply by passed gradient for the neuron
    #dinputs = np.dot(dvalues, weights.T)
    #dweights = np.dot(inputs.T, dvalues)
    #dbiases = np.sum(dvalues, axis=0, keepdims=True)
    print(drelu)

multiple_output_neurons()