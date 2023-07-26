import numpy as np
from nn import *
from cd import create_data


X, y = create_data(100, 3)

dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(64, 3)
activation2 = Activation_Softmax()

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_SGD(decay = 1e-3, momentum=0.9)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)

    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}')

    #backward pass begins
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)







'''
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

print(y)
loss = loss_function.calculate(activation2.output, y)

print("Loss: ", loss)

predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) ==2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)

print('Acc: ', accuracy)
'''
