import numpy as np
from nn import *
from cd import create_data


X, y = create_data(100, 3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_Categorical_CrossEntropy()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])


loss = loss_function.calculate(activation2.output, y)

print("Loss: ", loss)

predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) ==2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)

print('Acc: ', accuracy)
