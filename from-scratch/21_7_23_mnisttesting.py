#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import os.path
from nnfunctions import *

'''
digit_mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = digit_mnist.load_data()

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

train_images = train_images / 255.0
test_images = test_images / 255.0




plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()



plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''


data = pl.read_csv(os.path.join(os.getcwd(), "from-scratch/mnist-ds/mnist_train.csv"))
m, n = data.shape
data = np.array(data)
np.random.shuffle(data)

num_values = data.T[0] #all of the numbers
num_pixels = data.T[1:n] #pixel values of every number 
num_pixels = num_pixels / 255.0 #making all pixel values between 0-1

bs_i = 0 #batch start index
be_i = 3 #batch end index, change to 20 later

batch = num_pixels.T[bs_i:be_i] #transposes the transposed array so its in normal state

print(batch)

dense1 = Layer_Dense(784, 20)
output_layer = Layer_Dense(20, 10)

dense1.forward(batch)

'''
reLU1 = reLU()
reLU1.forward(dense1.output)

output_layer.forward(reLU1.output)

softmax = softmax()
softmax.forward(output_layer.output)
print(softmax.output)


loss_function = Loss_Categorical_CrossEntropy()
num_values = num_values[bs_i:be_i]
loss = loss_function.calculate(softmax.output, num_values[bs_i:be_i])

print("Loss: ", loss)

predictions = np.argmax(softmax.output, axis=1)
if len(num_values.shape) ==2:
    num_values = np.argmax(num_values, axis=1)
accuracy = np.mean(predictions==num_values)

print('Acc: ', accuracy)
'''



