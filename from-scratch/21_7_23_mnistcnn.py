import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import os.path

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
#print(m, n)

data = np.array(data)
np.random.shuffle(data)

print(data.T)

