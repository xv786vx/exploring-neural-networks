'''
import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

print(loss)

loss = -math.log(softmax_output[0])
print(loss) 
'''

import numpy as np
softmax_outputs = np.array([[0.7, 0.2, 0.1],
                            [0.5, 0.1, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([0, 1, 1])

predictions = np.argmax(softmax_outputs, axis=1)
print(predictions)

if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1)

accuracy = np.mean(predictions == class_targets)
print('Acc:', accuracy)




