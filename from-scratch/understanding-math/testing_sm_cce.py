import numpy as np
from sm_cce_funcs import *

output = np.array([[0.4, -0.5, -0.8, 1.1, 0.1, -0.9, 0.4, 0.7, -1, 0.2]])
true = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

softmax = Activation_Softmax()
cce = Loss_Categorical_CrossEntropy()

softmax.forward(output)
print(softmax.output)

cce.forward(output, true)
print(cce.calculate(output, true))