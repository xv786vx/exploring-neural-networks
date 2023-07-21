

x = [1.0, -2.0, 3.0] #input values
w = [-3.0, -1.0, 2.0] #weight values
b = 1.0 #bias of output neuron

xw0 = x[0] * w[0] #output of previous neurons times their weights
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
print(xw0, xw1, xw2)

z = xw0 + xw1 + xw2 + b #output of neuron + bias, value of output neuron

y = max(z, 0) #reLU

#derivative from next layer
dvalue = 1.0

#derivative of ReLU, chain rule
drelu_dz = dvalue * (1.0 if z > 0 else 0.0)
print(drelu_dz)

#partial derivatives of multiplication, chain rule
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1

drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

#partial dervatives of multiplication, the chain rule
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dx1 = drelu_dxw0 * dmul_dx1
drelu_dx2 = drelu_dxw0 * dmul_dx2
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dw1 = drelu_dxw0 * dmul_dw1
drelu_dw2 = drelu_dxw0 * dmul_dw2

print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

