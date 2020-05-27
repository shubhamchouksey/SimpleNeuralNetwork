import numpy as np
import matplotlib.pyplot as plt

# Neural Network for solving x1-x2 Problem
# 1 1 --> 0
# 1 0 --> 1
# 0 1 --> 1
# 0 0 --> 0


# Activation function: sigmoid
def sigmoid(x): 
    return np.exp(x)/(1 + np.exp(x))

# Sigmoid deriative
def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

# Forward function
def forward(x, w1, w2, predict=False):
    a1 = np.matmul(x, w1)
    z1 = sigmoid(a1)
  
    # create and add the bias
    bias = np.ones((len(z1), 1))
    z1 = np.concatenate((bias, z1), axis = 1)
    a2 = np.matmul(z1, w2)
    z2 = sigmoid(a2)
    if predict:
        return z2
    return a1, z1, a2, z2

# Backprop fuction
def backprop(a2, z0, z1, z2, y): 
    delta2 = z2 - y
    Delta2 = np.matmul(z1.T, delta2)
    delta1 = (delta2.dot(w2[1:,:].T))*sigmoid_deriv(a1)
    Delta1 = np.matmul(z0.T, delta1)
    return delta2, Delta1, Delta2


# Make Predictions for the training inputs
# z3 = model.forward(x,True)

# First column is the bias
X = np.array([[1, 1, 0],
              [1, 0, 1],
              [1, 0, 0],
              [1, 1, 1]])
y = np.array([[1], [1], [0], [0]])

# init weights 
w1 = np.random.randn(3,5)
w2 = np.random.randn(6,1)


# init learning rate
lr = 0.09
costs = []
# init epochs
epochs = 17000

m = len(X)

# Start training
for i in range(epochs):

    # Forward
    a1, z1, a2, z2 = forward(X, w1, w2)
    
    # Backprop 
    delta2, Delta1, Delta2 = backprop(a2, X, z1, z2, y)
    #print(delta2,Delta1,Delta2)

    w1 -= lr*(1/m)*Delta1
    w2 -= lr*(1/m)*Delta2

    # Add costs to list for plotting
    c = np.mean(np.abs(delta2))
    costs.append(c)
    
    if i % 1000 == 0:
        print("Iteration: ",i, "Error: ",c)
    
# Training complete
print('Training completed.')

z3 = forward(X,w1,w2,True)
print('Percentages: ')
print(z3)
print('Predictions: ')
print(np.round(z3))

# Plot cost 
plt.plot(costs)
plt.show()