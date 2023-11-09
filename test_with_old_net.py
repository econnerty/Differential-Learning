import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the activation function and its derivative
def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1.0 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)
def drelu(x):
    return np.where(x > 0, 1, 0)

# Initialize the weights for a 3-layer neural network
np.random.seed(42)  # For reproducibility
w1 = np.random.randn(1, 10) 
w2 = np.random.randn(10, 20)
w3 = np.random.randn(20, 1) 
b1 = np.zeros((1, 10))
b2 = np.zeros((1, 20))
b3 = np.zeros((1, 1))

# Predicted outputs for the hidden layers
p1 = np.random.randn(1, 10)
p2 = np.random.randn(1, 20)


# Initial state vector X
X = np.array([[-1], [-0.9], [-0.8], [-0.7], [-0.6], [-0.5], [-0.4], [-0.3], [-0.2], [-0.1],
              [0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1]])
Y = np.array([[-0.96], [-0.577], [-0.073], [0.377], [0.641], [0.66], [0.461], [0.134],
              [-0.201], [-0.434], [-0.5], [-0.393], [-0.165], [0.099], [0.307], [0.396],
              [0.345], [0.182], [-0.031], [-0.219], [-0.321]])

#sample points in the range [-1,1]
X2 = np.random.uniform(-1,1,(1000,1))

def plot_approximation(epoch):
    # Forward propagate through the network
    z1, z2, z3 = forward_pass(X2)
    # Sort the X and z3 arrays by the X values for proper plotting
    sorted_indices = np.argsort(X2[:, 0])
    sorted_X = X2[sorted_indices]
    sorted_z3 = z3[sorted_indices]
    
    plt.figure()
    plt.plot(X, Y, 'r-', label='Actual function')  # Plot as a red line
    plt.plot(sorted_X, sorted_z3, 'b-', label='NN approximation')  # Plot as a blue line
    plt.legend()
    plt.title(f'Function Approximation after {epoch} Epochs')
    plt.xlabel('X value')
    plt.ylabel('Predicted Y value')
    plt.show()

# Forward pass of the neural network
def forward_pass(X):
    global w1, w2, w3, b1, b2, b3
    z1 = relu(np.dot(X, w1) + b1)
    z2 = relu(np.dot(z1, w2) + b2)
    z3 = np.dot(z2, w3) + b3
    return z1, z2, z3


# Update functions for the predicted outputs and weights
def update_predicted_outputs(p, z, learn_rate):
    error = z - p
    p_new = p + learn_rate * error
    return p_new

def backpropagate(X, y_true, z1, z2, z3, eta):
    global w1, w2, w3, b1, b2, b3, p1, p2
    # Compute the gradient of the loss w.r.t. the final output
    grad_loss_z3 = 2 * (z3 - y_true)

    # Backpropagate this gradient to get gradients for w3 and b3 using predicted outputs (p2)
    grad_w3 = np.dot(p2.T, grad_loss_z3)
    grad_b3 = np.sum(grad_loss_z3, axis=0, keepdims=True)
    
    # Backpropagate through the second hidden layer using the predicted output p2
    grad_loss_p2 = np.dot(grad_loss_z3, w3.T) * drelu(p2)
    grad_w2 = np.dot(p1.T, grad_loss_p2)
    grad_b2 = np.sum(grad_loss_p2, axis=0, keepdims=True)
    
    # Backpropagate through the first hidden layer using the predicted output p1
    grad_loss_p1 = np.dot(grad_loss_p2, w2.T) * drelu(p1)
    grad_w1 = np.dot(X.T, grad_loss_p1)
    grad_b1 = np.sum(grad_loss_p1, axis=0, keepdims=True)

    # Update the weights and biases
    w1 -= eta * grad_w1
    b1 -= eta * grad_b1
    w2 -= eta * grad_w2
    b2 -= eta * grad_b2
    w3 -= eta * grad_w3
    b3 -= eta * grad_b3


# Mean Squared Error (MSE) loss function
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Training loop parameters
epochs = 1000
learn_rate = 0.001
hidden_layer_learn = .2


# Training loop with loss computation
losses = []  # To store loss at each epoch

for epoch in range(epochs):
    for input,output in zip(X,Y):
        z1, z2, z3 = forward_pass(input)
        p1 = update_predicted_outputs(p1, z1, hidden_layer_learn)
        p2 = update_predicted_outputs(p2, z2, hidden_layer_learn)
 
        # Backpropagation to update weights using predicted outputs
        backpropagate(input, output, z1, z2, z3, learn_rate)
        
        # Compute and store the loss
        loss = mse_loss(z3, output)
        losses.append(loss)
    #print("Epoch: " + str(np.mean(losses)))
    losses = []


        #Print loss for this epoch
    #print("Epoch: " + str(epoch) + " Loss: " + str(loss))

plot_approximation(epochs)
#print("Final Loss: " + losses[-1].astype(str))
# Plot the loss over epochs
"""plt.figure(figsize=(10, 5))
plt.plot(losses, label='MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MSE Loss Over Epochs')
plt.legend()
plt.show()"""