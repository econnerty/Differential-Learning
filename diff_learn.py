import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp,odeint

# Define the activation function and its derivative
def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1.0 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)
def drelu(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

# Initialize the weights for a 3-layer neural network
np.random.seed(42)  # For reproducibility
w1 = np.random.randn(2, 10) 
w2 = np.random.randn(10, 20)
w3 = np.random.randn(20, 2) 
b1 = np.zeros((1, 10))
b2 = np.zeros((1, 20))
b3 = np.zeros((1, 2))

# Predicted outputs for the hidden layers
p1 = np.random.randn(1, 10)
p2 = np.random.randn(1, 20)

# Lotka-Volterra model parameters
alpha, beta, gamma, delta = 1.8, 0.001, .1, .0005

# Initial state vector X
initial_conditions = np.array([[500, 10]])  # Example initial populations

# Forward pass of the neural network
def forward_pass(X):
    global w1, w2, w3, b1, b2, b3
    z1 = tanh(np.dot(X, w1) + b1)
    z2 = tanh(np.dot(z1, w2) + b2)
    z3 = np.dot(z2, w3) + b3
    return z1, z2, z3


# Define the Lotka-Volterra model for use with solve_ivp
def lotka_volterra_ivp(t, X, alpha, beta, gamma, delta):
    x, y = X
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]


def solve_odeint_step(X, dt, alpha, beta, gamma, delta, n_steps):
    #sol = odeint(lotka_volterra_odeint, X.flatten(), t_span, args=(alpha, beta, gamma, delta),rtol=1e-12, atol=1e-12)
    sol = solve_ivp(lotka_volterra_ivp, [0, dt], X.flatten(), args=(alpha, beta, gamma, delta),method='RK45',rtol=1e-13,atol=1e-13)
    return sol.y



# Update functions for the predicted outputs and weights
def update_predicted_outputs(p, z, eta):
    error = z - p
    p_new = p + eta * error
    return p_new

"""def update_weights_using_predictions(w, p, X, eta):
    grad_J_z = p * dtanh(p)
    grad_J_w = np.dot(X.T, grad_J_z)
    norm = np.linalg.norm(grad_J_w)
    norm_grad_J_w = grad_J_w / norm if norm != 0 else grad_J_w
    w_new = w - eta * norm_grad_J_w
    return w_new"""

def backpropagate(X, y_true, z1, z2, z3, eta):
    global w1, w2, w3, b1, b2, b3, p1, p2

    # Compute the gradient of the loss w.r.t. the final output
    grad_loss_z3 = (2 * (z3 - y_true) /2)

    # Backpropagate this gradient to get gradients for w3 and b3 using predicted outputs (p2)
    grad_w3 = np.dot(p2.T, grad_loss_z3)
    grad_b3 = np.sum(grad_loss_z3, axis=0, keepdims=True)
    
    # Backpropagate through the second hidden layer using the predicted output p2
    grad_loss_p2 = np.dot(grad_loss_z3, w3.T) * dtanh(p2)
    grad_w2 = np.dot(p1.T, grad_loss_p2)
    grad_b2 = np.sum(grad_loss_p2, axis=0, keepdims=True)
    
    # Backpropagate through the first hidden layer using the predicted output p1
    grad_loss_p1 = np.dot(grad_loss_p2, w2.T) * dtanh(p1)
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
epochs = 50
dt = 1000.0
learn_rate = 0.01
hidden_layer_learn = .02


# Training loop with loss computation
losses = []  # To store loss at each epoch

#Generate training data
data = solve_odeint_step(initial_conditions, dt, alpha, beta, gamma, delta, 1000)
train_data = data[:,:-1]
labels = data[:,1:]

print(labels.shape)

#Plot training data
plt.figure(figsize=(10, 5))
plt.plot(train_data[0,:], label='Prey')
plt.plot(train_data[1,:], label='Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Training Data')
plt.legend()
plt.show()

for epoch in range(epochs):
    for x,y in zip(train_data.T,labels.T):
        x = x.reshape(1, 2)
        y = y.reshape(1, 2)
        z1, z2, z3 = forward_pass(x)
        p1 = update_predicted_outputs(p1, z1, hidden_layer_learn)
        p2 = update_predicted_outputs(p2, z2, hidden_layer_learn)
        # Backpropagation to update weights using predicted outputs
        backpropagate(x, y, z1, z2, z3, learn_rate)
        
        # Compute and store the loss
        loss = mse_loss(z3, y)
        losses.append(loss)

            #Print loss for this epoch
        #print("Epoch: " + str(epoch) + " Loss: " + str(loss))

#print(p1)
print("Lowest Loss: " + np.array(losses).min().astype(str))
# Plot the loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(losses, label='MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MSE Loss Over Epochs')
plt.legend()
plt.show()



