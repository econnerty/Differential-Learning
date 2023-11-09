import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the activation function and its derivative
def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1.0 - np.tanh(x) ** 2

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
alpha, beta, gamma, delta = 0.2, 0.1, 0.3, 0.7

# Initial state vector X
X = np.array([[10, 5]])  # Example initial populations

# Forward pass of the neural network
def forward_pass(X, w1, b1, w2, b2, w3, b3):
    z1 = tanh(np.dot(X, w1) + b1)
    z2 = tanh(np.dot(z1, w2) + b2)
    z3 = tanh(np.dot(z2, w3) + b3)
    return z1, z2, z3


# Define the Lotka-Volterra model for use with solve_ivp
def lotka_volterra_ivp(t, X, alpha, beta, gamma, delta):
    x, y = X
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Use solve_ivp to solve the Lotka-Volterra model over one time step
def solve_ivp_step(X, dt, alpha, beta, gamma, delta):
    sol = solve_ivp(lotka_volterra_ivp, [0, dt], X.flatten(), args=(alpha, beta, gamma, delta))
    X_next = sol.y[:, -1].reshape(1, -1)  # Only take the last point
    return X_next



# Update functions for the predicted outputs and weights
def update_predicted_outputs(p, z, eta):
    error = z - p
    p_new = p + eta * error
    return p_new

def update_weights_using_predictions(w, p, X, eta):
    grad_J_z = p * dtanh(p)
    grad_J_w = np.dot(X.T, grad_J_z)
    norm = np.linalg.norm(grad_J_w)
    norm_grad_J_w = grad_J_w / norm if norm != 0 else grad_J_w
    w_new = w - eta * norm_grad_J_w
    return w_new

def backpropagate(X, y_true, z1, z2, z3, w1, w2, w3, p1, p2, eta):
    # Compute the gradient of the loss w.r.t. the final output
    grad_loss_z3 = 2 * (z3 - y_true) * dtanh(z3)

    # Backpropagate this gradient to get gradients for w3 using predicted outputs (p2)
    grad_w3 = np.dot(p2.T, grad_loss_z3)
    
    # Backpropagate through the second hidden layer using the predicted output p2
    grad_loss_p2 = np.dot(grad_loss_z3, w3.T) * dtanh(p2)
    grad_w2 = np.dot(p1.T, grad_loss_p2)
    
    # Backpropagate through the first hidden layer using the predicted output p1
    grad_loss_p1 = np.dot(grad_loss_p2, w2.T) * dtanh(p1)
    grad_w1 = np.dot(X.T, grad_loss_p1)

    # Update the weights
    w1 -= eta * grad_w1
    w2 -= eta * grad_w2
    w3 -= eta * grad_w3
    
    return w1, w2, w3


# Mean Squared Error (MSE) loss function
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)



def generate_lotka_volterra_data(alpha, beta, gamma, delta, initial_conditions, timesteps, dt):
    times = np.arange(0, timesteps*dt, dt)
    solution = solve_ivp(lotka_volterra_ivp, [times[0], times[-1]], initial_conditions, t_eval=times, args=(alpha, beta, gamma, delta))
    X = solution.y.T[:-1]  # all but the last to be used as inputs
    Y = solution.y.T[1:]   # all but the first to be used as labels
    return X, Y



# Training loop parameters
epochs = 10
dt = 0.01
learn_rate = .01
hidden_layer_learn = .002
#decay_rate = 0.999  # Decay rate per epoch
# Generate data
initial_conditions = [1000, 10]
timesteps = 1000  # Adjust as needed
dt = 0.1
X_data, Y_data = generate_lotka_volterra_data(alpha, beta, gamma, delta, initial_conditions, timesteps, dt)

# Training loop with loss computation
losses = []  # To store loss at each epoch
for epoch in range(epochs):
    for X,X_next in zip(X_data,Y_data):
        X = X.reshape(1,2)
        X_next = X_next.reshape(1,2)
        z1, z2, z3 = forward_pass(X, w1, b1, w2, b2, w3, b3)
        p1 = update_predicted_outputs(p1, z1, hidden_layer_learn)
        p2 = update_predicted_outputs(p2, z2, hidden_layer_learn)

        # Compute and store the loss
        loss = mse_loss(z3, X_next)
        losses.append(loss)
        X = X_next
    # Backpropagation to update weights using predicted outputs
    z1, z2, z3 = forward_pass(X, w1, b1, w2, b2, w3, b3)
    w1, w2, w3 = backpropagate(X, X_next, z1, z2, z3, w1, w2, w3, p1, p2, learn_rate)





print('Final loss:', losses[-1])

# Plot the loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(losses, label='MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MSE Loss Over Epochs')
plt.legend()
plt.show()
