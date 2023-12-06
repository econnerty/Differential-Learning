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
initial_conditions = np.array([[50, 5]])  # Example initial populations

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

    # Update rules for the output layer (traditional approach)
    loss_output = z3 - y_true
    grad_w3 = np.dot(p2.T, loss_output)
    grad_b3 = np.sum(loss_output, axis=0, keepdims=True)
    w3 -= eta * grad_w3
    b3 -= eta * grad_b3

    # Update rules for the second hidden layer
    loss_hidden_2 = z2 - p2
    grad_w2 = np.dot(p1.T, loss_hidden_2) * dtanh(z2)
    grad_b2 = np.sum(loss_hidden_2, axis=0, keepdims=True)
    w2 -= eta * grad_w2
    b2 -= eta * grad_b2

    # Update rules for the first hidden layer
    loss_hidden_1 = z1 - p1
    grad_w1 = np.dot(X.T, loss_hidden_1) * dtanh(z1)
    grad_b1 = np.sum(loss_hidden_1, axis=0, keepdims=True)
    w1 -= eta * grad_w1
    b1 -= eta * grad_b1



# Mean Squared Error (MSE) loss function
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Training loop parameters
epochs = 1000
dt = 1000.0
learn_rate = 0.001
hidden_layer_learn = .09


# Training loop with loss computation
losses = []  # To store loss at each epoch

#Generate training data
data = solve_odeint_step(initial_conditions, dt, alpha, beta, gamma, delta, 10000)
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
plt.savefig('training_data.png')

# Training loop with separate updates for predicted layers and weights
for epoch in range(epochs):
    # First loop to update the predicted outputs
    for x, y in zip(train_data.T, labels.T):
        x = x.reshape(1, 2)
        z1, z2, z3 = forward_pass(x)
        # Update predicted outputs for hidden layers
        p1 = update_predicted_outputs(p1, z1, hidden_layer_learn)
        p2 = update_predicted_outputs(p2, z2, hidden_layer_learn)

    # Second loop to update the weights
    for x, y in zip(train_data.T, labels.T):
        x = x.reshape(1, 2)
        y = y.reshape(1, 2)
        z1, z2, z3 = forward_pass(x)
        # Backpropagation to update weights using predicted outputs
        backpropagate(x, y, z1, z2, z3, learn_rate)

        # Compute and store the loss
        loss = mse_loss(z3, y)
        losses.append(loss)

    # Optionally, print loss for the epoch
    print(f"Epoch: {epoch}, Loss: {np.mean(losses[-len(train_data.T):])}")

# Plotting code remains unchanged


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

# Plot the neural networks predictions
predictions = []
for x, y in zip(train_data.T, labels.T):
    x = x.reshape(1, 2)
    y = y.reshape(1, 2)
    z1, z2, z3 = forward_pass(x)
    predictions.append(z3)

#Plot the predictions
plt.figure(figsize=(10, 5))
plt.plot(np.array(predictions)[:,:,0], label='Prey')
plt.plot(np.array(predictions)[:,:,1], label='Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Predictions')
plt.legend()
plt.show()



