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
w1 = np.random.randn(2, 20) / np.sqrt(2)
w2 = np.random.randn(20, 50) / np.sqrt(20)
w3 = np.random.randn(50, 2) / np.sqrt(50)
b1 = np.zeros((1, 20)) 
b2 = np.zeros((1, 50))
b3 = np.zeros((1, 2))

# Predicted outputs for the hidden layers
p1 = np.random.randn(1, 20)
p2 = np.random.randn(1, 50)

# Lotka-Volterra model parameters
alpha, beta, gamma, delta = 1.8, 0.001, .1, .0005

# Initial state vector X
initial_conditions = np.array([[500, 5]])  # Example initial populations

# Forward pass of the neural network
def forward_pass(X):
    global w1, w2, w3, b1, b2, b3
    z1 = relu(np.dot(X, w1) + b1)
    z2 = relu(np.dot(z1, w2) + b2)
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
def update_predicted_outputs(x,y_hat,eta):
    global w1, w2, w3, b1, b2, b3, p1, p2
    z1 = np.dot(x, w1) + b1
    a1 = relu(p1)
    z2 = np.dot(a1, w2) + b2
    a2 = relu(p2)
    z3 = np.dot(a2, w3) + b3
    y_hat = relu(z3)  # Final prediction

    layer1 = relu(p1)
    layer2 = relu(p2)

    # Gradients for z3 (last layer before the output)
    dJ_dz3 = (y_hat - y) * drelu(z3)

    # Gradients for z2 (second hidden layer)
    # We need to consider that z2 and a2 are row vectors now, so we will adjust dimensions accordingly
    dJ_da2 = np.dot(dJ_dz3, w3.T)  # Adjusted to row vector
    dJ_dz2 = dJ_da2 * drelu(a2)

    # Calculate the gradient of the loss with respect to z1 (output of the first hidden layer before activation)
    dJ_da1 = np.dot(dJ_dz2, w2.T)  # Gradient with respect to the output of the first hidden layer after activation
    dJ_dz1 = dJ_da1 * drelu(a1)  # Gradient with respect to z1 before activation

    # Update rules for p1 and p2
    p1_updated = p1 - eta * dJ_dz1
    p2_updated = p2 - eta * dJ_dz2

    return p1_updated, p2_updated

def update_weights(X, y_true, z1, z2, z3, eta):
    global w1, w2, w3, b1, b2, b3, p1, p2

    # Update rules for the output layer (traditional approach)
    loss_output = z3 - y_true
    grad_w3 = np.dot(p2.T, loss_output)
    grad_b3 = np.sum(loss_output, axis=0, keepdims=True)
    w3 -= eta * grad_w3
    b3 -= eta * grad_b3

    # Update rules for the second hidden layer
    loss_hidden_2 = z2 - p2
    grad_w2 = np.dot(p1.T, loss_hidden_2) * drelu(p2)
    grad_b2 = np.sum(loss_hidden_2, axis=0, keepdims=True)
    w2 -= eta * grad_w2
    b2 -= eta * grad_b2

    # Update rules for the first hidden layer
    loss_hidden_1 = z1 - p1
    grad_w1 = np.dot(X.T, loss_hidden_1) * drelu(p1)
    grad_b1 = np.sum(loss_hidden_1, axis=0, keepdims=True)
    w1 -= eta * grad_w1
    b1 -= eta * grad_b1



def min_max_normalize(data):
    return (data - data.min()) / (data.max() - data.min())

def rmse_loss(y_pred, y_true):
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    return rmse


# Training loop parameters
epochs = 100
dt = 100.0
learn_rate = 1e-3
learn_rate_decay=1.0
hidden_layer_learn = .001



# Training loop with loss computation
losses = []  # To store loss at each epoch

#Generate training data
data = solve_odeint_step(initial_conditions, dt, alpha, beta, gamma, delta, 10000)
train_data = data[:,:-1]
labels = data[:,1:]

train_data = min_max_normalize(train_data)
labels = min_max_normalize(labels)

print(labels.shape)

#Plot training data
plt.figure(figsize=(10, 5))
plt.plot(train_data[0,:], label='Prey')
plt.plot(train_data[1,:], label='Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Training Data')
plt.legend()
plt.savefig('./new_update/training_data.pdf')


subset_size = int(train_data.shape[1]/40) # Size of the random subset

#Train
for epoch in range(epochs):
    # First loop to update the weights
    for x, y in zip(train_data.T, labels.T):
        # Randomly select a subset of indices for the second loop
        subset_indices = np.random.choice(train_data.shape[1], subset_size, replace=False)
        x_subset = train_data[:, subset_indices].T
        y_subset = labels[:, subset_indices].T

        # Second loop to update the predicted outputs
        for x2, y2 in zip(x_subset, y_subset):
            x2 = x2.reshape(1, 2)
            #z1, z2, z3 = forward_pass(x2)

            # Update predicted outputs for hidden layers
            p1,p2 = update_predicted_outputs(x2,y2,hidden_layer_learn)
            #p2 = update_predicted_outputs(p2, z2, hidden_layer_learn)

        x = x.reshape(1, 2)
        y = y.reshape(1, 2)
        z1, z2, z3 = forward_pass(x)

        # Backpropagation to update weights using predicted outputs
        update_weights(x, y, z1, z2, z3, learn_rate)

        # Compute and store the loss
        loss = rmse_loss(z3, y)
        losses.append(loss)

    # Optionally, print loss for the epoch
    print(f"Epoch: {epoch+1}, Loss: {np.mean(losses[-len(train_data.T):])}")
    # Plot the neural networks predictions
    # Convert predictions to a numpy array for easier indexing
    predictions = []
    for x in train_data.T:
        x = x.reshape(1, 2)
        _, _, z3 = forward_pass(x)
        predictions.append(z3)
    predictions = np.array(predictions).squeeze()

    # Plot the original data and the predictions
    plt.figure(figsize=(10, 5))

    # Plot original data
    plt.plot(labels.T[:, 0], label='Actual Prey', linestyle='--')
    plt.plot(labels.T[:, 1], label='Actual Predator', linestyle='--')

    # Plot predictions with a dashed line
    plt.plot(predictions[:, 0], label='Predicted Prey', linestyle='-')
    plt.plot(predictions[:, 1], label='Predicted Predator', linestyle='-')

    # Adding labels and title
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title(f'Diff Learn Predictions vs Actual Data for Epoch {epoch+1}')

    # Show the legend
    plt.legend()

    # Save the plot
    plt.savefig(f'./new_update/prediction_{epoch+1}.pdf')
    plt.close()

    #Decrease learning rate
    learn_rate = learn_rate*learn_rate_decay

# Plotting code remains unchanged


#print(p1)
#print("Lowest Loss: " + np.array(losses).min().astype(str))
# Plot the loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(losses, label='MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MSE Loss Over Epochs')
plt.legend()
plt.savefig('./new_update/loss.pdf')

# Calculate predictions
predictions = []
for x in train_data.T:
    x = x.reshape(1, 2)
    _, _, z3 = forward_pass(x)
    predictions.append(z3)

# Convert predictions to a numpy array for easier indexing
predictions = np.array(predictions).squeeze()

# Plot the original data and the predictions
plt.figure(figsize=(10, 5))

# Plot original data
plt.plot(labels.T[:, 0], label='Actual Prey', linestyle='--')
plt.plot(labels.T[:, 1], label='Actual Predator', linestyle='--')

# Plot predictions with a dashed line
plt.plot(predictions[:, 0], label='Predicted Prey', linestyle='-')
plt.plot(predictions[:, 1], label='Predicted Predator', linestyle='-')

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Diff Learn Predictions vs Actual Data')

# Show the legend
plt.legend()

# Save the plot
plt.savefig('./new_update/final_predictions.pdf')
#plt.show()  # To display the plot inline if using a notebook


