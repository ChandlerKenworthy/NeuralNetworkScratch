# Implementation of a deep neural network using Python from first principles (i.e. no PyTorch/Tensorflow)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Activation function allows models to learn non-linearity
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return np.where(x > 0, 1, 0)


# For continuous output
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

class Neuron:
    # Represents a singular neuron in a fully connected neural network
    def __init__(self, weights, biases):
        self.weights = weights
        self.bias = biases

    def update_weights(self, step):
        self.weights -= step

    def update_bias(self, step):
        self.bias -= step

    def forward(self, inputs, activation='sigmoid'):
        total = np.dot(self.weights, inputs) + self.bias
        if activation == 'sigmoid':
            total = sigmoid(total)
        if activation == 'relu':
            total = relu(total)
        return total

class Network:
    def __init__(self):
        self.lr = 0.01 # learning rate
        self.epochs = 100

        # Initalise some random weights to start with
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

        self.h1 = Neuron(np.array([self.w1, self.w2]), self.b1)
        self.h2 = Neuron(np.array([self.w3, self.w4]), self.b2)
        self.o1 = Neuron(np.array([self.w5, self.w6]), self.b3)

        # For backprop intermediate step values in forward pass
        self.sum_h1 = 0
        self.sum_h2 = 0
        self.sum_o1 = 0
        self.h1_value = 0
        self.h2_value = 0
        self.losses = []

    def set_epochs(self, epochs):
        self.epochs = epochs

    def forward(self, inputs):
        # inputs is a numpy array with two input features x[0] = gender, x[1] = weight --> predict height
        self.sum_h1 = self.h1.forward(inputs, activation=None)
        self.h1_value = sigmoid(self.sum_h1)

        self.sum_h2 = self.h2.forward(inputs, activation=None)
        self.h2_value = sigmoid(self.sum_h2)

        self.sum_o1 = self.o1.forward(np.array([self.h1_value, self.h2_value]), activation=None)

        return sigmoid(self.sum_o1)
    
    def train(self, data, all_y_trues):
        """
        - data is a (n_samples x n_features) numpy array,
        - all_y_trues is a numpy array with n elements.
        Elements in all_y_trues correspond to those in data.
        """

        for epoch in range(self.epochs + 1):
            # Normally we "batch" the input in this case we do one example at a time (not optimal at all)
            for x, y_true in zip(data, all_y_trues):
                # Forward pass
                y_pred = self.forward(x)

                # Backprop - calculate partial derivatives
                dL_dypred = -2 * (y_true - y_pred) # Loss = (y_true - y_pred)**2 for a single sample

                # Neuron o1
                d_ypred_d_w5 = self.h1_value * dsigmoid(self.sum_o1)
                d_ypred_d_w6 = self.h2_value * dsigmoid(self.sum_o1)
                d_ypred_d_b3 = dsigmoid(self.sum_o1)

                d_ypred_d_h1 = self.w5 * dsigmoid(self.sum_o1)
                d_ypred_d_h2 = self.w6 * dsigmoid(self.sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * dsigmoid(self.sum_h1)
                d_h1_d_w2 = x[1] * dsigmoid(self.sum_h1)
                d_h1_d_b1 = dsigmoid(self.sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * dsigmoid(self.sum_h2)
                d_h2_d_w4 = x[1] * dsigmoid(self.sum_h2)
                d_h2_d_b2 = dsigmoid(self.sum_h2)

                # Update weights and biases
                # Neuron h1
                h1w_step = np.array([
                    self.lr * dL_dypred * d_ypred_d_h1 * d_h1_d_w1,
                    self.lr * dL_dypred * d_ypred_d_h1 * d_h1_d_w2
                ])
                self.h1.update_weights(h1w_step)
                self.h1.update_bias(self.lr * dL_dypred * d_ypred_d_h1 * d_h1_d_b1)

                # Neuron h2
                h2w_step = np.array([
                    self.lr * dL_dypred * d_ypred_d_h2 * d_h2_d_w3,
                    self.lr * dL_dypred * d_ypred_d_h2 * d_h2_d_w4
                ])
                self.h2.update_weights(h2w_step)
                self.h2.update_bias(self.lr * dL_dypred * d_ypred_d_h2 * d_h2_d_b2)

                # Neuron o1
                o1w_step = np.array([
                    self.lr * dL_dypred * d_ypred_d_w5,
                    self.lr * dL_dypred * d_ypred_d_w6
                ])
                self.o1.update_weights(o1w_step)
                self.o1.update_bias(self.lr * dL_dypred * d_ypred_d_b3)

            # Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.forward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                self.losses.append(loss)
                print("Epoch %d loss: %.3f" % (epoch, loss))


model = Network()
n_epochs = 500
model.set_epochs(n_epochs)

# Get training data
df = pd.read_csv("synthetic_sex_data.csv")
first_person = df.iloc[0]
df = df.drop(0, axis=0)

# Normalise the inputs for better weight convergence
height_mean = df['height'].mean()
height_std = df['height'].std()
weight_mean = df['weight'].mean()
weight_std = df['weight'].std()

df['height'] = (df['height'] - height_mean) / height_std # target
df['weight'] = (df['weight'] - weight_mean) / weight_std # input 1

X_train = df[['weight', 'is_male']].to_numpy()
y_train = df['height'].to_numpy()

model.train(X_train, y_train)

X_test = first_person[['weight', 'is_male']].to_numpy()
X_test[0] = (X_test[0] - weight_mean) / weight_std

prediction_scaled = model.forward(X_test)
print(prediction_scaled)

pred = (prediction_scaled * height_std) + height_mean
print(f"Actual height = {first_person['height']}\tPrediction = {pred:.2f} cm")

plt.plot(np.arange(0, n_epochs + 1, 10), model.losses)
plt.show()