import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

data = np.genfromtxt('winequality-red.csv', delimiter=";")

print('RAW DATA')

data = data[1:] # remove NaNs
print()

X = data[:, :-1] # get all of the rows and all but the last column (the last column is the labels)
print('FEATURES (design matrix), X:')
print(X)
print('Design matrix shape:', X.shape)
print()
print('CHOSEN FEATURE:', )



Y = data[:, -1] # get the last column as the labels
print('LABELS, Y:')
print(Y)
print('Labels shape:', Y.shape)

m = 20 # how many examples do we want?
order = 9
coeffs = np.random.randn(order + 1)
print('X:',X, '\n')
print('Y:',Y, '\n')
print('Ground truth coefficients:', coeffs, '\n')

class MultiVariableLinearHypothesis:
    def __init__(self, n_features, regularisation_factor): ## add regularisation factor as parameter
        self.n_features = n_features
        self.regularisation_factor = regularisation_factor ## add self.regularisation factor
        self.b = np.random.randn()
        self.w = np.random.randn(n_features)
        
    def __call__(self, X): # what happens when we call our model, input is of shape (n_examples, n_features)
        y_hat = np.matmul(X, self.w) + self.b # make prediction, now using vector of weights rather than a single value
        return y_hat # output is of shape (n_examples, 1)
    
    def update_params(self, new_w, new_b):
        self.w = new_w
        self.b = new_b
        
    def calc_deriv(self, X, y_hat, labels):
        diffs = y_hat-labels
        dLdw = 2 * np.array([np.sum(diffs * X[:, i]) / m for i in range(self.n_features)]) 
        dLdw += 2 * self.regularisation_factor * self.w ## add regularisation term gradient
        dLdb = 2 * np.sum(diffs) / m
        return dLdw, dLdb

def create_polynomial_inputs(X, order=3):
    new_dataset = np.array([np.power(X, i) for i in range(1, order + 1)]).T
    return new_dataset # new_dataset should be shape [m, order]

def train(num_epochs, X, Y, H):
    for e in range(num_epochs): # for this many complete runs through the dataset
        y_hat = H(X) # make predictions
        dLdw, dLdb = H.calc_deriv(X, y_hat, Y) # calculate gradient of current loss with respect to model parameters
        new_w = H.w - learning_rate * dLdw # compute new model weight using gradient descent update rule
        new_b = H.b - learning_rate * dLdb # compute new model bias using gradient descent update rule
#       norm = np.linalg.norm([[new_w - H.w], [new_b - H.w]], 2)
        H.update_params(new_w, new_b) # update model weight and bias
#     print(f'THE MODEL DIDNT CONVERGE IN {num_epochs} EPOCHS')
def plot_h_vs_y(X, y_hat, Y):
    plt.figure()
    plt.scatter(X, Y, c='r', label='Label')
    plt.scatter(X, y_hat, c='b', label='Prediction', marker='x')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


num_epochs = 500
learning_rate = 0.1
regularisation_factor = 0.1
highest_order_power = 9

polynomial_augmented_inputs = create_polynomial_inputs(X, highest_order_power) # need normalisation to put higher coefficient variables on the same order of magnitude as the others
H = MultiVariableLinearHypothesis(n_features=highest_order_power, regularisation_factor=regularisation_factor)

train(num_epochs, polynomial_augmented_inputs, Y, H)
plot_h_vs_y(X, H(polynomial_augmented_inputs), Y)
print(H.w)