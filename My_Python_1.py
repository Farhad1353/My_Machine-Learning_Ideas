import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

def plot_data(X, Y):
    plt.figure() # create a figure
    plt.scatter(X, Y, c='r') # plot the data in color=red
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def sample_polynomial_data(m=20, order=3, _range=1):
    coeffs = np.random.randn(order + 1) # initialise random coefficients for each order of the input + a constant offset
    print(Polynomial(coeffs))
    poly_func = np.vectorize(Polynomial(coeffs)) # 
    X = np.random.randn(m)
    X = np.random.uniform(low=-_range, high=_range, size=(m,))
    Y = poly_func(X)
    return X, Y, coeffs #returns X (the input), Y (labels) and coefficients for each power

m = 20 # how many examples do we want?
order = 3 # how many powers do we want to raise our input data to?
X, Y, ground_truth_coeffs = sample_polynomial_data(m, order)
print('X:',X, '\n')
print('Y:',Y, '\n')
print('Ground truth coefficients:', ground_truth_coeffs, '\n')
plot_data(X, Y)



# THIS IS ALL CODE WE'VE ALREADY COVERED, JUST RUN THIS CELL
class LinearHypothesis:
    def __init__(self):
        self.b = np.random.randn()
        self.w = np.random.randn()
        
    def __call__(self, X): #input is of shape (n_datapoints, n_vars)
        y_hat = self.w * X + self.b # linear hypothesis
        return y_hat # output is of shape (n_datapoints, 1)
    
    def update_params(self, new_w, new_b):
        self.w = new_w
        self.b = new_b
        
    def calc_deriv(self, X, y_hat, labels): 
        """calculates the gradient assuming that we are using mean squared error loss"""
        diffs = y_hat - labels ## calculate error
        dLdw = 2*np.sum(diffs*X)/m ## calculate gradient of MSE loss with respect to model weights
        dLdb = 2*np.sum(diffs)/m ## calculate gradient of MSE loss with respect to model bias
        return dLdw, dLdb
    
num_epochs = 100
learning_rate = 0.1
H = LinearHypothesis() ## instantiate linear hypothesis model model

def train(num_epochs, X, Y, H):
    for e in range(num_epochs): # for this many complete runs through the dataset
        y_hat = H(X) # make predictions
        # print(X.shape)
        # print(y_hat.shape)
        # print(Y.shape)
        dLdw, dLdb = H.calc_deriv(X, y_hat, Y) # calculate gradient of current loss with respect to model parameters
        new_w = H.w - learning_rate * dLdw # compute new model weight using gradient descent update rule
        new_b = H.b - learning_rate * dLdb # compute new model bias using gradient descent update rule
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
    
train(num_epochs, X, Y, H) # train model and plot cost curve
plot_h_vs_y(X, H(X), Y) # plot predictions and true data


# THIS IS ALL CODE WE'VE ALREADY COVERED, JUST RUN THIS CELL
class LinearHypothesis:
    def __init__(self):
        self.b = np.random.randn()
        self.w = np.random.randn()
        
    def __call__(self, X): #input is of shape (n_datapoints, n_vars)
        y_hat = self.w * X + self.b # linear hypothesis
        return y_hat # output is of shape (n_datapoints, 1)
    
    def update_params(self, new_w, new_b):
        self.w = new_w
        self.b = new_b
        
    def calc_deriv(self, X, y_hat, labels): 
        """calculates the gradient assuming that we are using mean squared error loss"""
        diffs = y_hat - labels ## calculate error
        dLdw = 2*np.sum(diffs*X)/m ## calculate gradient of MSE loss with respect to model weights
        dLdb = 2*np.sum(diffs)/m ## calculate gradient of MSE loss with respect to model bias
        return dLdw, dLdb
    
num_epochs = 100
learning_rate = 0.1
H = LinearHypothesis() ## instantiate linear hypothesis model model

def train(num_epochs, X, Y, H):
    for e in range(num_epochs): # for this many complete runs through the dataset
        y_hat = H(X) # make predictions
        # print(X.shape)
        # print(y_hat.shape)
        # print(Y.shape)
        dLdw, dLdb = H.calc_deriv(X, y_hat, Y) # calculate gradient of current loss with respect to model parameters
        new_w = H.w - learning_rate * dLdw # compute new model weight using gradient descent update rule
        new_b = H.b - learning_rate * dLdb # compute new model bias using gradient descent update rule
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
    
train(num_epochs, X, Y, H) # train model and plot cost curve
plot_h_vs_y(X, H(X), Y) # plot predictions and true data

num_epochs = 200
learning_rate = 0.1
highest_order_power = 4

polynomial_augmented_inputs = create_polynomial_inputs(X, highest_order_power) ## need normalization to put higher coefficient variables on the same order of magnitude as the others
H = MultiVariableLinearHypothesis(n_features=highest_order_power) ## initialise multivariate regression model

train(num_epochs, polynomial_augmented_inputs, Y, H) ## train model
plot_h_vs_y(X, H(polynomial_augmented_inputs), Y)


learning_rate = 0.1
highest_order_power = 3
# num_epochs = 10
m = 50

X, Y, ground_truth_coeffs = sample_polynomial_data(m, 1, _range=10)
new_dataset = create_polynomial_inputs(X, highest_order_power)

H = MultiVariableLinearHypothesis(n_features=highest_order_power)

print('Training')
train(num_epochs, new_dataset, Y, H) # train model and plot cost curve
print('Weights after training:', H.w)
print('True weights:', ground_truth_coeffs[1:])
plot_h_vs_y(X, H(new_dataset), Y)



learning_rate = 0.1
highest_order_power = 3
# num_epochs = 10
m = 50

X, Y, ground_truth_coeffs = sample_polynomial_data(m, 1, _range=10)
new_dataset = create_polynomial_inputs(X, highest_order_power)

H = MultiVariableLinearHypothesis(n_features=highest_order_power)

print('Training')
train(num_epochs, new_dataset, Y, H) # train model and plot cost curve
print('Weights after training:', H.w)
print('True weights:', ground_truth_coeffs[1:])
plot_h_vs_y(X, H(new_dataset), Y)



