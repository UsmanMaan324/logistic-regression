
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

reg_factor = 1
alpha = 0.001
def mapFeature(X1, X2, degree):
    res = np.ones(X1.shape[0])
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            res = np.column_stack((res, (X1 ** (i - j)) * (X2 ** j)))

    return res


def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))


def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)


def cost_function(theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    reg_term = (reg_factor * sum(theta[1:] ** 2)) / (2 * m)
    total_cost = total_cost + reg_term
    return total_cost


def gradient(theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)


def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))



def fit( x, y, theta):
    print("first cost",cost_function(theta,x,y))
    while cost_function(theta,x,y) >0.7:
        utheta = gradient(theta,x,y)
        theta[0][0] = utheta[0][0]
        theta[1:] = theta[1:] * (1 - alpha(reg_factor / x.shape[0])) - alpha * utheta[1:]

        print("value of cost", cost_function(theta, x, y))

    #("shape of X",x[:,0])
    #opt_weights = fmin_tnc(func=cost_function, x0=theta,
                  #fprime=gradient,args=(x, y.flatten()))
    return theta


# define a function to plot the decision boundary
def plotDecisionBoundary(theta, degree, axes, data):
    U = np.linspace(min(data.iloc[:,0]), max(data.iloc[:, 0]), 50)
    V = np.linspace(min(data.iloc[:, 1]), max(data.iloc[:, 1]), 50)
    #U, V = np.meshgrid(u, v)
    print(U)
    # convert U, V to vectors for calculating additional features
    # using vectorized implementation
    U = np.ravel(U)
    V = np.ravel(V)
    Z = np.zeros((len(U) * len(V)))

    X_poly = mapFeature(U, V, degree)
    #X_poly = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
    Z = sigmoid(X_poly.dot(theta))

    # reshape U, V, Z back to matrix
   # U = U.reshape((len(u), len(v)))
   # V = V.reshape((len(u), len(v)))
   # Z = Z.reshape((len(u), len(v)))

    cs = axes.contour(U, V, Z, 0)
    plt.legend(labels=['y = 1', 'y = 0', 'Decision Boundary'])
    #plt.show()
    return cs

# load the data from the file
data = pd.read_csv("ex2data2.txt", header=None)

# X = feature values, all the columns except the last column
X = data.iloc[:, :-1]
print("max 1", max(X.iloc[:, 0]))
print("max 2", max(X.iloc[:, 1]))

feature_matrix = mapFeature(X.iloc[:,0],X.iloc[:,1],6)
#feature_matrix = np.c_[np.ones((feature_matrix.shape[0], 1)), feature_matrix]
print(feature_matrix.shape)
# y = target values, last column of the data frame
y = data.iloc[:, -1]
y = y[:, np.newaxis]

theta = np.zeros((feature_matrix.shape[1], 1))
parameter = fit(feature_matrix,y,theta)
print(parameter)
# filter out the applicants that got admitted
admitted = data.loc[y == 1]
# filter out the applicants that din't get admission
not_admitted = data.loc[y == 0]
fig, axes = plt.subplots(1, 1)
axes.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], marker ='+', s=25, color = 'black', label='y = 1')
axes.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1],marker = 'o', s=25, color = 'y', label='y = 0')
#plt.legend()
plotDecisionBoundary(theta, 6, axes, X)