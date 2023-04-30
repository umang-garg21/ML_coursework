import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from hw2_q1 import gen_data, plot_gmm_data

# Problem 2

N = 200
data, label = gen_data(N)
plot_gmm_data(data, label)
plt.show()

# Define prior probabilities
prior_0 = 0.5 # Prior probability for class 0
prior_1 = 0.5  # Prior probability for class 1
data_0 = data[label.ravel()==0, :]
data_1 = data[label.ravel()==1, :]

# Define likelihood functions
mean_0 = np.mean(data_0, axis=0)
cov_0 = np.cov(data_0, rowvar=False)
mean_1 = np.mean(data_1, axis=0)
cov_1 = np.cov(data_1, rowvar=False)

def likelihood_0(x):
    # calculate the discriminant function for class 0
    # print(np.linalg.inv(cov_0) )
    g0 = -0.5 * (x - mean_0) @ np.linalg.inv(cov_0) @ (x - mean_0).T - 0.5 * np.log(np.linalg.det(cov_0))
    return g0

def likelihood_1(x):  
    # calculate the discriminant function for class 1
    g1 = -0.5 * (x - mean_1) @ np.linalg.inv(cov_1) @ (x - mean_1).T - 0.5 * np.log(np.linalg.det(cov_1))
    return g1

posterior_0 = np.zeros(data.shape[0])
posterior_1 = np.zeros(data.shape[0])
# Calculate posterior probabilities
for i, ele in enumerate(data):
    posterior_0[i] = likelihood_0(ele) + np.log(prior_0)
    posterior_1[i] = likelihood_1(ele) + np.log(prior_1)

preds = np.where(posterior_1 > posterior_0, 1, 0)
class0_preds = data[preds.ravel()==0, :]
class1_preds = data[preds.ravel()==1, :]
print(preds)

# Create a meshgrid of points for the decision boundary
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

# Calculate posterior probabilities for each point on the grid

grid_posterior_0 = np.zeros(grid.shape[0])
grid_posterior_1 = np.zeros(grid.shape[0])
# Calculate posterior probabilities
for i, ele in enumerate(grid):
    grid_posterior_0[i] = likelihood_0(ele) + np.log(prior_0)
    grid_posterior_1[i] = likelihood_1(ele) + np.log(prior_1)

grid_predictions = np.where(grid_posterior_1 > grid_posterior_0, 1, 0)

# Plot decision boundary and original data
plt.contourf(xx, yy, grid_predictions.reshape(xx.shape), alpha=0.4)
plt.scatter(class0_preds[:, 0], class0_preds[:, 1], c='r', marker='^', label='Class 0')
plt.scatter(class1_preds[:, 0], class1_preds[:, 1], c='b', marker='s', label='Class 1')
plt.legend()
plt.show()

test_Sample = [-8, 7]
test_posterior_0 = likelihood_0(test_Sample) * prior_0
test_posterior_1 = likelihood_1(test_Sample) * prior_1
test_pred = np.where(test_posterior_1 > test_posterior_0, 1, 0)
print(test_pred)