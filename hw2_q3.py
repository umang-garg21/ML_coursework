import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from hw2_q1 import gen_data, plot_gmm_data

# Problem 3

#N = [200, 400, 800, 2000, 4000, 8000, 20000]
N = [200]
loss_final_N = []

for n in N:
    
    data, label = gen_data(n)
    plot_gmm_data(data, label)
    plt.show()

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def mapFeature(X1, X2):
        degree = 3
        out = np.ones(X1.shape[0])[:,np.newaxis]
        for i in range(1, degree + 1):
            for j in range(i + 1):
                out = np.hstack((out, (X1**(i-j) * X2**j)[:,np.newaxis]))
        return out

    def costFunctionReg(theta, X, y, lmbda):
        eps = 1e-8
        m = len(y)
        temp = sigmoid(np.dot(X, theta))
        temp1 = np.dot(y.T, np.log(temp +eps))
        temp2 = np.dot((1-y).T, np.log(1-temp + eps))
        reg = (lmbda / (2 * m)) * np.sum(theta[1:]**2)
        J = -(1/m) * (temp1 + temp2) + reg
        return J

    def gradRegularization(theta, X, y, lmbda):
        m, n = X.shape
        theta = theta.reshape((n, 1))
        y = y.reshape((m, 1))
        h = sigmoid(X.dot(theta))
        grad = (1/m) * X.T.dot(h-y) + (lmbda/m) * np.vstack(([[0]], theta[1:]))
        return grad.flatten()

    def logisticRegression(X, y, theta, num_steps, lmbda):
        J_history = []
        for i in range(num_steps):
            cost = costFunctionReg(theta, X, y, lmbda)
            J_history.append(cost)
            H = hessian(X, y, lmbda, sigmoid(np.dot(X, theta)))
            H_inv = np.linalg.inv(H)
            grad = gradRegularization(theta, X, y, lmbda)
            theta -= np.dot(H_inv, grad)
            
        return theta, J_history

    def hessian(X,y,lmbda,h_theta):
        
        m,n = X.shape
        H = np.zeros((n,n))
        
        for i in range(m):
            H += (h_theta[i] * (1-h_theta[i]) * np.dot(X[i].reshape(-1,1), X[i].reshape(1,-1)))
            
        H /= m
        H[1:, 1:] += (lmbda/m)*np.eye(n-1)
        
        return H

    num_steps = 100
    lmbda = [0, 0.01, 0.1, 1, 2, 5, 10, 100]
    # Generate all possible combinations of linear, quadratic and cubic features
    X = mapFeature(data[:,0], data[:,1])

    # Initialize model parameters
    initial_theta = np.zeros(X.shape[1])

    # Create a meshgrid of points for the decision boundary
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Train model

    LR_param = []
    loss_final = []
    for i, l in enumerate(lmbda):
        theta, J_history = logisticRegression(X, label, initial_theta, num_steps, l)
        LR_param.append(theta)
        loss_final.append(J_history[-1])
        loss_final_N.append(J_history[-1])

        temp = sigmoid(np.dot(X, theta))
        preds = np.where(temp > 0.5, 1, 0)
        class0_preds = data[preds.ravel()==0, :]
        class1_preds = data[preds.ravel()==1, :]

        X_grid = mapFeature(grid[:,0], grid[:, 1])
        temp = sigmoid(np.dot(X_grid, theta))
        grid_preds = np.where(temp > 0.5, 1, 0)
        grid0_preds = grid[grid_preds.ravel()==0, :]
        grid1_preds = grid[grid_preds.ravel()==1, :]

        # Plot decision boundary and original data
        plt.figure()
        plt.contourf(xx, yy, grid_preds.reshape(xx.shape), alpha=0.4)
        plt.scatter(class0_preds[:, 0], class0_preds[:, 1], c='r', marker='^', label='Class 0')
        plt.scatter(class1_preds[:, 0], class1_preds[:, 1], c='b', marker='s', label='Class 1')
        plt.legend()
        plt.title('N = '+str(N))
        # plt.show()


    x = [i for i in range(num_steps)]
    fig = plt.figure()
    plt.plot(x, J_history)
    plt.title("Cost function with each iteration")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    # plt.show()

    fig2 = plt.figure()
    plt.plot(lmbda, loss_final, marker = 'o')
    plt.title("Final Cost for different lambda")
    plt.ylabel("Final Cost")
    plt.xlabel("lambda")
    plt.show()

"""
plt.figure()
plt.plot(N, loss_final_N, marker = 'o')
plt.title("Loss_vs_N")
plt.xlabel("N")
plt.ylabel("Final Cost")
plt.show()
"""


