import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

# Problem 1
mu_a = np.array([4, 0])
mu_b = np.array([-3, 3])
mu_c = np.array([0, 0])
mu_d = np.array([-6, -4])
pi_a = 2/3
pi_b = 1/3
pi_c = 3/4
pi_d = 1/4

def generate_cov_mtx(lambd_1: float, lambd_2: float, theta: float) -> np.ndarray:
    U = np.array([[np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]]).T
    L = np.diag([lambd_1, lambd_2])
    return U @ L @ U.T

C_a = generate_cov_mtx(1, 4, 0)
print(C_a)
C_b = generate_cov_mtx(1, 4, np.pi/4)
print(C_b)

def generate_gmm_data(num_data: int) -> (np.ndarray, np.ndarray):
    num_a = round(num_data * pi_a)
    num_b = num_data - num_a
    num_c = round(num_data * pi_c)
    num_d = num_data - num_c
    C_a = generate_cov_mtx(1, 4, 0)
    C_b = generate_cov_mtx(1, 4, np.pi/4)
    C_c = generate_cov_mtx(1, 2, np.pi/3)
    C_d = generate_cov_mtx(2, 1, np.pi/4)
    data_a = np.random.multivariate_normal(mu_a.ravel(), C_a, num_a)
    data_b = np.random.multivariate_normal(mu_b.ravel(), C_b, num_b)
    data_c = np.random.multivariate_normal(mu_c.ravel(), C_c, num_c)
    data_d = np.random.multivariate_normal(mu_d.ravel(), C_d, num_d)
    y = np.concatenate((np.zeros([num_data, 1]), np.ones([num_data, 1])))
    data = np.concatenate((data_a, data_b, data_c, data_d))
    return(data, y)

def plot_gmm_data(data: np.ndarray, label: np.ndarray):
    data_0 = data[label.ravel()==0, :]
    data_1 = data[label.ravel()==1, :]
    plt.figure()
    plt.scatter(data_0[:,0], data_0[:,1], c='red', alpha=0.6, marker='^', label='Class 0')
    plt.scatter(data_1[:,0], data_1[:,1], c='blue', alpha=0.6, label='Class 1')
    plt.legend()
    return

def gen_data(N):
    data, label = generate_gmm_data(N)
    return data, label
