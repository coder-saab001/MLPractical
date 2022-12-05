import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def estimate_coeff(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    SS_xy = np.sum(x*y) - (np.sum(x)*np.sum(y))/n
    SS_xx = np.sum(x*x) - (np.sum(x)*np.sum(x))/n
    b = SS_xy/SS_xx
    a = y_mean - x_mean * b
    return a, b

def plot_regression_line(x, y, a, b):
    plt.scatter(x, y)
    y_predicted = a + b * x
    plt.plot(x, y_predicted)
    plt.show()

data = pd.read_csv('data.csv')
x = np.array(data['YearsExperience'])
y = np.array(data['Salary'])

a, b = estimate_coeff(x, y)
print("Estimated Coefficients:\na = ", a, "\nb = ", b)
plot_regression_line(x, y, a, b)