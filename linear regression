import numpy as np
import matplotlib.pyplot as plt

def estimate_coeff(x, y):
    n = np.size(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    SS_xy = np.sum((x - mean_x) * (y - mean_y))
    SS_xx = np.sum((x - mean_x) ** 2) 
    b1 = SS_xy / SS_xx
    b0 = mean_y - b1 * mean_x
    return b0, b1

def plots(x, y, b):
    plt.scatter(x, y)
    y_pred = b[0] + b[1] * x
    plt.plot(x, y_pred, color='red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
a = estimate_coeff(x, y)
print("The Estimated Coefficients b0 and b1 are:", a)
plots(x, y, a)
