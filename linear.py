import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt


def LinearRegression(X, Y):
    n, m = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1)
    try:
        A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    except:
        def cost(b):
            return np.sum((X.dot(b) - Y) ** 2) / (2 * n)

        A = opt.minimize(cost, np.zeros(m + 1)).x.reshape(-1, 1)
    finally:
        return A

TOTAL = 100
STEP = 0.5


def func(x):
    return 0.8 * x + 18


def generate_sample(total=TOTAL):
    x = 0
    while x < total * STEP:
        yield func(x) + np.random.normal(0, 7)
        x += STEP


X = np.arange(0, TOTAL * STEP, STEP).reshape(TOTAL, 1)
Y = np.array([y for y in generate_sample(TOTAL)]).reshape(TOTAL, 1)
Y_real = np.array([func(x) for x in X])

plt.plot(X, Y, 'bo')
plt.plot(X, Y_real, 'g', linewidth=2.0)
plt.show()

b0, b1 = LinearRegression(X, Y)
Y_pred = b0 + b1 * X

plt.plot(X, Y, 'bo')
plt.plot(X, Y_real, 'g', linewidth=2.0)
plt.plot(X, Y_pred, 'r', linewidth=2.0)
plt.show()