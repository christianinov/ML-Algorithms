import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
import pandas as pd
from math import *


def LogisticRegression(X, Y):
    n, m = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1)

    def cost(theta):
        cost = 0
        for i in range(n):
            t = X[i].dot(theta)
            cost += y[i] * t - log(1 + exp(t))
        return -cost / n

    return opt.minimize(cost, np.zeros(m + 1)).x

df = pd.DataFrame.from_csv("http://roman-kh.github.io/files/linear-models/simple1.csv")
x = df.iloc[:,[1,3]].as_matrix() # просто выбираю пару столбцов, которые подходят для лог регрессии
x[:145,1] = x[:145,1] - 8
y = df.iloc[:,2].as_matrix()


logit = LogisticRegression(x, y)

x1 = x[:,0]
x2 = - (logit[0] + logit[1] * x1)/logit[2]
plt.scatter(x[:,0], x[:,1], c=y)
plt.plot(x1, x2, 'g', linewidth=2.0)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()



data = pd.read_excel('wine2.xls')
data = data[ : 130]
x = data.iloc[:,[1,6]].as_matrix()
y = data.iloc[:,0].as_matrix()
for i in range(len(y)):
    if y[i] == 2:
        y[i] = 0

logit = LogisticRegression(x, y)

x1 = x[:,0]
x2 = - (logit[0] + logit[1] * x1)/logit[2]
plt.scatter(x[:,0], x[:,1], c=y)
plt.plot(x1, x2, 'g', linewidth=2.0)
plt.xlabel('x1')
plt.ylabel('x2')
plt.ylim((1, 4))
plt.show()