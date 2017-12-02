import numpy as np
import pandas as pd
import time
from scipy import stats
from matplotlib import pyplot as plt

def normalize(data, weights=None):
    '''
        Function normalizes the distance
    '''
    if weights == None:
        weights = np.ones(data.shape[1])
    column_min = data.min(axis=0)
    column_dif = data.max(axis=0) - column_min
    return ((data - column_min)/column_dif) * weights


def sort_distancies(data_x, unknown):
    '''
        Function sorts indexes of known points by the distance for each unknown points
    '''
    # find distancies
    dists = np.sqrt(((data_x - unknown[:, np.newaxis]) ** 2).sum(axis=2))

    # sort indexes
    return dists.argsort(axis=1)

def predict(sorted_inds, data_y, k):
    '''
        Function predicts the class of the unknown point by the k nearest neighbours
    '''
    closest_y = data_y[[sorted_inds[:,:k]]]
    return stats.mode(closest_y, axis=1).mode.reshape(-1)

def accuracy(predicted,real):
    '''
        Calculates accuracy percentage
    '''
    correct = sum(predicted == real)
    total = len(predicted)
    return 100*correct/total


def compare_k(data_x, data_y, test_x, test_y, kmin=1, kmax=50, kstep=5, weights=None):
    '''
        Main comparing function
    '''
    k = list(range(kmin, kmax, kstep))
    steps = len(k)
    features = np.zeros((steps, 3))

    data_x = normalize(data_x, weights)
    test_x = normalize(test_x, weights)

    print('Sorting indexes by distance started')

    t0 = time.time()
    sorted_indexes = sort_distancies(data_x, test_x)
    t = time.time()
    s1 = data_x.shape[0]
    s2 = test_x.shape[0]

    print('Indexes sorted in %f seconds for %dx%d' % (t - t0, s1, s2))

    miss = []

    for j in range(steps):
        t0 = time.time()
        yk = predict(sorted_indexes, data_y, k[j])
        t = time.time() - t0
        features[j][0] = k[j]
        features[j][1] = accuracy(yk, test_y)
        features[j][2] = t
        cond = yk != test_y
        miss.append({
            'k': k[j],
            'acc': features[j][1],
            'x': test_x[cond]}
        )

        print('k={0}, accuracy = {1}%, time = {2} sec'.format(k[j], features[j][1], features[j][2]))

    return features, miss

num_observations = 300
x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
x2 = np.random.multivariate_normal([-2, 3], [[2, .75], [.75, 2]], num_observations)


X = np.vstack((x1, x2)).astype(np.float32)
Y = np.hstack((np.zeros(num_observations),
               np.ones(num_observations)))

splitRatio = 0.67

# Training set size
trainSize = int(X.shape[0] * splitRatio)

# List of randomly chosen indicies
indices = np.random.permutation(X.shape[0])

# Split indicies for training and test set by trainSize
training_idx, test_idx = indices[:trainSize], indices[trainSize:]

# Create training and test sets by indicies
x_trn = X[training_idx, :]
y_trn = Y[training_idx]
x_tst = X[test_idx, :]
y_tst = Y[test_idx]

res, ms = compare_k(x_trn, y_trn, x_tst, y_tst,1,102,10)

# initial data
fig = plt.figure()
plt.scatter(x1[:, 0], x1[:, 1], color='c',label='class1')
plt.scatter(x2[:, 0], x2[:, 1], color='y',label='class2')
# randomly selected data
plt.scatter(x_tst[:,0],x_tst[:,1],color='b',label='test')
plt.legend(loc='best')


# missidentifies for k = value
plt.figure()
plt.scatter(x1[:, 0], x1[:, 1], color='c', label='class1')
plt.scatter(x2[:, 0], x2[:, 1], color='y', label='class2')
plt.scatter(ms[-1]['x'][:,0],ms[-1]['x'][:,1],color='r',label='missidenity,k=%d'%ms[-1]['k'])
plt.legend(loc='best')
plt.xlabel('x1')
plt.ylabel('x2')

# accuracy plot
plt.figure()
k = plt.scatter(res[:, 0], res[:, 1])
plt.ylim(min(res[:, 1]) - 2, max(res[:, 1])+1, 4)
plt.xlabel('k')
plt.ylabel('accuracy, %')
plt.show()