import matplotlib.pyplot as plt
import numpy as np

# Generate Data
points = np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([1, 0])),
              (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
              (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))

def normalize(data, weights=None):
    '''
        Function normalizes the distance
    '''
    if weights == None:
        weights = np.ones(data.shape[1])
    column_min = data.min(axis=0)
    column_dif = data.max(axis=0) - column_min
    return (((data - column_min)/column_dif) * weights, column_dif, column_min)

def initialize_centroids(points, k):
    '''
        Selects k random points as initial
        points from dataset
    '''
    ind = np.random.randint(0, points.shape[0], 3)
    return points[ind]

def closest_centroid(points, centroids):
    '''
        Returns an array containing the index to the nearest centroid for each point
    '''
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def move_centroids(points, closest, k):
    '''
        Returns the new centroids assigned from the points closest to them
    '''
    return np.array([points[closest==i].mean(axis=0) for i in range(k)])


def main(points, num_iterations=100, k=3, weights=None):
    # Normalize points
    points, column_dif, column_min = normalize(points, weights)

    # Initialize centroids
    centroids = initialize_centroids(points, k)

    # Run iterative process
    for i in range(num_iterations):
        closest = closest_centroid(points, centroids)
        centroids = move_centroids(points, closest, k)

    fig = plt.figure()
    plt.scatter(points[:, 0] * column_dif[0] + column_min[0], points[:, 1] * column_dif[1] + column_min[1])
    plt.scatter(centroids[:, 0] * column_dif[0] + column_min[0], centroids[:, 1] * column_dif[1] + column_min[1], c='r',
                s=100)
    plt.show()

    return centroids

centroids = main(points)