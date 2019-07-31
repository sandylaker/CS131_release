"""
CS131 - Computer Vision: Foundations and Applications
Assignment 5
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 09/25/2018
Python Version: 3.5+
"""

import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float


# Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        # YOUR CODE HERE
        # compute the distances between points and cluster centers
        # Can use the x1.T * x2 / (||x1|| * ||x2||) as a measurement for the closeness here
        # euclid_dist = np.sum(features.reshape(N, 1, -1) * centers.reshape(1, D, -1), axis=-1)
        # norms = np.linalg.norm(features, axis=-1, keepdims=True) * np.linalg.norm(
        #     centers, axis=-1, keepdims=True).T
        # dist = euclid_dist / norms
        # # assignments_new = np.argmax(dist, axis=-1)
        # assign points to clusters

        dist = np.sum((features.reshape(N, 1, D) - centers.reshape(1, -1, D)) ** 2, axis=-1)
        assignments_new = np.argmin(dist, axis=1)
        # update the centers
        for j in range(k):
            centers[j] = np.mean(features[assignments_new == j], axis=0)

        # stop if converged
        if np.all(assignments == assignments_new):
            break
        assignments = assignments_new.copy()
        # END YOUR CODE

    return assignments


def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        # YOUR CODE HERE
        dist = np.sum((features.reshape(N, 1, D) - centers.reshape(1, -1, D)) ** 2, axis=-1)
        assignments_new = np.argmin(dist, axis=1)
        # update the centers
        for j in range(k):
            centers[j] = np.mean(features[assignments_new == j], axis=0)

        # stop if converged
        if np.all(assignments == assignments_new):
            break
        assignments = assignments_new.copy()
        # END YOUR CODE

    return assignments


def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N

    while n_clusters > k:
        # YOUR CODE HERE
        dist = squareform(pdist(centers, 'euclidean'))
        # we do not consider the distance of a point and itself
        np.fill_diagonal(dist, np.inf)
        # find the pair of clusters that have least distance (over all pairs)
        pair = (np.argmin(dist) // dist.shape[0], np.argmin(dist) % dist.shape[1])
        n_clusters -= 1

        # merge the clusters and update the centers
        centers[pair[0]] = (centers[pair[0]] + centers[pair[1]]) / 2
        # delete the merged center
        centers = np.delete(centers, pair[1], 0)

        # update assignments, we only need to consider the points that are assigned to pair[1],
        # and those points that are assigned to the clusters that have larger indices than pair[1],
        # since these clusters indices will be shifted as we merge and the pair[1] to pair[0]
        for i in range(N):
            if assignments[i] >= pair[1]:
                assignments[i] = pair[0] if assignments[i] == pair[1] else assignments[i] - 1
        # END YOUR CODE

    return assignments


# Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    # YOUR CODE HERE
    features = img.reshape(features.shape)
    # END YOUR CODE

    return features


def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    # YOUR CODE HERE
    color_feat = color_features(img)
    pos_feat = np.transpose(np.mgrid[0:H, 0:W], (1, 2, 0)).reshape(H*W, -1)
    features = np.concatenate((color_feat, pos_feat), axis=1)
    # normalize features
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0, ddof=1)
    # END YOUR CODE

    return features


def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    # YOUR CODE HERE
    pass
    # END YOUR CODE
    return features


# Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    # YOUR CODE HERE
    accuracy = np.mean(mask_gt == mask)
    # END YOUR CODE

    return accuracy


def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
