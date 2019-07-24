"""
CS131 - Computer Vision: Foundations and Applications
Assignment 2
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/18/2017
Python Version: 3.5+
"""

import numpy as np


def conv(image, kernel):
    """
    An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Parameters
    ----------
    image : numpy array of shape (Hi, Wi).

    kernel : numpy array of shape (Hk, Wk).

    Returns
    -------
    out : numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0, pad_width0), (pad_width1, pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    # YOUR CODE HERE
    kernel = np.flip(np.flip(kernel, 0), 1)
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(padded[i: i + Hk, j: j + Wk] * kernel)
    # END YOUR CODE

    return out


def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Parameters
    ----------
    size : int of the size of output matrix.

    sigma : float of sigma to calculate kernel.

    Returns
    -------
    kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    # YOUR CODE HERE
    for i in range(size):
        for j in range(size):
            normal_factor = 1 / (2 * np.pi * sigma ** 2)
            kernel[i, j] = normal_factor * np.exp(
                - ((i - size // 2) ** 2 + (j - size // 2) ** 2) / (
                        2.0 * sigma ** 2))
    # END YOUR CODE

    return kernel


def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """
    # YOUR CODE HERE
    kernel = 0.5 * np.array([[1, 0, -1]])
    out = conv(img, kernel)
    # END YOUR CODE

    return out


def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """
    # YOUR CODE HERE
    kernel = 0.5 * np.array([[1, 0, -1]]).reshape(-1, 1)
    out = conv(img, kernel)
    # END YOUR CODE

    return out


def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    # YOUR CODE HERE
    gx, gy = partial_x(img), partial_y(img)
    G = np.sqrt(gx ** 2 + gy ** 2)
    # np.rad2deg output range [-pi, +pi]
    theta = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 360
    # END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    # this will round the value in range [337.5, 360] to 360
    theta = (np.floor((theta + 22.5) / 45) * 45) % 360

    angle_to_direc = {0: ([0, -1], [0, 1]),
                      180: ([0, -1], [0, 1]),
                      45: ([-1, -1], [1, 1]),
                      225: ([-1, -1], [1, 1]),
                      90: ([-1, 0], [1, 0]),
                      270: ([-1, 0], [1, 0]),
                      135: ([-1, 1], [1, -1]),
                      315: ([-1, 1], [1, -1]),
                      }

    # BEGIN YOUR CODE
    # the gradient is measured clock-wisely
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            # get the index offsets of the two neighbors
            directions = angle_to_direc[theta[i, j]]
            neighbors = [G[i + delta[0], j + delta[1]] for delta in directions]
            if not (G[i, j] >= np.max(neighbors)):
                out[i, j] = 0
            else:
                out[i, j] = G[i, j]
    # END YOUR CODE

    return out


def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    # YOUR CODE HERE
    strong_edges = img > high
    weak_edges = np.logical_and(img <= high, img > low)
    # END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y - 1, y, y + 1):
        for j in (x - 1, x, x + 1):
            if 0 <= i < H and 0 <= j < W:
                if i == y and j == x:
                    continue
                neighbors.append((i, j))

    return neighbors


def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).

    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    edges = np.copy(strong_edges)

    # YOUR CODE HERE
    nodes_to_visit = []
    is_visited = np.zeros_like(edges)
    nodes_to_visit.append([0, 0])

    while len(nodes_to_visit) != 0:
        curr_i, curr_j = nodes_to_visit.pop(0)

        if is_visited[curr_i, curr_j] == 1:
            continue

        is_visited[curr_i, curr_j] = 1

        neighbors = get_neighbors(curr_i, curr_j, H, W)

        for x, y in neighbors:
            nodes_to_visit.append([x, y])
        # check if any neighbor is a strong edge
        is_connected = np.any([edges[x, y] for x, y in neighbors])
        # if current pixel is weak edge and is connected to a strong edge
        if weak_edges[curr_i, curr_j] and is_connected:
            edges[curr_i, curr_j] = True
    # END YOUR CODE

    return edges


def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    # YOUR CODE HERE
    # suppress noise
    img_smoothed = conv(img, gaussian_kernel(kernel_size, sigma))
    # compute gradient magnitude and direction
    G, theta = gradient(img_smoothed)
    # apply nms
    nms = non_maximum_suppression(G, theta)
    # apply double thresholding
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    # apply hysteresis and connectivity analysis
    edge = link_edges(strong_edges, weak_edges)
    # END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).

    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, int(diag_len * 2 + 1))
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    # YOUR CODE HERE
    for y, x in zip(ys, xs):
        for i in range(num_thetas):
            rho = x * cos_t[i] + y * sin_t[i]
            # rho plus diag_len, giving a value in range [0, 2 * diag_len]
            # this transforms the rho value into the row index of the accumulator
            accumulator[int(rho + diag_len), i] += 1
    # END YOUR CODE

    return accumulator, rhos, thetas
