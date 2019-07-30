"""
CS131 - Computer Vision: Foundations and Applications
Assignment 4
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 10/19/2018
Python Version: 3.5+
"""

import numpy as np
from skimage import color
import numba

def energy_function(image):
    """Computes energy of the input image.

    For each pixel, we will sum the absolute value of the gradient in each direction.
    Don't forget to convert to grayscale first.

    Hint: Use np.gradient here

    Args:
        image: numpy array of shape (H, W, 3)

    Returns:
        out: numpy array of shape (H, W)
    """
    H, W, _ = image.shape
    out = np.zeros((H, W))
    gray_image = color.rgb2gray(image)

    # YOUR CODE HERE
    grad_x, grad_y = np.gradient(gray_image)
    out = np.abs(grad_x) + np.abs(grad_y)
    # END YOUR CODE

    return out


# @numba.njit
def compute_cost(image, energy, axis=1):
    """Computes optimal cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.

    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    In the case that energies are equal, choose the left-most path. Note that
    np.argmin returns the index of the first ocurring minimum of the specified
    axis.

    Make sure your code is vectorized because this function will be called a lot.
    You should only have one loop iterating through the rows.

    Args:
        image: not used for this function
               (this is to have a common interface with compute_forward_cost)
        energy: numpy array of shape (H, W)
        axis: compute cost in width (axis=1) or height (axis=0)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """
    energy = energy.copy()

    if axis == 0:
        energy = np.transpose(energy, (1, 0))

    H, W = energy.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=int)

    # Initialization
    cost[0] = energy[0]
    paths[0] = 0  # we don't care about the first row of paths

    # YOUR CODE HERE
    for i in range(1, H):
        tmp = np.zeros((3, W))
        # E(i, j) + M(i-1, j-1), sum the left most element in i-th row with 0
        # go left when back tracking from bottom to up
        tmp[0] = np.concatenate(([np.inf], energy[i, 1:] + cost[i - 1, :-1]))
        # E(i, J) + M(i-1, j)
        tmp[1] = energy[i, :] + cost[i-1, :]
        # E(i, j) + M(i-1, j+1), sum the right most element in i-th row with 0
        # go right when back tracking from bottom to up
        tmp[2] = np.concatenate((energy[i, :-1] + cost[i - 1, 1:], [np.inf]))
        cost[i] = np.min(tmp, axis=0)
        # minus 1 to form the mapping {0: -1, 1: 0, 2: 1}
        paths[i] = np.argmin(tmp, axis=0) - 1
    # END YOUR CODE

    if axis == 0:
        cost = np.transpose(cost, (1, 0))
        paths = np.transpose(paths, (1, 0))

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


# @numba.njit
def backtrack_seam(paths, end):
    """Backtracks the paths map to find the seam ending at (H-1, end)

    To do that, we start at the bottom of the image on position (H-1, end), and we
    go up row by row by following the direction indicated by paths:
        - left (value -1)
        - middle (value 0)
        - right (value 1)

    Args:
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
        end: the seam ends at pixel (H, end)

    Returns:
        seam: np.array of indices of shape (H,). The path pixels are the (i, seam[i])
    """
    H, W = paths.shape
    # initialize with -1 to make sure that everything gets modified
    seam = - np.ones(H, dtype=int)

    # Initialization
    seam[H-1] = end

    # YOUR CODE HERE
    for i in range(2, H+1):
        seam[H - i] = paths[H - i + 1, seam[H - i + 1]] + seam[H - i + 1]
    # END YOUR CODE

    # Check that seam only contains values in [0, W-1]
    assert np.all(np.all([seam >= 0, seam < W], axis=0)), "seam contains values out of bounds"

    return seam


# @numba.njit
def remove_seam(image, seam):
    """Remove a seam from the image.

    This function will be helpful for functions reduce and reduce_forward.

    Args:
        image: numpy array of shape (H, W, C) or shape (H, W)
        seam: numpy array of shape (H,) containing indices of the seam to remove

    Returns:
        out: numpy array of shape (H, W-1, C) or shape (H, W-1)
             make sure that `out` has same type as `image`
    """

    # Add extra dimension if 2D input
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    out = None
    H, W, C = image.shape
    # YOUR CODE HERE
    mask = np.ones((H, W), dtype=bool)
    mask[np.arange(H), seam] = 0
    out = image[mask].reshape(H, W-1, C).astype(image.dtype)
    # END YOUR CODE
    out = np.squeeze(out)  # remove last dimension if C == 1

    # Make sure that `out` has same type as `image`
    assert out.dtype == image.dtype, \
       "Type changed between image (%s) and out (%s) in remove_seam" % (image.dtype, out.dtype)

    return out


# @numba.njit
def reduce(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process.

    At each step, we remove the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.
    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, 3)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, 3) if axis=0, or (H, size, 3) if axis=1
    """

    out = np.copy(image)

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    # YOUR CODE HERE
    energy = efunc(out)
    cost, paths = cfunc(out, energy)
    for i in range(W - size):
        # find the seam
        seam = backtrack_seam(paths, end=cost[-1].argmin())
        # remove the seam
        out = remove_seam(out, seam)
        # recompute the energy, cost and paths
        energy = efunc(out)
        cost, paths = cfunc(out, energy)
    # END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


# @numba.njit
def duplicate_seam(image, seam):
    """Duplicates pixels of the seam, making the pixels on the seam path "twice larger".

    This function will be helpful in functions enlarge_naive and enlarge.

    Args:
        image: numpy array of shape (H, W, C)
        seam: numpy array of shape (H,) of indices

    Returns:
        out: numpy array of shape (H, W+1, C)
    """

    H, W, C = image.shape
    out = np.zeros((H, W + 1, C))
    # YOUR CODE HERE
    for i, j in enumerate(seam):
        out[i, :j, :] = image[i, :j, :]
        out[i, j+1:, :] = image[i, j:, :]
    out[np.arange(H), seam, :] = image[np.arange(H), seam, :]
    # END YOUR CODE

    return out


# @numba.njit
def enlarge_naive(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Increases the size of the image using the seam duplication process.

    At each step, we duplicate the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.
    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - duplicate_seam

    Args:
        image: numpy array of shape (H, W, C)
        size: size to increase height or width to (depending on axis)
        axis: increase in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert size > W, "size must be greather than %d" % W

    # YOUR CODE HERE
    energy = efunc(out)
    cost, paths = cfunc(image, energy)

    for _ in range(size - W):
        # find the seam
        seam = backtrack_seam(paths, cost[-1].argmin())
        # duplicate the seam
        out = duplicate_seam(out, seam)
        # re-compute the energy, cost and paths
        energy = efunc(out)
        cost, paths = cfunc(out, energy)
    # END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


# @numba.njit
def find_seams(image, k, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Find the top k seams (with lowest energy) in the image.

    We act like if we remove k seams from the image iteratively, but we need to store their
    position to be able to duplicate them in function enlarge.

    We keep track of where the seams are in the original image with the array seams, which
    is the output of find_seams.
    We also keep an indices array to map current pixels to their original position in the image.

    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, C)
        k: number of seams to find
        axis: find seams in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        seams: numpy array of shape (H, W)
    """

    image = np.copy(image)
    if axis == 0:
        image = np.transpose(image, (1, 0, 2))

    H, W, C = image.shape
    assert W > k, "k must be smaller than %d" % W

    # Create a map to remember original pixel indices
    # At each step, indices[row, col] will be the original column of current pixel
    # The position in the original image of this pixel is: (row, indices[row, col])
    # We initialize `indices` with an array like (for shape (2, 4)):
    #     [[1, 2, 3, 4],
    #      [1, 2, 3, 4]]
    indices = np.tile(range(W), (H, 1))  # shape (H, W)

    # We keep track here of the seams removed in our process
    # At the end of the process, seam number i will be stored as the path of value i+1 in `seams`
    # An example output for `seams` for two seams in a (3, 4) image can be:
    #    [[0, 1, 0, 2],
    #     [1, 0, 2, 0],
    #     [1, 0, 0, 2]]
    seams = np.zeros((H, W), dtype=np.int)

    # Iteratively find k seams for removal
    for i in range(k):
        # Get the current optimal seam
        energy = efunc(image)
        cost, paths = cfunc(image, energy)
        end = np.argmin(cost[H - 1])
        seam = backtrack_seam(paths, end)

        # Remove that seam from the image
        image = remove_seam(image, seam)

        # Store the new seam with value i+1 in the image
        # We can assert here that we are only writing on zeros (not overwriting existing seams)
        assert np.all(seams[np.arange(H), indices[np.arange(H), seam]] == 0), \
            "we are overwriting seams"
        seams[np.arange(H), indices[np.arange(H), seam]] = i + 1

        # We remove the indices used by the seam, so that `indices` keep the same shape as `image`
        indices = remove_seam(indices, seam)

    if axis == 0:
        seams = np.transpose(seams, (1, 0))

    return seams


# @numba.njit
def enlarge(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Enlarges the size of the image by duplicating the low energy seams.

    We start by getting the k seams to duplicate through function find_seams.
    We iterate through these seams and duplicate each one iteratively.

    Use functions:
        - find_seams
        - duplicate_seam

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: enlarge in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    # Transpose for height resizing
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H, W, C = out.shape

    assert size > W, "size must be greather than %d" % W

    assert size <= 2 * W, "size must be smaller than %d" % (2 * W)

    # YOUR CODE HERE
    # get k seams
    seams = find_seams(out, size - W)
    # record the seam indices of each seam, seam_index has shape (i, H) in the loop
    seam_index = []
    # duplicate each seam iteratively
    for i in range(1, size - W + 1):
        _, seam_cur = np.nonzero(seams == i)
        seam_index.append(seam_cur.copy())
        # get the order of the seam index, which is the shift for computing the index
        # of the current seam, since if the column index (in certain row) of the current seam in
        # the original image is located on the right of any previous seam, it will be shifted
        # onto the right and that shift will accumulate, depending on the how many seams
        # are located on the left side of current seam in that row.
        index_shift = np.argsort(np.argsort(np.asarray(seam_index), axis=0), axis=0)[-1, :]
        out = duplicate_seam(out, seam_cur + index_shift)

    # END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


# @numba.njit
def compute_forward_cost(image, energy):
    """Computes forward cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.
    Make sure to add the forward cost introduced when we remove the pixel of the seam.

    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    Args:
        image: numpy array of shape (H, W, 3) or (H, W)
        energy: numpy array of shape (H, W)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """

    image = color.rgb2gray(image)
    H, W = image.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    for j in range(W):
        if 0 < j < W - 1:
            cost[0, j] += np.abs(image[0, j+1] - image[0, j-1])
    paths[0] = 0  # we don't care about the first row of paths

    # YOUR CODE HERE
    for i in range(1, H):
        m1 = np.insert(image[i, :W-1], 0, 0, axis=0)
        m2 = np.insert(image[i, 1:], W-1, 0, axis=0)
        m3 = image[i-1]
        m4 = np.insert(image[i - 1, :W-1], 0, 0, axis=0)
        m5 = np.insert(image[i - 1, 1:], W-1, 0, axis=0)
        c_v = np.abs(m2 - m1) + np.abs(m5 - m4)
        c_v[0] = 0
        c_v[-1] = 0
        c_l = np.abs(m2 - m1) + np.abs(m3 - m1)
        c_r = np.abs(m2 - m1) + np.abs(m3 - m2)
        c_l[0] = 0
        c_r[-1] = 0
        i1 = np.insert(cost[i-1, 0: W-1], 0, np.inf, axis=0)
        i2 = cost[i-1]
        i3 = np.insert(cost[i-1, 1: W], W-1, np.inf, axis=0)
        C = np.concatenate(((i1 + c_l)[np.newaxis, :],
                            (i2 + c_v)[np.newaxis, :],
                            (i3 + c_r)[np.newaxis, :]), axis=0)
        # removing image[i, j] will loose the energy on pixel [i, j], so this part of energy
        # must also be counted in.
        cost[i] = energy[i] + np.min(C, axis=0)
        paths[i] = np.argmin(C, axis=0) - 1
    # END YOUR CODE

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


# @numba.njit
def reduce_fast(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process. Faster than `reduce`.

    Use your own implementation (you can use auxiliary functions if it helps like `energy_fast`)
    to implement a faster version of `reduce`.

    Hint: do we really need to compute the whole cost map again at each iteration?
    
    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    # YOUR CODE HERE
    energy = efunc(out)
    for _ in range(W - size):
        cost, paths = cfunc(out, energy)
        end = np.argmin(cost[-1])
        seam = backtrack_seam(paths, end)
        # get the seam area
        i = np.min(seam)
        j = np.max(seam)
        out = remove_seam(out, seam)
        # only re-compute the energy inside the seam area
        if i < 1:
            if j <= W-3:
                energy = np.c_[efunc(out[:, :j+1]), energy[:, j+2:]]
            else:
                energy = efunc(out)
        else:
            if j <= W-3:
                energy = np.c_[energy[:, i], efunc(out[:, i: j+1]), energy[:, j+2:]]
            else:
                energy = np.c_[energy[:, i], efunc(out[:, i:])]
    # END YOUR CODE
    print(out.shape)
    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


# @numba.njit
def remove_object(image, mask):
    """Remove the object present in the mask.

    Returns an output image with same shape as the input image, but without the object in the mask.

    Args:
        image: numpy array of shape (H, W, 3)
        mask: numpy boolean array of shape (H, W)

    Returns:
        out: numpy array of shape (H, W, 3)
    """
    assert image.shape[:2] == mask.shape

    H, W, _ = image.shape
    out = np.copy(image)

    # YOUR CODE HERE
    pass
    # END YOUR CODE

    assert out.shape == image.shape

    return out
