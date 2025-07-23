#!/usr/bin/env python

import os
import numpy as np
from PIL import Image
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import matplotlib.pyplot as plt
from pylbfgs import owlqn

def dct2(x):
    """Return 2D discrete cosine transform.
    """
    return spfft.dct(
        spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def idct2(x):
    """Return inverse 2D discrete cosine transform.
    """
    return spfft.idct(
        spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def evaluate(x, g, step):
    """An in-memory evaluation callback.
    """

    # we want to return two things:
    # (1) the norm squared of the residuals, sum((Ax-b).^2), and
    # (2) the gradient 2*A'(Ax-b)

    # expand x columns-first
    x2 = x.reshape((_image_dims[1], _image_dims[0])).T

    # Ax is just the inverse 2D dct of x2
    Ax2 = idct2(x2)

    # stack columns and extract samples
    Ax = Ax2.T.flat[_ri_vector].reshape(_b_vector.shape)

    # calculate the residual Ax-b and its 2-norm squared
    Axb = Ax - _b_vector
    fx = np.sum(np.power(Axb, 2))

    # project residual vector (k x 1) onto blank image (ny x nx)
    Axb2 = np.zeros(x2.shape)
    Axb2.T.flat[_ri_vector] = Axb  # fill columns-first

    # A'(Ax-b) is just the 2D dct of Axb2
    AtAxb2 = 2 * dct2(Axb2)
    AtAxb = AtAxb2.T.reshape(x.shape)  # stack columns

    # copy over the gradient vector
    np.copyto(g, AtAxb)

    return fx

def progress(x, g, fx, xnorm, gnorm, step, k, ls):
    """Just display the current iteration.
    """
    #print('Iteration {}'.format(k))
    return 0

def lbfgs_reconstruct_image(Xorig):
    global _image_dims, _ri_vector, _b_vector

    # Evaluation method (always use #1)
    # 1: in-memory dct2 version (fast, efficient for all size images)
    # 2: in-memory kron version (fast, but only for images smaller than 100x100)
    # 3: file-based kron version (experimental, slow, lots of disk
    #    space required [Order(rows x cols x samples)])
    EVAL_METHOD = 1

    # Coeefficient for the L1 norm of variables (see OWL-QN algorithm)
    ORTHANTWISE_C = 5

    b = Xorig.T.flatten()  # flatten image to vector
    mask = b > 0#np.random.choice(Xorig.shape[0]*Xorig.shape[1], 100000, replace=False)  # random sample of indices
    b = b[mask]  # remove zero values
    print('Flattened image shape: {}'.format(b.shape))

    # save image dims, sampling vector, and b vector and to global vars
    _image_dims = (Xorig.shape[0], Xorig.shape[1])
    _ri_vector = mask
    _b_vector = np.expand_dims(b, axis=1)

    # perform the L1 minimization in memory
    Xat2 = owlqn(_image_dims[0] * _image_dims[1], evaluate, progress, ORTHANTWISE_C)

    Xat = Xat2.reshape(_image_dims[1], _image_dims[0]).T  # stack columns
    Xa = idct2(Xat)

    return Xa

if __name__ == "__main__":
    # File paths
    ORIG_IMAGE_PATH = "/scratchdata/nyu_depth_crop/train/bedroom_0004/sync_depth_00000.png"

    # read image in grayscale, then downscale it
    Xorig = Image.open(ORIG_IMAGE_PATH)
    Xorig = np.array(Xorig, dtype=float)  # convert to float
    print('Original image shape: {}'.format(Xorig.shape))
    print('Original image max, min: {}, {}'.format(Xorig.max(), Xorig.min()))

    Xa = lbfgs_reconstruct_image(Xorig)
    print(Xa.max(), Xa.min())
    print('Reconstructed image shape: {}'.format(Xa.shape))

    Image.fromarray(Xa.astype(np.uint16)).save('reconstructed_image.png')

    plt.imsave('vis.png', Xa, cmap='gray')

    from metric import evaluateMetrics
    evaluateMetrics(Xorig, Xa)