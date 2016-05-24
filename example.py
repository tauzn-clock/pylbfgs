#!/usr/bin/env python

import os
import numpy as np
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import matplotlib.pyplot as plt
import pylbfgs as l1


EVAL_METHOD = 1 # 1: in-memory version, 2: file-based version
SCALE = 0.1 # fraction to scale the original image
SAMPLE = 0.4 # fraction of the scaled image to randomly sample
ORTHANTWISE_C = 2 # coeefficient for the L1 norm of variables
ORIG_IMAGE_PATH = 'test/testimage.png'
A_FILE_PATH = 'test/a_matrix.dat'
B_FILE_PATH = 'test/b_vector.dat'


def dct2(x):
    """Return 2D discrete cosine transform."""
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def idct2(x):
    """Return inverse 2D discrete cosine transform."""
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def kron_rows(A, B, I, f=None):
    """Return individual rows of K=kron(A,B) if `f` is None. Otherwise \
    save the matrix to file."""

    # find row indices of A and B
    ma, na = A.shape
    mb, nb = B.shape
    R = np.floor(I / mb).astype('int') # A row indices of interest
    S = np.mod(I, mb) # B row indices of interest

    # calculate kronecker product rows
    n = na * nb
    if f is None:
        K = np.zeros((I.size, n))
        
    for j,(r,s) in enumerate(zip(R, S)):
        row = np.multiply(
            np.kron(A[r,:], np.ones((1, nb))), 
            np.kron(np.ones((1, na)), B[s,:])
            )
        if f is None:
            K[j,:] = row
        else:
            # array('d', row.squeeze()).tofile(f)
            row.tofile(f)

    if f is None:
        return K


def progress(x, g, fx, xnorm, gnorm, step, k, ls):
    """Just display the current iteration."""

    print('Iteration {}'.format(k))
    return 0


_A_matrix = None # reference the dct matrix operator A here
_b_vector = None # reference the sample vector b here
def evaluate(x, g, step):
    """An in-memory evaluation callback."""

    # calculate the 2-norm squared of the residual vector
    p = np.dot(_A_matrix, x)
    fx = np.sum(np.power(_b_vector - p, 2))

    # calculate the gradient vector
    atax = np.dot(_A_matrix.T, p)
    atb = np.dot(_A_matrix.T, _b_vector)
    np.copyto(g, 2 * (atax - atb))

    return fx


def evaluate_from_file(x, g, step):
    """A slower, but more memory efficient, evaluation callback."""

    # read in b data from file
    with open(B_FILE_PATH, 'rb') as f:
        b = np.fromfile(f)

    # allocate vectors
    k = b.size
    n = x.size
    Ax = np.zeros(b.shape)
    Atax = np.zeros(x.shape)
    Atb = np.zeros(x.shape)
    
    # read in A data from file
    with open(A_FILE_PATH, 'rb') as f:
        for i in range(k):
            row = np.fromfile(f, count=n)
            Ax[i] = np.dot(row, x)
            for j in range(n):
                Atax[j] += row[j] * Ax[i]
                Atb[j] += row[j] * b[i]

    # calculate the 2-norm squared of the residual vector
    fx = np.sum(np.power(b - Ax, 2))

    # calculate the gradient vector
    np.copyto(g, 2 * (Atax - Atb))

    return fx


def main():
    
    # read image in grayscale, then downscale it
    Xorig = spimg.imread(ORIG_IMAGE_PATH, flatten=True, mode='L') # read in grayscale
    X = spimg.zoom(Xorig, SCALE)
    ny,nx = X.shape

    # take random samples of image, store them in a vector b
    k = round(nx * ny * SAMPLE)
    ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices
    b = X.T.flat[ri].astype(float) # important: cast to 64 bit

    if EVAL_METHOD == 1:

        # save refs to global vars
        global _b_vector, _A_matrix
        _b_vector = np.expand_dims(b, axis=1)
        _A_matrix = kron_rows(
            spfft.idct(np.identity(nx), norm='ortho', axis=0), 
            spfft.idct(np.identity(ny), norm='ortho', axis=0),
            ri
            )

        # perform the L1 minimization in memory
        Xat2 = l1.owlqn(nx*ny, evaluate, progress, ORTHANTWISE_C)


    elif EVAL_METHOD == 2:

        # save the b samples to file
        with open(B_FILE_PATH, 'wb') as f:
            print('Writing {} values to {}'.format(k, B_FILE_PATH))
            b.tofile(f)

        # create dct matrix operator A, saving it directly to file
        try:
            os.remove(A_FILE_PATH)
        except: 
            pass
        with open(A_FILE_PATH, 'w+b') as f:
            print('Writing {} values to {}'.format(nx * ny * k, A_FILE_PATH))
            kron_rows(
                spfft.idct(np.identity(nx), norm='ortho', axis=0), 
                spfft.idct(np.identity(ny), norm='ortho', axis=0),
                ri, f
                )

        # perform the L1 minimization from file
        Xat2 = l1.owlqn(nx*ny, evaluate_from_file, progress, ORTHANTWISE_C)

    # transform the output back into the spatial domain
    Xat = Xat2.reshape(nx, ny).T # stack columns
    Xa = idct2(Xat)

    # display the result
    f, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].imshow(X, cmap='gray', interpolation='none')
    ax[1].imshow(Xa, cmap='gray', interpolation='none')
    plt.show()


if __name__ == '__main__':
    main()
