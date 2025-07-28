import numpy as np
from pylbfgs import owlqn
from example import evaluate, progress
from pylbfgs import owlqn
import scipy.fft as spfft

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

    # Current logic: We are trying to minimise complexity of freq basis??
    # Initialise all mag of each frequency to 1
    # x2 is the representation of the image in the frequency domain???
    # Is that actually true??????
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


def evaluate_kron(x, g, step):
    """An in-memory evaluation callback using the kronecker product.
    """

    # calculate the 2-norm squared of the residual vector
    p = np.dot(_A_matrix, x)
    fx = np.sum(np.power(_b_vector - p, 2))

    # calculate the gradient vector
    atax = np.dot(_A_matrix.T, p)
    atb = np.dot(_A_matrix.T, _b_vector)
    np.copyto(g, 2 * (atax - atb))

    return fx


def kron_rows(A, B, I, f=None):
    """Return individual rows of K=kron(A,B) if `f` is None. Otherwise
    save the matrix to file.
    """

    # find row indices of A and B
    ma, na = A.shape
    mb, nb = B.shape
    R = np.floor(I / mb).astype('int')  # A row indices of interest
    S = np.mod(I, mb)  # B row indices of interest

    # calculate kronecker product rows
    n = na * nb
    if f is None:
        K = np.zeros((I.size, n))

    for j, (r, s) in enumerate(zip(R, S)):
        row = np.multiply(
            np.kron(A[r, :], np.ones((1, nb))),
            np.kron(np.ones((1, na)), B[s, :])
            )
        if f is None:
            K[j, :] = row
        else:
            row.tofile(f)

    if f is None:
        return K

def progress(x, g, fx, xnorm, gnorm, step, k, ls):
    """Just display the current iteration.
    """
    #print('Iteration {}'.format(k))
    return 0

global _b_vector, _A_matrix, _image_dims, _ri_vector

#X = np.array([[1,2,3,4],[2,4,6,8],[3,6,9,12]])
#ri = np.array([[0,1,1,1],[1,0,0,1],[1,0,1,1]]).astype(bool).flatten()

X = np.array([[1,1.1,1,1.1]]) * 100
N = X.shape[1]  # number of columns
arr = []
for k in range(N):
    store = []
    for i in range(N):
        store.append(k * (2*i+1))
    arr.append(store)
    
arr = np.array(arr)*np.pi/(2*N)
arr = 2 * np.cos(arr) / (2*N)**0.5  # custom DCT operation
arr[0] /= 2**0.5  # adjust for the first element
custom_dct = (arr @ X.T).T
print("custom_dct:", custom_dct)

arr = []
for k in range(N):
    store = []
    for i in range(N):
        store.append(i * (2*k+1))
    arr.append(store)
arr = np.array(arr)*np.pi/(2*N)
arr = 2 * np.cos(arr) / (2*N)**0.5
arr[:,0] /= 2**0.5  # adjust for the first element
custom_idct = (arr @ X.T).T
print("custom_idct:", custom_idct)

ri = np.array([[1,1,1,0]]).astype(bool).flatten()
ri = np.where(ri)[0]  # Get the indices of the valid values
b = X.T.flatten()[ri].astype(float)  # Valid values turned into column vector
ny, nx = X.shape

ORTHANTWISE_C = 5

if True:
    # save image dims, sampling vector, and b vector and to global vars
    _image_dims = (ny, nx)
    _ri_vector = ri
    _b_vector = np.expand_dims(b, axis=1)

    # perform the L1 minimization in memory
    Xat2 = owlqn(nx*ny, evaluate, progress, ORTHANTWISE_C)
else:
    # save refs to global vars
    _b_vector = np.expand_dims(b, axis=1)
    _A_matrix = kron_rows(
        spfft.idct(np.identity(nx), norm='ortho', axis=0),
        spfft.idct(np.identity(ny), norm='ortho', axis=0),
        ri
        )
    # perform the L1 minimization in memory
    Xat2 = owlqn(nx*ny, evaluate_kron, progress, ORTHANTWISE_C)

Xat = Xat2.reshape(nx, ny).T  # stack columns
print("Xat:", Xat)
Xa = idct2(Xat)
print("Xa:", Xa)