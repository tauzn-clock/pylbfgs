from pylbfgs import owlqn
import scipy.fft as spfft
import numpy as np

global _b_vector, _A_matrix, _image_dims, _ri_vector

def set_global_param(b_vector, image_dims, ri_vector):
    """Set the global parameters for the evaluation function.
    
    Args:
        b_vector (np.ndarray): The b vector used in the evaluation.
        image_dims (tuple): The dimensions of the image (ny, nx).
        ri_vector (np.ndarray): The sampling vector indicating valid indices.
    """
    global _b_vector, _image_dims, _ri_vector
    _b_vector = b_vector
    _image_dims = image_dims
    _ri_vector = ri_vector

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
    Ax2 = spfft.idctn(x2, norm='ortho')

    # stack columns and extract samples
    Ax = Ax2.T.flat[_ri_vector].reshape(_b_vector.shape)

    # calculate the residual Ax-b and its 2-norm squared
    Axb = Ax - _b_vector
    fx = np.sum(np.power(Axb, 2))

    # project residual vector (k x 1) onto blank image (ny x nx)
    Axb2 = np.zeros(x2.shape)
    Axb2.T.flat[_ri_vector] = Axb  # fill columns-first

    # A'(Ax-b) is just the 2D dct of Axb2
    AtAxb2 = 2 * spfft.dctn(Axb2, norm='ortho')
    AtAxb = AtAxb2.T.reshape(x.shape)  # stack columns

    # copy over the gradient vector
    np.copyto(g, AtAxb)

    return fx

def progress(x, g, fx, xnorm, gnorm, step, k, ls):
    """Just display the current iteration.
    """
    #print('Iteration {}'.format(k))
    return 0

def rescale_ratio(depth, est):
    """Rescale the depth map based on the estimated depth.

    Args:
        depth (np.ndarray): The original depth map.
        est (np.ndarray): The estimated depth map.

    Returns:
        np.ndarray: The rescaled depth map.
    """
    ratio = depth / est
    ri = ratio !=0
    ratio[~ri] = 1
    ratio -= 1
    print("Ratio max, min:", ratio.max(), ratio.min())
    ri = np.where(ri.flatten())[0]
    b = ratio.T.flatten()[ri].astype(float)
    ny, nx = ratio.shape

    set_global_param(b, (ny, nx), ri)
    
    out = owlqn(nx * ny, evaluate, progress, 5)

    return spfft.idctn(out.reshape((nx, ny)).T, norm='ortho') + 1