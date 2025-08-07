from pylbfgs import owlqn
import scipy.fft as spfft
import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt

global _b_vector, _A_matrix, _image_dims, _ri_vector

def dwt2_haar_recursive(arr):
    coeffs = np.zeros_like(arr)
    height, width = arr.shape
    if height == 1 or width == 1:
        return arr
    
    A, (H, V, D) = pywt.dwt2(arr, 'haar')
    coeffs[:(height+1)//2, (width+1)//2:] = V[:(height+1)//2, :width - (width+1)//2]
    coeffs[(height+1)//2:, :(width+1)//2] = H[:height-(height+1)//2, :(width+1)//2]
    coeffs[(height+1)//2:, (width+1)//2:] = D[:height-(height+1)//2, :width - (width+1)//2]
    A = dwt2_haar_recursive(A)
    coeffs[:(height+1)//2, :(width+1)//2] = A[:(height+1)//2, :(width+1)//2]
     
    return coeffs

def idwt2_haar_recursive(coeffs):
    
    height, width = coeffs.shape
    if height == 1 or width == 1:
        return coeffs
    A = coeffs[:(height+1)//2, :(width+1)//2]
    A = idwt2_haar_recursive(A)
    V = coeffs[:(height+1)//2, (width+1)//2:]
    H = coeffs[(height+1)//2:, :(width+1)//2]
    D = coeffs[(height+1)//2:, (width+1)//2:]
        
    # Pad H, V, D to match A
    H = np.pad(H, ((0, A.shape[0] - H.shape[0]), (0, A.shape[1] - H.shape[1])), mode='constant')
    V = np.pad(V, ((0, A.shape[0] - V.shape[0]), (0, A.shape[1] - V.shape[1])), mode='constant')
    D = np.pad(D, ((0, A.shape[0] - D.shape[0]), (0, A.shape[1] - D.shape[1])), mode='constant')

    arr = pywt.idwt2((A, (H,V,D)), 'haar')
    arr = arr[:height, :width]  # Ensure the output matches the original shape
    return arr

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

def rescale_ratio(depth, est, ORTHANTWISE_C=5, relative_C=None):
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
    #print("Ratio max, min:", ratio.max(), ratio.min())
    ri = np.where(ri.T.flatten())[0]
    b = ratio.T.flatten()[ri].astype(float)
    ny, nx = ratio.shape

    set_global_param(b, (ny, nx), ri)
    
    if not relative_C is None:
        ORTHANTWISE_C = np.mean(np.abs(b)) * relative_C
    
    out = owlqn(nx * ny, evaluate, progress, ORTHANTWISE_C)

    return spfft.idctn(out.reshape((nx, ny)).T, norm='ortho') + 1


#test = np.array([1,2,3,4,5])
#print(pywt.dwt(test, 'haar'))
#print(pywt.dwt(pywt.dwt(test, 'haar')[0], 'haar'))
#exit()

test = np.array([[1,2,3,4,5],
                [5,6,7,8,6],
                [9,10,11,12,6],
                [13,14,15,16,3],
                [14,5,6,7,7]], dtype=float)
mask = np.array([[1,1,1,0],
                [0,1,0,1],
                [1,0,1,0],
                [0,1,0,1]], dtype=bool)

ri = np.where(mask.flatten())[0]
b = test.flatten()[ri].astype(float)
ny, nx = test.shape

set_global_param(b, (ny, nx), ri)

out = owlqn(nx * ny, evaluate, progress, 0.5)

test = Image.open("/scratchdata/depth_prompting_nyu/gt/0.png")
test = np.array(test, dtype=float)
print(test.shape)
output = dwt2_haar_recursive(test)
print(output.max(), output.min()) 

plt.imsave("input.png", test, cmap='gray')
plt.imsave("output.png", np.log(abs(output)+1), cmap='gray')