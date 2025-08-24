from pylbfgs import owlqn
import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt

def dwt2_haar_recursive(arr, scale, r=-1):
    coeffs = np.zeros_like(arr, dtype=float)
    height, width = arr.shape
    if (height == 1 and width == 1) or r==0:
        return arr
    
    A, (H, V, D) = pywt.dwt2(arr, 'haar', mode='periodization')
    coeffs[:(height+1)//2, (width+1)//2:] = V[:(height+1)//2, :width - (width+1)//2]
    coeffs[(height+1)//2:, :(width+1)//2] = H[:height-(height+1)//2, :(width+1)//2]
    coeffs[(height+1)//2:, (width+1)//2:] = D[:height-(height+1)//2, :width - (width+1)//2]
    A = dwt2_haar_recursive(A, scale,r-1) / scale
    coeffs[:(height+1)//2, :(width+1)//2] = A[:(height+1)//2, :(width+1)//2]
     
    return coeffs

def idwt2_haar_recursive(coeffs, scale, r=-1):    
    height, width = coeffs.shape
    if (height == 1 and width == 1) or r==0:
        return coeffs
    coeffs = np.pad(coeffs, ((0, height%2), (0, width%2)), mode='constant')  # Ensure even dimensions for IDWT
    A = coeffs[:(height+1)//2, :(width+1)//2]
    A = idwt2_haar_recursive(A, scale,r-1) * scale
    V = coeffs[:(height+1)//2, (width+1)//2:]
    H = coeffs[(height+1)//2:, :(width+1)//2]
    D = coeffs[(height+1)//2:, (width+1)//2:]

    arr = pywt.idwt2((A, (H,V,D)), 'haar', mode='periodization')
    arr = arr[:height, :width]  # Ensure the output matches the original shape
    return arr

def set_global_param(b_vector, image_dims, ri_vector, scale):
    """Set the global parameters for the evaluation function.
    
    Args:
        b_vector (np.ndarray): The b vector used in the evaluation.
        image_dims (tuple): The dimensions of the image (ny, nx).
        ri_vector (np.ndarray): The sampling vector indicating valid indices.
    """
    global _b_vector, _image_dims, _ri_vector, _scale

    _b_vector = b_vector
    _image_dims = image_dims
    _ri_vector = ri_vector
    _scale = scale

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
    Ax2 = idwt2_haar_recursive(x2, _scale)

    # stack columns and extract samples
    Ax = Ax2.T.flat[_ri_vector].reshape(_b_vector.shape)

    # calculate the residual Ax-b and its 2-norm squared
    Axb = Ax - _b_vector
    fx = np.sum(np.power(Axb, 2))

    # project residual vector (k x 1) onto blank image (ny x nx)
    Axb2 = np.zeros(x2.shape)
    Axb2.T.flat[_ri_vector] = Axb  # fill columns-first

    # A'(Ax-b) is just the 2D dct of Axb2
    AtAxb2 = 2 * dwt2_haar_recursive(Axb2, _scale)
    AtAxb = AtAxb2.T.reshape(x.shape)  # stack columns

    # copy over the gradient vector
    np.copyto(g, AtAxb)

    return fx

def progress(x, g, fx, xnorm, gnorm, step, k, ls):
    """Just display the current iteration.
    """
    #print('Iteration {}'.format(k))
    return 0

def rescale_ratio(depth, est, scale, ORTHANTWISE_C=5, relative_C=None):
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
    ratio = np.log(ratio)
    ri = np.where(ri.T.flatten())[0]
    b = ratio.T.flatten()[ri].astype(float)
    ny, nx = ratio.shape

    set_global_param(b, (ny, nx), ri, scale)
    
    if not relative_C is None:
        ORTHANTWISE_C = np.mean(np.abs(b)) * relative_C
    
    out = owlqn(nx * ny, evaluate, progress, ORTHANTWISE_C)

    return np.exp(idwt2_haar_recursive(out.reshape((nx, ny)).T, 2))

if __name__ == "__main__":
    i = 300
    Xorig = Image.open(f"/scratchdata/depth_prompting_nyu/gt/{i}.png")
    Xorig = np.array(Xorig, dtype=float) / 1000
    Xpred = Image.open(f"/scratchdata/depth_prompting_nyu/depthformer/{i}.png")
    Xpred = np.array(Xpred, dtype=float) / 1000

    tmp = dwt2_haar_recursive(Xorig, scale=4)
    print(tmp.max(), tmp.min())

    np.random.seed(42)
    R = 0.5
    # Sample some r percent of the pixels
    Xsample = Xorig.copy()
    mask = np.random.rand(*Xorig.shape) < R
    Xsample[~mask] = 0  # Set unselected pixels to 0
    plt.imsave("sampled.png", Xsample, cmap='gray')

    new_ratio = rescale_ratio(Xsample, Xpred, scale=4, ORTHANTWISE_C=0.00005)
    print(new_ratio.max(), new_ratio.min())
    plt.imsave("ratio.png", new_ratio, cmap='gray')

    exit()

    scale = 1
    test = np.array([[1,2,3,4,5],[-1,-2,-3,-4,-5]]) 
    print(dwt2_haar_recursive(test,scale))
    mask = np.array([[1,1,1,0,1],[1,1,0,1,1]])
    ri = np.where(mask.T.flatten())[0]
    b = test.T.flatten()[ri].astype(float)
    ny, nx = test.shape
    set_global_param(b, (ny,nx), ri, scale)
    out = owlqn(nx * ny, evaluate, progress, 0.00005)
    print(out.reshape((nx, ny)).T)
    print(idwt2_haar_recursive(out.reshape((nx, ny)).T, scale))