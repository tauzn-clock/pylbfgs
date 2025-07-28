import numpy as np
from pylbfgs import owlqn
import scipy.fft as spfft
from compressed_sensing import evaluate, progress, set_global_param

global _b_vector, _A_matrix, _image_dims, _ri_vector
ORTHANTWISE_C = 5

X = np.array([[1,1.1,1,1.1]]) * 10
ri = np.array([[1,1,1,0]]).astype(bool).flatten()
ri = np.where(ri)[0]  # Get the indices of the valid values
b = X.T.flatten()[ri].astype(float)  # Valid values turned into column vector
ny, nx = X.shape

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

# save image dims, sampling vector, and b vector and to global vars
set_global_param(b, (ny, nx), ri)

# perform the L1 minimization in memory
Xat2 = owlqn(nx*ny, evaluate, progress, ORTHANTWISE_C)

Xat = Xat2.reshape(nx, ny).T  # stack columns
print("Xat:", Xat)
Xa = spfft.idctn(Xat, norm='ortho')
print("Xa:", Xa)