import scipy.misc
import numpy as np

def vis(X, filename, n=20, w=28):
    assert len(X) >= n*n
    X = X[:n*n]
    X = X.reshape((n, n, w, w))
    img = np.zeros((n*w, n*w))
    for i in range(n):
        for j in range(n):
            img[i*w:(i+1)*w, j*w:(j+1)*w] = X[i, j, :, :]
    img = img.clip(0, 255).astype(np.uint8)
    scipy.misc.imsave(filename, img)
