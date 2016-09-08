import numpy as np
from sklearn.decomposition import SparseCoder, DictionaryLearning
from sklearn.preprocessing import normalize
from keras.datasets import mnist
import scipy.misc

f = 28*28
c = 400
n = 400
density = 0.05


def vis(X, filename, n=20):
    w = 28
    assert len(X) >= n*n
    X = X[:n*n]
    X = X.reshape((n, n, w, w))
    img = np.zeros((n*w, n*w))
    for i in range(n):
        for j in range(n):
            img[i*w:(i+1)*w, j*w:(j+1)*w] = X[i, j, :, :]
    img = img.clip(0, 255).astype(np.uint8)
    scipy.misc.imsave(filename, img)


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, f)
X = X_train[:n]
X_test = X_test[:n].reshape(n, f)

vis(X_test, "orig.png")

do_random_dictionary_test = False

if do_random_dictionary_test:
    W = np.random.normal(size=(c, f))
    W = normalize(W, axis=1)

    print X.shape, W.shape

    sc = SparseCoder(dictionary=W, transform_n_nonzero_coefs=int(c * density))

    Y = sc.transform(X)

    print "nonzero ratio:", float(np.count_nonzero(Y)) / Y.size

    X_prime = np.dot(Y, W)

    print X.shape, X_prime.shape
    vis(X_prime, "recons.png")


cached = True

if not cached:
    dl = DictionaryLearning(n_components=c, max_iter=10, transform_n_nonzero_coefs=int(c * density), verbose=True)
    dl.fit(X)
    print

    W_learned = dl.components_

    with file("dict.npz", "wb") as npzfile:
        np.savez(npzfile, W_learned)

    Y_learned = dl.transform(X_test)

else:
    W_learned = np.load(file("dict.npz"))['arr_0']
    sc = SparseCoder(dictionary=W_learned, transform_n_nonzero_coefs=int(c * density))
    Y_learned = sc.transform(X_test)

X_prime_learned = np.dot(Y_learned, W_learned)

nonzero = np.apply_along_axis(np.count_nonzero, axis=1, arr=Y_learned)
print "Average density of nonzero elements in the code: ", np.average(nonzero) / c

reconsError = np.sum(np.square(X_prime_learned-X_test)) / n / 255 / 255
print "Reconstruction error: ", reconsError

vis(X_prime_learned, "dl.png")

W_scaled = W_learned - np.min(W_learned)
W_scaled /= np.max(W_scaled)
W_scaled *= 255

vis(W_scaled, "dict.png", n=10)
