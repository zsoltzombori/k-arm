import numpy as np

from scipy.linalg import eigh as largest_eigh
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from sklearn.preprocessing import normalize


from keras.datasets import mnist
# The data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

nb_features = 28*28
# flatten the 28x28 images to arrays of length 28*28:
X_train = X_train.reshape(60000, nb_features)
X_test = X_test.reshape(10000, nb_features)

# convert brightness values from bytes to floats between 0 and 1:
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

dict_size = 300

X = X_train[:999]

input_size = X.shape[0]
W = np.random.normal(size=[dict_size, nb_features])

W = normalize(W, axis=1)


zeroOut = np.zeros(shape=[input_size, dict_size])
print X.shape
print W.shape
print zeroOut.shape

print np.linalg.eigvals(W.dot(W.T))

# evals_large, evecs_large = largest_eigh(np.dot(W,WT), eigvals=(N-2,N-1))
#print evals_large
#print evecs_large

iterations = 10
threshold = 0.001
alpha = 0.001

def armderiv(x, y, W, alpha, threshold):
    linout = y - alpha * (y.dot(W) - x).dot(W.T)
    out = np.sign(linout) * np.maximum(0, np.absolute(linout) - threshold)
    print np.min(y), np.max(y), np.mean(y)
    return out

def arm(x, count, W, alpha, threshold):
    if count==0:
        outApprox = zeroOut 
    else:
        outApprox = arm(x, count-1, W, alpha, threshold)
    return armderiv(x, outApprox, W, alpha, threshold)


Y = arm(X,iterations,W,alpha,threshold)
X_prime = Y.dot(W)


print np.histogram(X)
print np.histogram(X_prime)
print np.histogram(X - X_prime)

print np.linalg.norm(X), np.linalg.norm(X_prime), np.linalg.norm(X - X_prime)
