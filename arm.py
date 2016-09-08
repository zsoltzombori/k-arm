import numpy as np

from scipy.linalg import eigh as largest_eigh
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh

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

dict_size = 100
input_size = X_train.shape[0]
W = np.random.normal(size=[nb_features,dict_size])
WT = np.matrix.transpose(W)
zeroOut = np.zeros(shape=[input_size,dict_size])
print X_train.shape
print W.shape
print WT.shape
print zeroOut.shape

# evals_large, evecs_large = largest_eigh(np.dot(W,WT), eigvals=(N-2,N-1))
#print evals_large
#print evecs_large

iterations = 10
threshold = 0.1
alpha = 0.002

def armderiv(x,y,W,WT,alpha,threshold):
    linout = y - alpha * np.dot((np.dot(y,WT) - x),W)
    out = np.sign(linout) * np.maximum(0,np.absolute(linout) - threshold)
    return out

def arm(x,count,W,WT,alpha,threshold):
    if count==0:
        outApprox = zeroOut 
    else:
        outApprox = arm(x,count-1,W,WT,alpha,threshold)
    return armderiv(x,outApprox,W,WT,alpha,threshold)


output = arm(X_train,iterations,W,WT,alpha,threshold)


