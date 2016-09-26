import numpy as np
from arm import ArmLayer
from keras.datasets import mnist
from vis import *
from model import *
import data
import argparse

parser = argparse.ArgumentParser(description="Sparse image encoding using k-arm.")
parser.add_argument('--iteration', dest="iteration", type=int, default=1, help="Number of iterations in k-arm approximation")
parser.add_argument('--threshold', dest="threshold", type=float, default=0.5, help="Sparsity coefficient")
parser.add_argument('--dict', dest="dict", type=int, default=1000, help="Size of the feature dictionary")
parser.add_argument('--randomWeights', dest="randomWeights", type=int, default=0, help="0 if start from random weight matrix, 1 if use pretrained.")
args = parser.parse_args()
dict_size = args.dict
iteration = args.iteration
threshold = args.threshold
randomWeights = args.randomWeights
f = 28*28
n = 400

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.float32(X_train.reshape(60000, f))
X_test = np.float32(X_test.reshape(10000, f))
X_train = X_train[:n]
X_test = X_test[:n]
X_train /= 255
X_test /= 255
vis(X_test * 255, "orig.png")

if randomWeights == 1:
    weights = None
else:
    weights = np.load(file("dict1000.npz"))['arr_0']

model = build_single_layer(input_shape=X_test.shape, iteration=iteration, threshold=threshold, dict_size=dict_size, weights=weights)
Y_learned = model.predict_on_batch(X_test)
W_learned = model.layers[1].get_weights()[0]
X_prime_learned = np.dot(Y_learned, W_learned)


nonzero = np.apply_along_axis(np.count_nonzero, axis=1, arr=Y_learned)
print "Average density of nonzero elements in the code: ", np.average(nonzero) / dict_size

reconsError = np.sum(np.square(X_prime_learned-X_test)) / n
print "Reconstruction error: ", reconsError

vis(X_prime_learned * 255, "ista.png")
