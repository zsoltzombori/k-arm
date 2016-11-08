import os.path
import argparse
import numpy as np
from sklearn.decomposition import SparseCoder, DictionaryLearning, MiniBatchDictionaryLearning, TruncatedSVD
from sklearn.preprocessing import normalize
from keras.datasets import mnist
import scipy.misc
from evaluate import *
import data

parser = argparse.ArgumentParser(description="Sparse image encoding using k-arm.")
parser.add_argument('--threshold', dest="threshold", type=float, default=0.01, help="Sparsity coefficient")
parser.add_argument('--dict', dest="dict", type=int, default=400, help="Size of the feature dictionary")
parser.add_argument('--trainSize', dest="trainSize", type=int, default=5000, help="Training set size")
parser.add_argument('--testSize', dest="testSize", type=int, default=1000, help="Test set size")
parser.add_argument('--batch', dest="batchSize", type=int, default=128, help="Batch size")
parser.add_argument('--iteration', dest="iteration", type=int, default=10, help="Iterations")
parser.add_argument('--resultFile', dest="resultFile", default=None, help="File to write results")
parser.add_argument('--dictInput', dest="dictInput", default=None, help="File to read dictionary from")
parser.add_argument('--dictOutput', dest="dictOutput", default=None, help="File to write dictionary")
parser.add_argument('--method', dest="method", default="omp", help="Learning method:[omp,lasso,svd]")
parser.add_argument('--density', dest="density", type=float, default=0.1, help="Output code density")
args = parser.parse_args()
dict_size = args.dict
threshold = args.threshold
trainSize = args.trainSize
testSize = args.testSize 
batchSize = args.batchSize
iteration = args.iteration

(X_train, Y_train), (X_test, Y_test), datagen, test_datagen, nb_classes = data.load_data('mnist')
nb_features = np.prod(X_test.shape[1:])
X_train = X_train[:trainSize].reshape(trainSize, nb_features)
X_test = X_test[:testSize].reshape(testSize, nb_features)

if args.dictInput is None:
    if args.method == "omp":
        coder = MiniBatchDictionaryLearning(n_components=dict_size, transform_algorithm='omp', alpha=threshold, transform_alpha=threshold, transform_n_nonzero_coefs=int(dict_size*args.density), batch_size=batchSize, n_iter=iteration, verbose=True)
    elif args.method == "lasso":
        coder = MiniBatchDictionaryLearning(n_components=dict_size, transform_algorithm='lasso_lars', transform_alpha=threshold, batch_size=batchSize, n_iter=iteration, verbose=True)
    elif args.method == "svd":
        coder = TruncatedSVD(n_components=dict_size)
        
    coder.fit(X_train)
    W_learned = coder.components_
else:
    W_learned = np.load(file(args.dictInput))['arr_0']
    coder = SparseCoder(dictionary=W_learned, transform_n_nonzero_coefs=int(dict_size * args.density))

Y_learned = coder.transform(X_test)
evaluate(X_test, Y_learned, W_learned, iteration, threshold, args.method, args.resultFile, args.dictOutput)
