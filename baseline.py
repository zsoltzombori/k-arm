from diff_vis import *
import os.path
import argparse
import numpy as np
from sklearn.decomposition import SparseCoder, DictionaryLearning, MiniBatchDictionaryLearning
from sklearn.preprocessing import normalize
from keras.datasets import mnist
import scipy.misc

parser = argparse.ArgumentParser(description="Sparse image encoding using k-arm.")
parser.add_argument('--threshold', dest="threshold", type=float, default=0.01, help="Sparsity coefficient")
parser.add_argument('--dict', dest="dict", type=int, default=400, help="Size of the feature dictionary")
parser.add_argument('--trainSize', dest="trainSize", type=int, default=5000, help="Training set size")
parser.add_argument('--testSize', dest="testSize", type=int, default=1000, help="Test set size")
parser.add_argument('--batch', dest="batchSize", type=int, default=16, help="Batch size")
parser.add_argument('--iteration', dest="iteration", type=int, default=10, help="Iterations")
parser.add_argument('--resultFile', dest="resultFile", default=None, help="File to write results")
args = parser.parse_args()
dict_size = args.dict
threshold = args.threshold
trainSize = args.trainSize
testSize = args.testSize 
batchSize = args.batchSize
iteration = args.iteration
resultFile = args.resultFile
nb_features = 28*28
density = 0.1

print "Dict: {}, \nThreshold: {}, \nTrainSize: {}, \nTestSize: {}, \nBatchSize: {}".format(dict_size, threshold, trainSize, testSize, batchSize)
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
X_train = X_train[:trainSize].reshape(trainSize, nb_features)
X_test = X_test[:testSize].reshape(testSize, nb_features)
X_train = X_train / 255.0
X_test = X_test / 255.0
vis(X_test * 255, "orig.png")

# do_random_dictionary_test = False

# if do_random_dictionary_test:
#     W = np.random.normal(size=(dict_size, nb_features))
#     W = normalize(W, axis=1)

#     print X_train.shape, W.shape

#     sc = SparseCoder(dictionary=W, transform_n_nonzero_coefs=int(dict_size * density))

#     Y = sc.transform(X)

#     print "nonzero ratio:", float(np.count_nonzero(Y)) / Y.size

#     X_prime = np.dot(Y, W)

#     print X.shape, X_prime.shape
#     vis(X_prime * 255, "recons.png")


cached = False

if not cached:
    dl = MiniBatchDictionaryLearning(n_components=dict_size, transform_algorithm='lasso_lars', transform_alpha=threshold, batch_size=batchSize, n_iter=iteration, verbose=True)
    dl.fit(X_train)
    print

    W_learned = dl.components_

    with file("dict.npz", "wb") as npzfile:
        np.savez(npzfile, W_learned)

    Y_learned = dl.transform(X_test)

# else:
#     W_learned = np.load(file("dict.npz"))['arr_0']
#     sc = SparseCoder(dictionary=W_learned, transform_n_nonzero_coefs=int(dict_size * density))
#     Y_learned = sc.transform(X_test)

X_prime_learned = np.dot(Y_learned, W_learned)

nonzero = np.apply_along_axis(np.count_nonzero, axis=1, arr=Y_learned)
nonzeroHist = np.histogram(nonzero, bins=10)
print nonzeroHist[0]
print nonzeroHist[1]

nonzeroW = np.apply_along_axis(np.count_nonzero, axis=0, arr=Y_learned)
nonzeroWHist = np.histogram(nonzeroW, bins=10)
print nonzeroWHist[0]
print nonzeroWHist[1]

print "Average density of nonzero elements in the code: ", np.average(nonzero) / dict_size
nonzeroInt = int(np.average(nonzero))
print "Average number of nonzero elements in the code: ", nonzeroInt
reconsError = np.sum(np.square(X_prime_learned-X_test)) / testSize / nb_features
print "Reconstruction error: ", reconsError
sparsity_loss = threshold * np.sum(np.abs(Y_learned)) / testSize / nb_features
total_loss = reconsError + sparsity_loss
print "Total loss: ", total_loss

suffix = "it{}_th{}_dict{}_train{}_test{}_batch{}_loss{:.3f}.png".format(iteration,threshold,dict_size,trainSize,testSize,batchSize,total_loss)
outputFile = "output/lasso_"+suffix
vis(X_prime_learned * 255, outputFile)
W_scaled = W_learned - np.min(W_learned)
W_scaled /= np.max(W_scaled)
W_scaled *= 255
outputFile = "output/lasso_dict_" + suffix
vis(W_scaled, outputFile, n=int(np.sqrt(dict_size)))
outputFile = "output/lasso_diff_" + suffix
diff_vis(X_test[:400],X_prime_learned[:400],28,28,20,20,outputFile)

# if resultFile was provided then add new line to result file
if resultFile is not None:
    if os.path.exists(resultFile):
        with open(resultFile, "a") as file:
            file.write("{};{};{};{};{};{}\n".format(iteration,threshold,reconsError,sparsity_loss,nonzeroInt,total_loss))
    else:
        with open(resultFile, "w") as file:
            file.write("Iteration;Threshold;ReconsLoss;SparsityLoss;Nonzero;Loss\n")
            file.write("{};{};{};{};{};{}\n".format(iteration,threshold,reconsError,sparsity_loss,nonzeroInt,total_loss))
