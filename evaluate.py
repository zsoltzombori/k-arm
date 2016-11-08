import numpy as np
from vis import *
from diff_vis import *
import os.path
import scipy.misc

def evaluate(X_test, Y_learned, W_learned, iteration, threshold, outputPrefix, resultFile=None, dictOutput=None):
    X_prime_learned = np.dot(Y_learned, W_learned)
    X_prime_learned = X_prime_learned.reshape(X_test.shape)
    testSize = X_test.shape[0]
    dict_size = Y_learned.shape[1]
    nb_features = np.prod(X_test.shape[1:])
    
    # histogram to show sparsity
    nonzero = np.apply_along_axis(np.count_nonzero, axis=1, arr=Y_learned)
    nonzeroHist = np.histogram(nonzero, bins=10)
    print nonzeroHist[0]
    print nonzeroHist[1]

    # histogram to show how much the dictionary is used
    nonzeroW = np.apply_along_axis(np.count_nonzero, axis=0, arr=Y_learned)
    nonzeroWHist = np.histogram(nonzeroW, bins=10)
    print nonzeroWHist[0]
    print nonzeroWHist[1]

    print "Average density of nonzero elements in the code: ", np.average(nonzero) / dict_size
    nonzeroInt = int(np.average(nonzero))
    print "Average number of nonzero elements in the code: ", nonzeroInt
    reconsError = 0.5 * np.sum(np.square(X_prime_learned-X_test)) / testSize / nb_features
    print "Reconstruction error: ", reconsError
    sparsity_loss = threshold * np.sum(np.abs(Y_learned)) / testSize / nb_features
    total_loss = reconsError + sparsity_loss
    print "Total loss: ", total_loss

    if not os.path.exists("recons"): os.makedirs("recons")
    if not os.path.exists("dictImage"): os.makedirs("dictImage")

    suffix = "it{}_th{}_dict{}_test{}_loss{:.3f}.png".format(iteration,threshold,dict_size,testSize,total_loss)
    vis_image(X_prime_learned * 255, "recons/" + outputPrefix + "_recons_" + suffix)
    W_scaled = W_learned - np.min(W_learned)
    W_scaled /= np.max(W_scaled)
    W_scaled *= 255
    W_scaled = np.reshape(W_scaled, [dict_size, int(np.sqrt(nb_features)), -1])
    vis(W_scaled, "dictImage/" + outputPrefix + "_dict_" + suffix, n=int(np.sqrt(dict_size)))
#    diff_vis(X_test[:400],X_prime_learned[:400],28,28,20,20, "diff/" + outputPrefix + "_diff_" + suffix)

    # if resultFile was provided then add new line to result file
    if resultFile is not None:
        if os.path.exists(resultFile):
            with open(resultFile, "a") as file:
                file.write("{};{};{};{};{};{}\n".format(iteration,threshold,reconsError,sparsity_loss,nonzeroInt,total_loss))
        else:
            with open(resultFile, "w") as file:
                file.write("Iteration;Threshold;ReconsLoss;SparsityLoss;Nonzero;Loss\n")
                file.write("{};{};{};{};{};{}\n".format(iteration,threshold,reconsError,sparsity_loss,nonzeroInt,total_loss))

    if dictOutput is not None:
        with open(dictOutput, "wb") as npzfile:
                np.savez(npzfile, W_learned)
    
