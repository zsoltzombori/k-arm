import numpy as np
from arm import ArmLayer
from vis import *
from diff_vis import *
from model import *
import data
import argparse
import re
import os.path
import sys
from evaluate import *
sys.setrecursionlimit(2**20)

parser = argparse.ArgumentParser(description="Sparse image encoding using k-arm.")
parser.add_argument('--iteration', dest="iteration", type=int, default=6, help="Number of iterations in k-arm approximation")
parser.add_argument('--threshold', dest="threshold", type=float, default=0.5, help="Sparsity coefficient")
parser.add_argument('--dict', dest="dict", type=int, default=400, help="Size of the feature dictionary")
parser.add_argument('--epoch', dest="epoch", type=int, default=30, help="Number of epochs")
parser.add_argument('--lr', dest="lr", type=float, default=0.001, help="learning rate")
parser.add_argument('--trainSize', dest="trainSize", type=int, default=5000, help="Training set size")
parser.add_argument('--testSize', dest="testSize", type=int, default=1000, help="Test set size")
parser.add_argument('--batch', dest="batchSize", type=int, default=16, help="Batch size")
parser.add_argument('--resultFile', dest="resultFile", default=None, help="File to write results")
parser.add_argument('--dictInput', dest="dictInput", default=None, help="File to read dictionary from")
parser.add_argument('--dictOutput', dest="dictOutput", default=None, help="File to write dictionary")
args = parser.parse_args()
dict_size = args.dict
iteration = args.iteration
threshold = args.threshold
nb_epoch = args.epoch
trainSize = args.trainSize
testSize = args.testSize 
batchSize=args.batchSize


print "Dict: {}, \nThreshold: {}, \nTrainSize: {}, \nTestSize: {}, \nBatchSize: {}".format(dict_size, threshold, trainSize, testSize, batchSize)

(X_train, Y_train), (X_test, Y_test), datagen, test_datagen, nb_classes = data.load_mnist()
X_train = X_train[:trainSize]
X_test = X_test[:testSize]
vis(X_test * 255, "orig.png")
nb_features = np.prod(X_test.shape[1:])

if args.dictInput is not None:
    weights = np.load(file(args.dictInput))['arr_0']
else:
    weights = None

model = build_encode_decode_layer(input_shape=X_test.shape, iteration=iteration, threshold=threshold, dict_size=dict_size, weights=weights, lr=args.lr)

#fit the model on the batches generated by datagen.flow()
model.fit_generator(datagen.flow(X_train, X_train, batch_size=batchSize, shuffle=True),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=test_datagen.flow(X_test, X_test, batch_size=batchSize),
                        nb_val_samples=X_test.shape[0]
                        )

y_fun = K.function([model.layers[0].input], [model.layers[1].output])
Y_learned = y_fun([X_test])[0]
W_learned = model.layers[1].get_weights()[0]

evaluate(X_test, Y_learned, W_learned, iteration, threshold, "lista", args.resultFile, args.dictOutput)


# X_prime_learned = model.predict_on_batch(X_test)

# nonzero = np.apply_along_axis(np.count_nonzero, axis=1, arr=Y_learned)
# nonzeroHist = np.histogram(nonzero, bins=10)
# print nonzeroHist[0]
# print nonzeroHist[1]

# nonzeroW = np.apply_along_axis(np.count_nonzero, axis=0, arr=Y_learned)
# nonzeroWHist = np.histogram(nonzeroW, bins=10)
# print nonzeroWHist[0]
# print nonzeroWHist[1]

# print "Average density of nonzero elements in the code: ", np.average(nonzero) / dict_size
# nonzeroInt = int(np.average(nonzero))
# print "Average number of nonzero elements in the code: ", nonzeroInt
# reconsError = np.sum(np.square(X_prime_learned-X_test)) / testSize / nb_features
# print "Reconstruction error: ", reconsError
# sparsity_loss = threshold * np.sum(np.abs(Y_learned)) / testSize / nb_features
# total_loss = reconsError + sparsity_loss
# print "Total loss: ", total_loss

# suffix = "it{}_th{}_dict{}_lr{}_train{}_test{}_batch{}_epoch{}_loss{:.3f}.png".format(iteration,threshold,dict_size,lr,trainSize,testSize,batchSize,nb_epoch,total_loss)
# outputFile = "output/lista_"+suffix
# vis(X_prime_learned * 255, outputFile)
# 
# W_scaled = W_learned - np.min(W_learned)
# W_scaled /= np.max(W_scaled)
# W_scaled *= 255
# outputFile = "output/lista_dict_" + suffix
# vis(W_scaled, outputFile, n=int(np.sqrt(dict_size)))
# outputFile = "output/lista_diff_" + suffix
# diff_vis(X_test[:400],X_prime_learned[:400],28,28,20,20,outputFile)


# # if resultFile was provided then add new line to result file
# if resultFile is not None:
#     if os.path.exists(resultFile):
#         with open(resultFile, "a") as file:
#             file.write("{};{};{};{};{};{}\n".format(iteration,threshold,reconsError,sparsity_loss,nonzeroInt,total_loss))
#     else:
#         with open(resultFile, "w") as file:
#             file.write("Iteration;Threshold;ReconsLoss;SparsityLoss;Nonzero;Loss\n")
#             file.write("{};{};{};{};{};{}\n".format(iteration,threshold,reconsError,sparsity_loss,nonzeroInt,total_loss))
            
