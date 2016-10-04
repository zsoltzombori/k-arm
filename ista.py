import numpy as np
from arm import ArmLayer
from keras.datasets import mnist
from vis import *
from model import *
import data
import argparse
import re
from diff_vis import *
import os.path

parser = argparse.ArgumentParser(description="Sparse image encoding using k-arm.")
parser.add_argument('--iteration', dest="iteration", type=int, default=1, help="Number of iterations in k-arm approximation")
parser.add_argument('--threshold', dest="threshold", type=float, default=0.5, help="Sparsity coefficient")
parser.add_argument('--dict', dest="dict", type=int, default=1000, help="Size of the feature dictionary")
parser.add_argument('--weightFile', dest="weightFile", default=None, help="dictionary matrix file")
parser.add_argument('--testSize', dest="testSize", type=int, default=2000, help="Test set size")
args = parser.parse_args()
dict_size = args.dict
iteration = args.iteration
threshold = args.threshold
weightFile = args.weightFile
testSize = args.testSize

(X_train, Y_train), (X_test, Y_test), datagen, test_datagen, nb_classes = data.load_mnist()
X_test = X_test[:testSize]
vis(X_test * 255, "orig.png")
nb_features = np.prod(X_test.shape[1:])

resultFile = "results.csv"

if weightFile is not None:
    weights = np.load(file(weightFile))['arr_0']
    trainIteration = re.search('it(.+?)_', weightFile).group(1)
    trainThreshold = re.search('th(.+?).npz', weightFile).group(1)
else:
    weights = None
    trainIteration = 0
    trainTheshold = 0

model = build_encode_decode_layer(input_shape=X_test.shape, iteration=iteration, threshold=threshold, dict_size=dict_size, weights=weights)

y_fun = K.function([model.layers[0].input], [model.layers[1].output])
Y_learned = y_fun([X_test])[0]
X_prime_learned = model.predict_on_batch(X_test)
    
#model = build_single_layer(input_shape=X_test.shape, iteration=iteration, threshold=threshold, dict_size=dict_size, weights=weights)
#Y_learned = model.predict_on_batch(X_test)
#W_learned = model.layers[1].get_weights()[0]
#X_prime_learned = np.dot(Y_learned, W_learned)


nonzero = np.apply_along_axis(np.count_nonzero, axis=1, arr=Y_learned)
print "Average density of nonzero elements in the code: ", np.average(nonzero) / dict_size
nonzeroInt = int(np.average(nonzero))
print "Average number of nonzero elements in the code: ", nonzeroInt

reconsError = np.sum(np.square(X_prime_learned-X_test)) / testSize / nb_features
print "Reconstruction error: ", reconsError

sparsity_loss = threshold * np.sum(np.abs(Y_learned)) / testSize / nb_features
total_loss = reconsError + sparsity_loss
print "Total loss: ", total_loss

outputFile = "output/ista_it{}_th{}_trainit{}_trainth{}_loss{:.3f}.png".format(iteration,threshold,trainIteration,trainThreshold,total_loss)
vis(X_prime_learned * 255, outputFile)

diff_vis(X_test[:400],X_prime_learned[:400],28,28,20,20,"diff")

if os.path.exists(resultFile):
    with open(resultFile, "a") as file:
        file.write("{},{},{},{},{}\n".format(trainIteration,trainThreshold,iteration,threshold,total_loss))
else:
    with open(resultFile, "w") as file:
        file.write("TrainIteration,TrainThreshold,Iteration,Threshold,Loss\n")
        file.write("{},{},{},{},{}\n".format(trainIteration,trainThreshold,iteration,threshold,total_loss))
 
