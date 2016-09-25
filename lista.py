import numpy as np
from arm import ArmLayer
from vis import *
from model import *
import data
import argparse

parser = argparse.ArgumentParser(description="Sparse image encoding using k-arm.")
parser.add_argument('--iteration', dest="iteration", type=int, default=1, help="Number of iterations in k-arm approximation")
parser.add_argument('--threshold', dest="threshold", type=float, default=0.5, help="Sparsity coefficient")
parser.add_argument('--dict', dest="dict", type=int, default=1000, help="Size of the feature dictionary")
args = parser.parse_args()
dict_size = args.dict
iteration = args.iteration
threshold = args.threshold
n = 10000
batch_size=128
nb_epoch = 30
import sys
sys.setrecursionlimit(2**20)

(X_train, Y_train), (X_test, Y_test), datagen, test_datagen, nb_classes = data.load_mnist()
X_train = X_train[:n]
X_test = X_test[:n]
Y_train = Y_train[:n]
Y_test = Y_test[:n]
vis(X_test * 255, "orig.png")

nb_features = np.prod(X_test.shape[1:])
weights = np.random.normal(size=[dict_size, nb_features])
weights = normalize(weights, axis=1)
weights = weights.astype('float32')
shared_weights = K.variable(weights, name='shared_W')

model = build_encode_decode_layer(input_shape=X_test.shape, iteration=iteration, threshold=threshold, dict_size=dict_size, weights=weights, shared_weights=shared_weights)
nb_val_samples = X_test.shape[0]
samples_per_epoch = X_train.shape[0]   
# fit the model on the batches generated by datagen.flow()
model.fit_generator(datagen.flow(X_train, X_train, batch_size=batch_size, shuffle=True),
                        samples_per_epoch=samples_per_epoch,
                        nb_epoch=nb_epoch,
                        validation_data=test_datagen.flow(X_test, X_test, batch_size=batch_size),
                        nb_val_samples=nb_val_samples
                        )
X_prime_learned = model.predict_on_batch(X_test)

#nonzero = np.apply_along_axis(np.count_nonzero, axis=1, arr=Y_learned)
#print "Average density of nonzero elements in the code: ", np.average(nonzero) / dict_size

reconsError = np.sum(np.square(X_prime_learned-X_test)) / n
print "Reconstruction error: ", reconsError

vis(X_prime_learned * 255, "karm.png")
