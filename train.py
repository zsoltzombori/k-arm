import numpy as np
import argparse
import data
from model import *
from vis import *

parser = argparse.ArgumentParser(description="Image classifer using sparsifying arm layers.")
parser.add_argument('--iteration', dest="iteration", type=int, default=6, help="Number of iterations in k-arm approximation")
parser.add_argument('--threshold', dest="threshold", type=float, default=0.02, help="Sparsity coefficient")
parser.add_argument('--dict', dest="dict_size", type=int, default=400, help="Size of the feature dictionary")
parser.add_argument('--epoch', dest="epoch", type=int, default=5, help="Number of epochs")
parser.add_argument('--lr', dest="lr", type=float, default=0.001, help="learning rate")
parser.add_argument('--batch', dest="batchSize", type=int, default=128, help="Batch size")
parser.add_argument('--armLayers', dest="armLayers", type=int, default=1, help="Arm layer count")
parser.add_argument('--denseLayers', dest="denseLayers", type=int, default=1, help="Dense layer count")
parser.add_argument('--convLayers', dest="convLayers", type=int, default=0, help="Convolution layer count")
parser.add_argument('--dataset', dest="dataset", default="cifar10", help="mnist/cifar10")
args = parser.parse_args()

print "dataset: {}, convLayers: {}, armLayers: {}, denseLayers: {}, iteration: {}, threshold: {}, dict_size: {}, lr: {}, batch: {}, epoch: {}".format(args.dataset,args.convLayers,args.armLayers,args.denseLayers,args.iteration,args.threshold,args.dict_size,args.lr,args.batchSize,args.epoch)
(X_train, Y_train), (X_test, Y_test), datagen, test_datagen, nb_classes = data.load_data(args.dataset)

model = build_classifier(X_train.shape, nb_classes, args.convLayers, args.armLayers, args.denseLayers, args.lr, args.iteration, args.threshold, args.dict_size)

model.summary()

# fit the model on the batches generated by datagen.flow()
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=args.batchSize, shuffle=True),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=args.epoch,
                        validation_data=test_datagen.flow(X_test, Y_test, batch_size=args.batchSize),
                        nb_val_samples=X_test.shape[0]
                        )

lastArmLayer = model.get_layer(name="arm_{}".format(args.armLayers-1))

if args.dataset == "mnist":
    W_learned = lastArmLayer.W.eval()
    W_scaled = W_learned - np.min(W_learned)
    W_scaled /= np.max(W_scaled)
    W_scaled *= 255
    vis(W_scaled, "dictImage/classif.png", n=int(np.sqrt(args.dict_size)))
    

