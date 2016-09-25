import numpy as np
from arm import ArmLayer
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.preprocessing import normalize
from keras import backend as K
from keras.callbacks import Callback

default_iteration = 2
default_threshold = 0.5
default_dict_size = 1000
weight_decay = 1e-4



def build_model(input_shape, nb_classes, layer_count=1, iterations=None, thresholds=None, dict_sizes=None):
    assert layer_count > 0
    input = Input(shape=input_shape[1:])
    if iterations is not None:
        assert len(iterations) == layer_count
    else:
        iterations = [default_iteration] * layer_count

    if thresholds is not None:
        assert len(thresholds) == layer_count
    else:
        thresholds = [default_threshold] * layer_count

    if dict_sizes is not None:
        assert len(dict_sizes) == layer_count
    else:
        dict_sizes = [default_dict_size] * layer_count

    output = input
    for i in range(layer_count):
        output = ArmLayer(
            dict_size=dict_sizes[i],
            iteration = iterations[i],
            threshold = thresholds[i])(output)
    output = Dense(nb_classes, activation="softmax", W_regularizer=l2(weight_decay))(output)
    model = Model(input=input, output=output)
    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_single_layer(input_shape, iteration, threshold, dict_size, weights=None, shared_weights=None):
    input = Input(shape=input_shape[1:])
    output = ArmLayer(
        dict_size=dict_size,
        iteration = iteration,
        threshold = threshold,
        weights = weights,
        shared_weights = shared_weights)(input)
    model = Model(input=input, output=output)
    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def mse_loss(y_true,y_pred):
    return K.sum(K.square(y_true - y_pred[0]))

def build_encode_decode_layer(input_shape, iteration, threshold, dict_size, weights, shared_weights):
    input = Input(shape=input_shape[1:])
    nb_features = np.prod(input_shape[1:])
    armLayer = ArmLayer(
        dict_size=dict_size,
        iteration = iteration,
        threshold = threshold,
        weights = weights,
        shared_weights = shared_weights)
    Y = armLayer(input)
    lambdaLayer = Lambda(lambda x: K.dot(x,armLayer.W), output_shape=[nb_features], name="decode_layer")    
    output = lambdaLayer((Y))
    output = Reshape(input_shape[1:])(output)
    model = Model(input=input, output=output)
    sgd = SGD(lr=0.000001, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="mse")
    sparsity_loss = threshold * K.sum(K.abs(Y)) / (28*28*128)
    model.total_loss += sparsity_loss
    return model
