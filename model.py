import numpy as np
from arm import ArmLayer
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from keras.optimizers import *
from keras.regularizers import l2
from sklearn.preprocessing import normalize
from keras import backend as K
from keras.callbacks import Callback

weight_decay = 1e-4



def build_classifier(input_shape, nb_classes, layers, iteration, threshold, dict_size, lr):
    assert layers > 0
    input = Input(shape=input_shape[1:])
    output = input
    for i in range(layers):
        output = ArmLayer(
            dict_size=dict_size,
            iteration = iteration,
            threshold = threshold)(output)
    output = Dense(nb_classes, activation="softmax", W_regularizer=l2(weight_decay))(output)
    model = Model(input=input, output=output)
    optimizer = RMSprop(lr=lr)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_single_layer(input_shape, iteration, threshold, dict_size, weights=None):
    input = Input(shape=input_shape[1:])
    output = ArmLayer(
        dict_size=dict_size,
        iteration = iteration,
        threshold = threshold,
        weights = weights)(input)
    model = Model(input=input, output=output)
    rmsprop = RMSprop()
    adagrad = Adagrad()
    model.compile(optimizer=adagrad, loss="mse")
    return model

def build_encode_decode_layer(input_shape, iteration, threshold, dict_size, weights, lr):
    input = Input(shape=input_shape[1:])
    nb_features = np.prod(input_shape[1:])
    armLayer = ArmLayer(
        dict_size=dict_size,
        iteration = iteration,
        threshold = threshold,
        weights = weights)
    Y = armLayer(input)
    lambdaLayer = Lambda(lambda x: K.dot(x,armLayer.W), output_shape=[nb_features], name="decode_layer")    
    output = lambdaLayer((Y))
    output = Reshape(input_shape[1:])(output)
    model = Model(input=input, output=output)
    optimizer = RMSprop(lr=lr)
    # optimizer = Nadam()
    
    model.compile(optimizer=optimizer, loss="mse")
    return model

def build_encode_decode_layers(input_shape, iteration, threshold, dict_size_list, lr, layers, weights_list=None):
    assert len(dict_size_list) >= layers
    if weights_list is not None:
        assert len(weights_list) == layers
    
    input = Input(shape=input_shape[1:])
    nb_features = np.prod(input_shape[1:])
    output = input
    
    # build layers number of arm layers
    Ws = []
    for i in range(layers):
        dict_size = dict_size_list[i]
        if weights_list is not None:
            weights = weights_list[i]
        else:
            weights = None
        
        currentLayer = ArmLayer(dict_size=dict_size,iteration=iteration, threshold=threshold, weights=weights, name="armLayer{}".format(i))
        output = currentLayer(output)
        Ws.append(currentLayer.W)

    # build layers number of decoding layers
    output_shape_list = [nb_features] + list(dict_size_list)
    for i in reversed(range(layers)):
        output = Lambda(lambda y: K.dot(y,Ws[i]), output_shape=[output_shape_list[i]], name="decodeLayer{}".format(i))(output)

    # restore the original shape of the input
    output = Reshape(input_shape[1:], name="reshapeLayer")(output)
    model = Model(input=input, output=output)
    rmsprop = RMSprop(lr=lr)
    model.compile(optimizer=rmsprop, loss="mse")
    return model
