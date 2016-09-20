from arm import ArmLayer
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD

default_iteration = 20
default_threshold = 0.05
default_dict_size = 400

def build_model(input_shape, layer_count=1, iterations=None, thresholds=None, dict_sizes=None):
    assert layer_count > 0
    batch_size = input_shape[0]
    input = Input(shape=input_shape)
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
            batch_size=batch_size,
            iteration = iterations[i],
            threshold = thresholds[i],
            dict_size=dict_sizes[i])(output)
    model = Model(input=input, output=output)
    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
