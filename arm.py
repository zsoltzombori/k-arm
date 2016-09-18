import numpy as np
from sklearn.preprocessing import normalize
import scipy.misc
from keras import backend as K
from keras.engine.topology import Layer

class ArmLayer(Layer):
    def __init__(self, weights = None, iteration = 10, threshold = 0.05, dict_size = 400, batch_size=100, **kwargs):
        self.initial_weights = weights
        self.iteration = iteration
        self.threshold = threshold
        self.dict_size = dict_size
        self.batch_size = batch_size
        super(ArmLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        nb_features = np.prod(input_shape[1:])

        if self.initial_weights is not None:
            initial_weight_value = self.initial_weights
            print "Using provided weights"
        else:
            initial_weight_value = np.random.normal(size=[self.dict_size, nb_features])

        intial_weight_value = normalize(initial_weight_value, axis=1)

        # set alpha
        eigvals = np.linalg.eigvals(initial_weight_value.dot(initial_weight_value.T))
        maxEigval = np.max(np.absolute(eigvals))
        self.alpha = 1/maxEigval

        self.W = K.variable(initial_weight_value, name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]


    def armderiv(self,x, y):
        hard_thresholding = False
        linout = y - self.alpha * K.dot(K.dot(y,self.W) - x,self.W.T)
        if hard_thresholding:
            out = K.greater(K.abs(linout),self.threshold) * linout
        else:
            out = K.sign(linout) * K.max(K.abs(linout) - self.threshold,0)
        return out

    def arm(self, x, iteration):
        if iteration==0:
            outApprox = K.zeros(shape=[self.batch_size, self.dict_size])
        else:
            outApprox = self.arm(x, iteration-1)
        return self.armderiv(x, outApprox)

    def call(self, x, mask=None):
        # flatten the images to arrays
        x_flattened = K.reshape(x,[K.shape(x)[0],K.prod(K.shape(x)[1:])])
        
        y = self.arm(x_flattened, self.iteration)
        #nonzero = np.apply_along_axis(np.count_nonzero, axis=1, arr=y)
        #averageDensity = np.average(nonzero) / self.dict_size
        reconsError = K.sum(K.square(y.dot(self.W)-x_flattened))/x_flattened.shape[0]
        
        return y, reconsError
    
    def get_output_shape_for(self,input_shape):
        return(self.batch_size, self.dict_size)
