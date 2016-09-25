import numpy as np
from sklearn.preprocessing import normalize
import scipy.misc
from keras import backend as K
from keras.engine.topology import Layer

class ArmLayer(Layer):
    def __init__(self, dict_size, weights = None, iteration = 10, threshold = 0.5, shared_weights=None, **kwargs):
        self.np_weights = weights
        self.shared_weights = shared_weights
        self.iteration = iteration
        self.threshold = threshold
        self.dict_size = dict_size
        super(ArmLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        nb_features = np.prod(input_shape[1:])

        if self.shared_weights is not None:
            print "Using provided shared weights"
            self.W = self.shared_weights
            self.np_weights = self.shared_weights.get_value()
        elif self.np_weights is not None:
            print "Using provided np weights"
            self.np_weights = np.float32(normalize(self.np_weights, axis=1))
            self.W = K.variable(self.np_weights, name='{}_W'.format(self.name))
        else:
            self.np_weights =  np.random.normal(size=[self.dict_size, nb_features])
            self.np_weights = np.float32(normalize(self.np_weights, axis=1))
            self.W = K.variable(self.np_weights, name='{}_W'.format(self.name))

        # set alpha
        eigvals = np.linalg.eigvals(self.np_weights.dot(self.np_weights.T))
        maxEigval = 1.5 * np.max(np.absolute(eigvals))
        self.alpha = np.float32(1/maxEigval)

        Wzero = np.float32(np.zeros(shape=[nb_features, self.dict_size]))
        self.Wzero = K.variable(Wzero, name='{}_Wzero'.format(self.name))
        
        self.trainable_weights = [self.W]


    def armderiv(self,x, y):
        hard_thresholding = False
        linout = y - self.alpha * K.dot(K.dot(y,self.W) - x,self.W.T)
        if hard_thresholding:
            out = K.greater(K.abs(linout),self.threshold) * linout
        else:
            out = K.sign(linout) * K.maximum(K.abs(linout) - self.threshold,0)
        return out

    def arm(self, x, iteration):
        if iteration==0:
            outApprox = K.dot(x, self.Wzero)
        else:
            outApprox = self.arm(x, iteration-1)
        return self.armderiv(x, outApprox)

    def call(self, x, mask=None):
        # flatten the images to arrays
        x_flattened = K.reshape(x,[K.shape(x)[0],K.prod(K.shape(x)[1:])])
        
        y = self.arm(x_flattened, self.iteration)        
        return y
    
    def get_output_shape_for(self,input_shape):
        return(input_shape[0], self.dict_size)
