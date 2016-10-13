from keras.regularizers import Regularizer
from keras import backend as K

class reconsRegularizer(Regularizer):

    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.uses_learning_phase = True
        self.layer = None

    def set_layer(self,layer):
        if self.layer is not None:
            raise Exception('Regularizers cannot be reused')
        self.layer = layer

    def __call__(self,loss):
        if self.layer is None:
            raise Exception("Need to call 'set_layer' on reconsRegularizer first")
        regularized_loss = loss
        x = self.layer.input
        y = self.layer.output
        recons = K.dot(y,self.layer.W)
        if self.l1:
            regularized_loss += K.mean(self.l1 * K.abs(x-recons))
        if self.l2:
            regularized_loss += K.mean(self.l2 * K.square(x-recons))
        return K.in_train_phase(regularized_loss, loss)

