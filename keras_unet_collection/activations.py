
from tensorflow import math
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


def gelu_(X):
    return 0.5*X*(1.0 + math.tanh(0.7978845608028654*(X + 0.044715*math.pow(X, 3))))

def snake_(X, beta):
    return X + (1/beta)*math.square(math.sin(beta*X))

def esh_(X):
    return X * K.tanh(K.sigmoid(X))

class Swish(Layer):
    '''
    Swish Activation Function.
    
    Swish is a smooth, self-gated activation function discovered by researchers at Google.
    
    Y = Swish()(X)
    
    Ramachandran, P., Zoph, B. and Le, Q.V., 2017. Searching for activation functions. arXiv preprint arXiv:1710.05941.
    
    Usage: use it as a tf.keras.Layer
    '''
    def __init__(self, trainable=False, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.trainable = trainable

    def build(self, input_shape):
        super(Swish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return self.swish_(inputs)

    def get_config(self):
        config = {'trainable': self.trainable}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    @staticmethod
    def swish_(inputs):
        return inputs * K.sigmoid(inputs)

class Mish(Layer):
    '''
    Mish Activation Function.
    
    Mish is a smooth, non-monotonic function that can improve the 
    performance of deep learning models.
    
    Y = Mish()(X)
    
    Mish: A Self Regularized Non-Monotonic Activation Function
    Diganta Misra
    
    Usage: use it as a tf.keras.Layer
    '''
    def __init__(self, trainable=False, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True
        self.trainable = trainable

    def build(self, input_shape):
        super(Mish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return self.mish_(inputs)

    def get_config(self):
        config = {'trainable': self.trainable}
        base_config = super(Mish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    @staticmethod
    def mish_(inputs):
        return inputs * K.tanh(K.softplus(inputs))


class Esh(Layer):
    '''
    Esh Activation Function.
    '''
    def __init__(self, trainable=False, **kwargs):
        super(Esh, self).__init__(**kwargs)
        self.supports_masking = True
        self.trainable = trainable

    def build(self, input_shape):
        super(Esh, self).build(input_shape)

    def call(self, inputs, mask=None):
        return esh_(inputs)

    def get_config(self):
        config = {'trainable': self.trainable}
        base_config = super(Esh, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class GELU(Layer):
    '''
    Gaussian Error Linear Unit (GELU), an alternative of ReLU
    
    Y = GELU()(X)
    
    ----------
    Hendrycks, D. and Gimpel, K., 2016. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415.
    
    Usage: use it as a tf.keras.Layer
    
    
    '''
    def __init__(self, trainable=False, **kwargs):
        super(GELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.trainable = trainable

    def build(self, input_shape):
        super(GELU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return gelu_(inputs)

    def get_config(self):
        config = {'trainable': self.trainable}
        base_config = super(GELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        return input_shape

    
class Snake(Layer):
    '''
    Snake activation function $X + (1/b)*sin^2(b*X)$. Proposed to learn periodic targets.
    
    Y = Snake(beta=0.5, trainable=False)(X)
    
    ----------
    Ziyin, L., Hartwig, T. and Ueda, M., 2020. Neural networks fail to learn periodic functions 
    and how to fix it. arXiv preprint arXiv:2006.08195.
    
    '''
    def __init__(self, beta=0.5, trainable=False, **kwargs):
        super(Snake, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        self.beta_factor = K.variable(self.beta, dtype=K.floatx(), name='beta_factor')
        if self.trainable:
            self._trainable_weights.append(self.beta_factor)

        super(Snake, self).build(input_shape)

    def call(self, inputs, mask=None):
        return snake_(inputs, self.beta_factor)

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta, 'trainable': self.trainable}
        base_config = super(Snake, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    