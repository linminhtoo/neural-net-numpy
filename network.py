from abc import ABC
from collections import OrderedDict
from typing import List, Union

import numpy as np

from act import BaseActivation
from layer import BaseLayer


class BaseNetwork(ABC):
    def __init__(self):
        '''
        define the variables to store state of network,
        and other variables as necessary
        '''
        self.net = OrderedDict()
        self.cache = OrderedDict()
        self.grads = OrderedDict()
        ...

    def build(self, layers: list):
        '''
        build network structure from list of layers
        store the structure as an OrderedDict in self.net
        '''
        ...

    def forward(self, x):
        '''
        implement forward pass sequentially through all layers in network
        store the intermediate activations in a dictionary
        '''
        ...

    def backward(self, dloss):
        '''
        implement backward pass using chain rule,
        given the gradient of loss function with respect to y_pred
        store the gradients at each layer & activation in a dictionary
        '''
        ...

    def update(self, learning_rate: float):
        '''
        update the learnable parameters in network,
        using the backpropagated gradients and learning rate
        '''


class MLP(BaseNetwork):
    '''
    defines a simple, sequential multi-layer perceptron

    an obvious limitation is that we only support strictly sequential operations
    to support a more general structure, eg multiple prediction heads from a backbone,
    we need to define a directed computation graph,
    that tells the input data & the gradients where to go next
    '''
    def __init__(self, layers: List[Union[BaseActivation, BaseLayer]]):
        super().__init__()

        self.build(layers)

    def build(self, layers: List[Union[BaseActivation, BaseLayer]]):
        self.net = OrderedDict()

        # a simple and straightforward sequential naming convention
        for layer_idx, layer in enumerate(layers):
            self.net[f'{str(layer)}_{layer_idx}'] = layer

    def forward(self, x: np.ndarray):
        '''
        args:
            x: expecting shape [batch, feature_dim]
        '''
        # tranpose input from [BSZ, D_in] to [D_in, BSZ]
        x_ = x.T

        for layer in self.net.values():
            x_ = layer.forward(x_)

        # tranpose output from [D_out, BSZ] to [BSZ, D_out]
        return x_.T

    def backward(self, dloss: float):
        '''
        args:
            dloss: gradient of loss function wrt model's output, y_pred
        '''
        # first gradient comes from loss function
        grad = dloss

        # run backward on every layer from last layer to input layer
        for layer in reversed(self.net.values()):
            grad = layer.backward(grad)

    def update(self, learning_rate: float):
        # the simplest update - SGD
        # of course, we can add more sophisticated optimizers - SGD with momentum is a start
        for layer in self.net.values():
            if layer.learnable:
                layer.update(learning_rate)
