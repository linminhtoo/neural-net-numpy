from abc import ABC

import numpy as np

class BaseActivation(ABC):
    def __init__(self):
        '''
        initialise parameters, if any
        e.g. for PReLU, we may initialise the learnable coefficient
        '''
        self.cache = {}

        # typically has no parameter to update
        # if True, must define a self.update() method
        self.learnable = False

    def __repr__(self):
        '''
        name of activation
        '''
        ...

    def forward(self, z):
        '''
        implement forward pass

        args:
            z: input matrix from previous layer, will be stored in self.cache
        returns:
            a: output matrix after applying activation
        '''
        ...

    def backward(self, da):
        '''
        implement backward pass with chain rule to calculate gradients

        args:
            da: gradient backpropagated from subsequent layer
            z: input matrix from previous layer, retrieved from self.cache
        returns:
            dz: gradient to backpropagate to previous layer
        '''
        ...

class ReLU(BaseActivation):
    '''
    the good ol' ReLU activation as we know it
    '''
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'ReLU'

    def forward(self, z):
        self.cache["z"] = z
        return np.maximum(z, 0)

    def backward(self, da):
        z = self.cache["z"]
        # preserve shape of da
        return da * (z >= 0).astype(float)

class Sigmoid(BaseActivation):
    '''
    the name says it all.
    handy to map values into the [0, 1] range
    '''
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'Sigmoid'

    def forward(self, z):
        self.cache["z"] = z
        return 1 / (1 + np.exp(-z))

    def backward(self, da):
        z = self.cache["z"]
        # preserve shape of da
        return da * (self.forward(z) * (1 - self.forward(z)))