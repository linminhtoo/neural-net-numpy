from abc import ABC

import numpy as np

def init_normal(W, mean=0, std=1):
    '''
    initialise linear weight matrix to follow a normal distribution
    '''
    assert W.ndim == 2
    W[:] = np.random.normal(mean, std, W.shape)

class BaseLayer(ABC):
    def __init__(self, in_dim, out_dim,
                 init_mode="normal"):
        '''
        create the matrices to store the layer's parameters
        and then initialise them
        '''
        self.cache = {}
        self.grads = {}

        self.learnable = True # whether this layer should be updated

    def __repr__(self):
        '''
        name of layer
        '''
        ...

    def forward(self, a):
        '''
        implement forward pass

        args:
            a: input matrix from previous layer, will be stored in self.cache
        returns:
            z: output matrix after applying linear layer
        '''
        ...

    def backward(self, dz):
        '''
        implement backward pass with chain rule to calculate gradients

        args:
            dz: gradient backpropagated from subsequent layer
            a: input matrix from previous layer, retrieved from self.cache
        returns:
            dW: gradient wrt weights, to update the weights parameter
            db: gradient wrt bias, to update the bias parameter
            da: gradient to backpropagate to previous layer
        '''
        ...

    def update(self, learning_rate):
        '''
        update weights of layer given a learning rate
        simple stochastic gradient descent
        '''
        ...

class Linear(BaseLayer):
    '''
    a simple linear layer with weight and bias
    '''
    def __init__(self, in_dim, out_dim,
                init_mode="normal"):
        super().__init__(in_dim, out_dim, init_mode)

        self.W = np.zeros((out_dim, in_dim))
        self.b = np.zeros((out_dim, 1))

        if init_mode == "normal":
            init_normal(self.W)
        else:
            raise ValueError(f'unrecognized init_mode: {init_mode}')

    def __repr__(self):
        return 'Linear'

    def forward(self, a):
        self.cache["a"] = a # [in_dim, bsz]
        return self.W @ a + self.b

    def backward(self, dz): # [out_dim, bsz]
        a = self.cache["a"] # [in_dim, bsz]

        # [out_dim, bsz] @ [bsz, in_dim]
        dW = (dz @ a.T) / a.shape[1] # divide by batch_size
        db = np.mean(dz, axis=1, keepdims=True) # average across batch_size dim
        da = self.W.T @ dz

        self.grads["dW"] = dW
        self.grads["db"] = db
        return da

    def update(self, learning_rate):
        self.W -= self.grads['dW'] * learning_rate
        self.b -= self.grads['db'] * learning_rate