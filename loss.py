from abc import ABC

import numpy as np

class BaseLoss(ABC):
    def __init__(self, reduction="mean"):
        '''
        initialise the loss,
        e.g. reduction method, and other settings as necessary
        '''
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'invalid reduction: {reduction}')
        self.reduction = reduction

        self.y_pred = None
        self.y_true = None
        ...

    def __repr__(self):
        '''
        name of loss
        '''
        ...

    def forward(self, y_pred, y_true):
        '''
        calculates loss value

        args:
            y_pred: output from model, to store as self.y_pred
            y_true: label, to store as self.y_true
            expecting shape [BSZ, d_out] for both y_pred and y_true
        returns:
            loss: the reduced loss value
        '''
        ...

    def backward(self):
        '''
        calculates gradient of loss wrt y_pred

        args:
            y_pred: output from model, retrieved from self.y_pred
            y_true: label, retrieved from self.y_true
        returns:
            dloss: gradient to backpropagate to model,
            of shape [d_out, BSZ]
        '''
        ...

class MSE(BaseLoss):
    '''
    Mean Squared Error loss
    '''
    def __init__(self, reduction="mean"):
        super().__init__(reduction)

    def __repr__(self):
        return 'MSE'

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        y_mse = (y_pred - y_true) ** 2

        if self.reduction == 'mean':
            return y_mse.mean()
        elif self.reduction == 'sum':
            return y_mse.sum()

    def backward(self):
        dloss = 2 * (self.y_pred - self.y_true)
        return dloss.T # [BSZ, d_out] --> [d_out, BSZ]

class BCE(BaseLoss):
    '''
    Binary Cross Entropy loss
    '''
    def __init__(self, eps=1e-8, reduction="mean"):
        super().__init__(reduction)
        self.eps = eps

    def __repr__(self):
        return 'BCE'

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        y_bce = -(y_true * np.log(y_pred + self.eps) + \
                  (1 - y_true) * np.log(1 - y_pred + self.eps))

        if self.reduction == 'mean':
            return y_bce.mean()
        elif self.reduction == 'sum':
            return y_bce.sum()

    def backward(self):
        dloss = -((self.y_true / (self.y_pred + self.eps)) - \
                  ((1 - self.y_true) / (1 - self.y_pred + self.eps)))
        return dloss.T # [BSZ, d_out] --> [d_out, BSZ]