import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        W1 = np.array(W1)
        W2 = np.array(W2)
        y_true = np.array(y_true)

        print(x.shape, W1.shape, W2.shape, y_true.shape)
        z1 = x @ W1.T + b1 
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2.T + b2
        y_hat = z2
        n = len(y_true)

        loss = np.round(np.mean((y_hat - y_true) ** 2) ,4)
        # dl/dz2 =
        dz2 = (2 / n * (z2 - y_true))
        # dL/dW2 = dL/dz2 * dz2/dW2
        dW2 = np.outer(dz2, a1)
        # dL/dB2 = dL/dz2 * dz2/db2
        db2 = dz2 # * 1
        # dL/dW1 = dL/dz2 * dz2/dA1 * dA1/dZ1 * dz1/DW1
        dz1 = dz2 @ W2 * (a1 > 0).astype('float')
        dW1 = np.outer(dz1, x)
        # dL/db1 = dL/z1 
        db1 = dz1

        return {
            'loss': loss,
            'dW1': np.round(dW1, 4),
            'db1': np.round(db1, 4),
            'dW2': np.round(dW2, 4),
            'db2': np.round(db2, 4)
        }