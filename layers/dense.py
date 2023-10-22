import numpy as np
from ..utils.initializers import ActivationInitializer, WeightInitializer
from .base_layer import BaseLayer

class Dense(BaseLayer):
    def __init__(self, units, act_fn=None, init="glorot_uniform", optimizer=None):
        self.units = units
        self.init = init
        self.n_in = None
        self.n_out = units
        self.act_fn = ActivationInitializer(act_fn)()
        self.parameters = {"W": None, "b": None}
        self.is_initialized = False

    def _init_params(self):
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)

        b = np.zeros((1, self.n_out))
        W = init_weights((self.n_in, self.n_out))
        self.parameters = {"W": W, "b": b}
        self.derived_variables = {"Z": []}
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        self.is_initialized = True

    @property
    def hyperparameters(self):
        return {
            "layer": "Dense",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters
            }
        }

    def forward(self, X, retain_derived=True):
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()

        Y, Z = self._fwd(X)

        if retain_derived:
            self.X.append(X)
            self.derived_variables["Z"].append(Z)

        return Y

    def _fwd(self, X):
        W = self.parameters["W"]
        b = self.parameters["b"]
        Z = X @ W + b
        Y = self.act_fn(Z)
        return Y, Z

    def backward(self, dLdy, retain_grads=True):
        assert self.trainable, "Layer is frozen"
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        dX = []
        X = self.X
        for dy, x in zip(dLdy, X):
            dx, dw, db = self._bwd(dy, x)
            dX.append(dx)
            if retain_grads:
                self.gradients["W"] += dw
                self.gradients["b"] += db
        return dX[0] if len(X)==1 else dX

    def _bwd(self, dLdy, X):
        W = self.parameters["W"]
        b = self.parameters["b"]
        Z = X @ W + b
        dZ = dLdy * self.act_fn.grad(Z)
        dX = dZ @ W.T
        dW = X.T @ dZ
        dB = dZ.sum(axis=0, keepdims=True)
        return dX, dW, dB

    def _bwd2(self, dLdy, X, dLdy_bwd):
        W = self.parameters["W"]
        b = self.parameters["b"]

        dZ = self.act_fn.grad(X @ W + b)
        ddZ = self.act_fn.grad2(X @ W + b)
        ddX = dLdy @ W * dZ
        ddW = dLdy.T @ (dLdy_bwd * dZ)
        ddB = np.sum(dLdy @ W * dLdy_bwd * ddZ, axis=0, keepdims=True)
        return ddX, ddW, ddB
