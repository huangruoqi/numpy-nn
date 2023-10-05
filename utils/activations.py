import numpy as np
from math import erf
from abc import ABC, abstractmethod

class ActivationBase(ABC):
    def __init__(self, **kwargs):
        """Initialize the ActivationBase object"""
        super().__init__()

    def __call__(self, z: np.ndarray):
        """Apply the activation function to an input"""
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.fn(z)

    @abstractmethod
    def fn(self, z: np.ndarray):
        """Apply the activation function to an input"""
        raise NotImplementedError

    @abstractmethod
    def grad(self, x: np.ndarray, **kwargs):
        """Compute the gradient of the activation function wrt the input"""
        raise NotImplementedError

class Sigmoid(ActivationBase):
    def __str__(self):
        return "Sigmoid"

    def fn(self, z: np.ndarray):
        return 1 / (1 + np.exp(-z))
    
    def grad(self, x: np.ndarray):
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x)

    def grad2(self, x: np.ndarray):
        fn_x = self.fn(x)
        return (1 - 2 * fn_x) * (fn_x * (1 - fn_x))

class ReLU(ActivationBase):
    def __str__(self):
        return "ReLU"

    def fn(self, z: np.ndarray):
        return np.clip(z, 0, np.inf)
    
    def grad(self, x: np.ndarray):
        return (x > 0).astype('i4')

    def grad2(self, x: np.ndarray):
        return np.zeros_like(x)

class LeakyReLU(ActivationBase):
    def __init__(self, alpha):
        self.alpha = alpha
        super(ActivationBase, self).__init__()

    def __str__(self):
        return f"LeakyReLU(alpha={self.alpha})"

    def fn(self, z: np.ndarray):
        _z = z.copy()
        _z[z < 0] = _z[z < 0] * self.alpha
    
    def grad(self, x: np.ndarray):
        out = np.ones_like(x)
        out[x < 0] = self.alpha
        return out

    def grad2(self, x: np.ndarray):
        return np.zeros_like(x)

class GELU(ActivationBase):
    def __init__(self, approximate=True):
        self.approximate = approximate
        super(ActivationBase, self).__init__()

    def __str__(self):
        return f"GELU(approximate={self.approximate})"

    def fn(self, z: np.ndarray):
        pi, sqrt, tanh = np.pi, np.sqrt, np.tanh
        if self.approximate:
            return 0.5 * z * (1 + tanh(sqrt(2 / pi) * (z + 0.044715 * z ** 3)))
        return 0.5 * z * (1 + erf(z / sqrt(2)))
    
    def grad(self, x: np.ndarray):
        pi, exp, sqrt, tanh = np.pi, np.exp, np.sqrt, np.tanh

        s = x / sqrt(2)
        erf_prime = lambda x: (2 / sqrt(pi)) * exp(-(x ** 2))

        if self.approximate:
            approx = tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3))
            dx = 0.5 + 0.5 * approx + ((0.5 * x * erf_prime(s)) / sqrt(2))
        else:
            dx = 0.5 + 0.5 * erf(s) + ((0.5 * x * erf_prime(s)) / sqrt(2))
        return dx


    def grad2(self, x: np.ndarray):
        pi, exp, sqrt = np.pi, np.exp, np.sqrt
        s = x / sqrt(2)

        erf_prime = lambda x: (2 / sqrt(pi)) * exp(-(x ** 2))
        erf_prime2 = lambda x: -4 * x * exp(-(x ** 2)) / sqrt(pi)
        ddx = (1 / 2 * sqrt(2)) * (1 + erf_prime(s) + (erf_prime2(s) / sqrt(2)))

class Tanh(ActivationBase):
    def __str__(self):
        return f"Tanh"

    def fn(self, z: np.ndarray):
        return np.tanh(z)
    
    def grad(self, x: np.ndarray):
        return 1 - np.tanh(x) ** 2

    def grad2(self, x: np.ndarray):
        tanh_x = np.tanh(x)
        return 2 * tanh_x * (1 - tanh_x ** 2)