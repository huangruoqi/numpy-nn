import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional
from ..utils.initializers import (
    WeightInitializer,
    OptimizerInitializer,
    ActivationInitializer,
)
from ..utils.wrappers import init_wrappers, Dropout


class BaseLayer(ABC):
    def __init__(self, optimizer: Optional[Union[str, dict]]=None):
        self.X = []
        self.act_fn = None
        self.trainable = True
        self.optimizer = OptimizerInitializer(optimizer)()
        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {}
        super(BaseLayer, self).__init__()

    @abstractmethod
    def _init_params(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def foward(self, **kwargs):
        """Perform a forward pass through the layer"""
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, **kwargs):
        """Perform a backward pass through the layer"""
        raise NotImplementedError

    def freeze(self):
        self.trainable = False

    def unfreeze(self):
        self.trainable = True

    def flush_graidents(self):
        """Erase all the layer's derived vairables and gradients"""
        assert self.trainable, "Layer is frozen"
        self.X = []
        for k in self.derived_variables:
            self.derived_variables[k] = []

        for k, v in self.gradients.items():
            self.gradients[k] = np.zeros_like(v)

    def update(self, cur_loss: Optional[float]=None):
        """
        Update the layer parameters using the accrued gradients and layer 
        optimizer. Flush all gradients once the update is complete.
        """
        assert self.trainable, "Layer is frozen"
        self.optimizer.step()
        for k, v in self.gradients.items():
            if k in self.parameters:
                self.parameters[k] = self.optimizer(self.parameters[k], v, k, cur_loss)
        self.flush_graidents()

    def set_params(self, summary_dict: dict):
        layer, sd = self, summary_dict
        flatten_keys = ["parameters", "hyperparameters"]
        for k in flatten_keys:
            if k in sd:
                entry = sd[k]
                sd.update(entry)
                del sd[k]
        for k, v in sd.items():
            if k in self.parameters:
                layer.parameters[k] = k
            if k in self.hyperparameters:
                if k == "act_fn":
                    layer.act_fn = ActivationInitializer(v)()
                elif k == "optimizer":
                    layer.optimizer = OptimizerInitializer(sd[k])()
                elif k == "wrappers":
                    layer = init_wrappers(layer, sd[k])
                else:
                    setattr(layer, k, v)

    def summary(self):
        return {
            "layer": self.hyperparameters["layer"],
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters
        }
