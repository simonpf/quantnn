"""
quantnn.normalizer
==================

This module provides a simple normalizer class, which can be used
to normalize input data and store the normalization data.
"""
import numpy as np

class Normalizer:
    """
   

    """
    def __init__(self, x):
        x = x.astype(np.float64)
        self.means = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)

    def __call__(self, x):
        return (x.astype(np.float64) - self.means) / self.std
