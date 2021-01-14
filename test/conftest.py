"""
Contains fixtures that are automatically available in all test files.
"""
import pytest
import numpy

BACKENDS = [numpy]

#try:
#    import tensorflow as tf
#    BACKENDS.append(tf)
#except ModuleNotFoundError:
#    pass

try:
    import torch
    BACKENDS += [torch]
except ModuleNotFoundError:
    pass

try:
    import jax.numpy as jnp
    BACKENDS.append(jnp)
except ModuleNotFoundError:
    pass

def pytest_configure():
    pytest.backends = BACKENDS

