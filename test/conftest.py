"""
Contains fixtures that are automatically available in all test files.
"""
import pytest
import numpy

BACKENDS = [numpy]

try:
    import torch
    BACKENDS += [torch]
except:
    pass

def pytest_configure():
    pytest.backends = BACKENDS

