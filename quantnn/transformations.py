import numpy as np

from quantnn.backends import get_tensor_backend

class Log10:
    """
    Transforms values to log space.
    """
    def __init__(self):
        self.xp = None

    def __call__(self, x):
        if self.xp is None:
            xp = get_tensor_backend(x)
            self.xp = xp
        else:
            xp = self.xp
        return xp.log(x) / np.log(10)

    def invert(self, y):
        if self.xp is None:
            xp = get_tensor_backend(y)
            self.xp = xp
        else:
            xp = self.xp
        return xp.exp(np.log(10) * y)

class Log:
    """
    Transforms values to log space.
    """
    def __init__(self):
        self.xp = None

    def __call__(self, x):
        if self.xp is None:
            xp = get_tensor_backend(x)
            self.xp = xp
        else:
            xp = self.xp
        return xp.log(x)

    def invert(self, y):
        if self.xp is None:
            xp = get_tensor_backend(y)
            self.xp = xp
        else:
            xp = self.xp
        return xp.exp(y)
