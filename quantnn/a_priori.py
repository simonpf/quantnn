"""
================
quantnn.a_priori
================

Defines classes to represent a priori distributions.
"""
from quantnn.generic import (
    get_array_module,
    expand_dims,
    as_type,
    concatenate,
    tensordot,
    exp,
)


class LookupTable:
    """
    Tensor formulation of a simple piece-wise linear lookup
    table.

    The a priori here is described as univariate function represented
    by its values at a sequence of nodes and the corresponding probabilities.
    """

    def __init__(self, x, y):
        """
        Create a new lookup table a priori instance.

        Args:
            x: The x values at which the value of the a priori is known.
            y: The corresponding non-normalized density.
        """
        self.x = x
        self.y = y

    def __call__(self, x, dist_axis=1):
        """
        Evaluate the a priori.

        Args:
            x: Tensor containing the values at which to evaluate the a priori.
            dist_axis: The axis along which the tensor x is sorted.

        Returns;
            Tensor with the same size as 'x' containing the values of the a priori
            at 'x' obtained by linear interpolation.
        """
        if len(x.shape) == 1:
            dist_axis = 0
        xp = get_array_module(x)
        n_dims = len(x.shape)

        n = x.shape[dist_axis]
        x_index = [slice(0, None)] * n_dims
        x_index[dist_axis] = 0

        selection_l = [slice(0, None)] * n_dims
        selection_l[dist_axis] = slice(0, -1)
        selection_l = tuple(selection_l)
        selection_r = [slice(0, None)] * n_dims
        selection_r[dist_axis] = slice(1, None)
        selection_r = tuple(selection_r)

        r_shape = [1] * n_dims
        r_shape[dist_axis] = -1
        r_x = self.x.reshape(r_shape)
        r_y = self.y.reshape(r_shape)

        r_x_l = r_x[selection_l]
        r_x_r = r_x[selection_r]
        r_y_l = r_y[selection_l]
        r_y_r = r_y[selection_r]

        rs = []

        for i in range(0, n):
            x_index[dist_axis] = slice(i, i + 1)
            index = tuple(x_index)
            x_i = x[index]

            mask = as_type(xp, (r_x_l < x_i) * (r_x_r >= x_i), x_i)
            r = r_y_l * (r_x_r - x_i) * mask
            r += r_y_r * (x_i - r_x_l) * mask
            r /= mask * (r_x_r - r_x_l) + (1.0 - mask)
            r = expand_dims(xp, r.sum(dist_axis), dist_axis)
            rs.append(r)

        r = concatenate(xp, rs, dist_axis)
        return r


class Gaussian:
    def __init__(self, x_a, s, dist_axis=-1):
        self.x_a = x_a
        self.s = s
        self.dist_axis = -1

    def __call__(self, x, dist_axis=1):
        xp = get_array_module(x)
        n_dims = len(x.shape)
        shape = [1] * n_dims
        shape[self.dist_axis] = -1
        x_a = self.x_a.reshape(shape)

        dx = x - x_a

        sdx = tensordot(xp, dx, self.s, ((self.dist_axis,), (-1,)))
        l = -0.5 * (dx * sdx).sum(self.dist_axis)

        return exp(xp, l)
