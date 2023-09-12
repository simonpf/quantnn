"""
quantnn.packed_tensor
=====================

Provides an implementation of a packed tensor class that can be used
to represent sparse batches.
"""
from typing import Union

import torch


class PackedTensor:
    """
    Special Tensor that to represent a sparse batch.

    The class adds two additional attributes: ``batch_size`` and
    ``batch_indices`` to the tensor. ``batch_size`` holds the original
    size of the batch, while ``batch_indices`` holds the batch indices
    that the data stored in the associated tensor corresponds to.
    """

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Unwraps packed tensors in torch function call.
        """
        if kwargs is None:
            kwargs = {}

        def get_attribute(name, arg):
            if isinstance(arg, list) or isinstance(arg, tuple):
                batch_sizes = sum([get_attribute(name, nested) for nested in arg], [])
                return [bs for bs in batch_sizes if not bs is None]
            return [getattr(arg, name, None)]

        batch_sizes = get_attribute("batch_size", args)
        batch_indices = get_attribute("batch_indices", args)

        def replace_args(arg):
            if hasattr(arg, "_t"):
                return arg._t
            if isinstance(arg, list):
                return [replace_args(nested) for nested in arg]
            if isinstance(arg, tuple):
                return tuple([replace_args(nested) for nested in arg])
            return arg

        args = list(map(replace_args, args))
        if len(batch_indices) > 1:

            def is_same(bi):
                return batch_indices[0] == bi

            if not all(map(is_same, batch_indices[1:])):
                raise ValueError(
                    "Native torch operations can only be applied to packed"
                    " tensors with the same batch indices."
                )

        ret = func(*args, **kwargs)
        return PackedTensor(ret, batch_sizes[0], batch_indices[0])

    @classmethod
    def stack(cls, tensors):
        """
        Stack a list of tensors into a PackedTensor.

        Args:
            tensors: A list containing dense tensors corresponding to valid
                samples interleaved with ``None`` where no valid batch is
                available.

        Return:
            A ``PackedTensor`` representation of samples given in ``tensors``
            stacked into a batch.
        """
        batch_size = len(tensors)
        batch_indices = []
        parts = []
        for index, tensor in enumerate(tensors):
            if isinstance(tensor, torch.Tensor):
                parts.append(tensor)
                batch_indices.append(index)
            elif tensor is not None:
                raise ValueError(
                    "List of samples may only contain ``torch.Tensor`` objects"
                    " or ``None``. Found '%s'",
                    type(tensor),
                )
        if len(parts) == 0:
            data = torch.ones((0, 1, 1, 1))
        else:
            data = torch.stack(parts)
        return PackedTensor(data, batch_size, batch_indices)

    def __init__(self, x, batch_size, batch_indices):
        """
        Initializes a new packed tensor.

        Args:
            x: The packed tensor data.
            batch_size: The original size of the batch.
            batch_indices: List of indices to which the elements
                in ``x`` correspond.
        """
        self._t = torch.as_tensor(x)
        self._batch_size = torch.as_tensor(batch_size)

        try:
            batch_indices = [ind.item() for ind in batch_indices]
        except AttributeError:
            pass
        self._batch_indices = list(batch_indices)

    @property
    def batch_size(self):
        """The full batch size."""
        return self._batch_size

    @property
    def batch_indices(self):
        """The batch indices."""
        return self._batch_indices

    @property
    def shape(self):
        """
        The shape of the data tensor.

        NOTE: This is not the shape of the full tensor.
        """
        return self._t.shape

    @property
    def empty(self):
        """
        Checks whether batch contains data.
        """
        return len(self._batch_indices) == 0

    @property
    def not_empty(self):
        """
        Checks whether batch contains data.
        """
        return len(self._batch_indices) > 0

    @property
    def tensor(self):
        """
        The 'torch.Tensor' containing the data of the packed tensor.
        """
        return self._t

    @property
    def ndim(self):
        """
        The rank of the tensor
        """
        return self._t.ndim

    def dim(self):
        return self._t.dim()

    def __repr__(self):
        s = (
            f"PackedTensor(batch_size={self.batch_size}, "
            f"batch_indices={self.batch_indices})"
        )
        return s

    def __getitem__(self, slices):
        """
        Get slice from data tensor.

        Only allowed if no slicing is performed along first axis.
        """
        sl = slices[0]
        if sl == Ellipsis:
            return PackedTensor(
                self._t.__getitem__(slices),
                self.batch_size,
                self.batch_indices
            )
        if isinstance(sl, slice):
            if sl.start not in [0, None] or sl.stop is not None:
                raise ValueError(
                    "Slicing of PackedTensors only possible if no slicing"
                    "is performed along the first axis."
                )
            return PackedTensor(
                self._t.__getitem__(slices),
                self.batch_size,
                self.batch_indices
            )
        raise ValueError(
            "Slicing of PackedTensors only possible if no slicing"
            "is performed along the first axis."
        )

    def __setitem__(self, slices, data):
        """
        Set slice of data tensor.

        Only allowed if no slicing is performed along first axis.
        """
        sl = slices[0]
        if sl == Ellipsis or isinstance(sl, slice):
            if sl.start not in [0, None] or sl.stop is not None:
                raise ValueError(
                    "Slicing of PackedTesnors only possible if no slicing"
                    "is performed along the first axis."
                )
            return self._t.__setitem__(slices, data)
        raise ValueError(
            "Slicing of PackedTensors only possible if no slicing"
            "is performed along the first axis."
        )

    def expand(self):
        """
        Expand tensor to full size using 0-padding.

        Return:
            A new tensor expanded to the full batch size by filling
            missing
        """
        if len(self.batch_indices) == 0:
            shape = [self.batch_size] + list(self._t.shape[1:])
            full = torch.zeros(tuple(shape)).type_as(self._t)
            full.__class__ = torch.Tensor
            return full
        shape = list(self._t.shape)
        shape[0] = self.batch_size
        full = []
        running_ind = 0
        for i in range(self.batch_size):
            if i in self.batch_indices:
                full.append(torch.Tensor(self._t[running_ind]))
                running_ind += 1
            else:
                full.append(torch.zeros_like(self._t[0]))
        full = torch.stack(full, 0)
        return full

    def intersection(self, other):
        """
        Extract samples that are present in both of two tensors.

        Args:
            other: The other tensor.

        Return:
            A tuple ``(parts_self, parts_other)`` containing a pair
            of packed tensors each containing the samples corresponding
            to batch indices that are present in both ``self`` and
            ``other``.
            If the intersection is empty, ``None, None`` is returned.
        """
        if isinstance(other, PackedTensor):
            other_indices = other.batch_indices

            def get_slice(i):
                return other._t[i]

        else:
            other_indices = list(range(self.batch_size))

            def get_slice(i):
                return other[i]

        indices = sorted(list(set(self.batch_indices) & set(other_indices)))

        parts_self = []
        for i, index in enumerate(self.batch_indices):
            if index in indices:
                parts_self.append(self._t[i])

        parts_other = []
        for i, index in enumerate(other_indices):
            if index in indices:
                parts_other.append(get_slice(i))

        if not parts_self:
            return None, None

        parts_self = torch.stack(parts_self, 0)
        parts_other = torch.stack(parts_other, 0)

        parts_self = PackedTensor(parts_self, self.batch_size, indices)
        parts_other = PackedTensor(parts_other, self.batch_size, indices)
        return parts_self, parts_other

    def difference(self, other):
        """
        Extract samples that are present in only one of two tensors.

        Args:
            other: The other tensor.

        Return:
            A packed tensor containing the samples corresponding to batch
            indices that are present in either ``self`` or ``other`` but not
            both.
            If the intersection is empty, ``None, None`` is returned.
        """
        if isinstance(other, PackedTensor):
            other_indices = other.batch_indices

            def get_slice(i):
                return other._t[i]

        else:
            other_indices = range(self.batch_size)

            def get_slice(i):
                return other[i]

        indices = sorted(list(set(self.batch_indices) ^ set(other_indices)))

        if len(self.batch_indices) > 0:
            pattern = self._t[0]
        else:
            pattern = get_slice(0)

        parts = []
        running_ind_self = 0
        running_ind_other = 0
        for i in range(self.batch_size):
            if i in indices:
                if i in self.batch_indices:
                    parts.append(torch.as_tensor(self._t[running_ind_self]))
                elif i in other.batch_indices:
                    parts.append(torch.as_tensor(get_slice(running_ind_other)))

            running_ind_self += i in self.batch_indices
            running_ind_other += i in other.batch_indices

        if len(parts) == 0:
            return None

        return PackedTensor(torch.stack(parts), self.batch_size, indices)

    def split_parts(self, other):
        """
        Split tensors up into sample that are present only in 'self', samples
        only present in 'other' and samples present in both tensors.

        Args:
            other: The other tensor.

        Return:
            A tuple ``(x_1_only, x_2_only, x_1_both, x_2_both)`` containing
            packed tensors that contain:
                - x_1_only: The samples that are present only in x_1.
                - x_2_only: The samples that are present only in x_2.
                - x_1_both: The samples of x_1 that are present in both.
                - x_2_both: The samples of x_2 that are present in both.

        """
        if not isinstance(other, PackedTensor):
            batch_size = other.shape[0]
            other = PackedTensor(other, batch_size, range(batch_size))

        inds_both = sorted(list(
            set(self.batch_indices) & set(other.batch_indices)
        ))
        inds_x1 = self.batch_indices
        inds_x2 = other.batch_indices

        x1_only = []
        x2_only = []
        x1_both = []
        x2_both = []

        running_ind_self = 0
        running_ind_other = 0
        for i in range(self.batch_size):
            if i in inds_both:
                x1_both.append(running_ind_self)
                running_ind_self += 1
                x2_both.append(running_ind_other)
                running_ind_other += 1
            elif i in inds_x1:
                x1_only.append(running_ind_self)
                running_ind_self += 1
            elif i in inds_x2:
                x2_only.append(running_ind_other)
                running_ind_other += 1

        if len(x1_only) == 0:
            x1_only = None
        else:
            binds = [self.batch_indices[ind] for ind in x1_only]
            x1_only = PackedTensor(self._t[x1_only], self.batch_size, binds)

        if len(x2_only) == 0:
            x2_only = None
        else:
            binds = [other.batch_indices[ind] for ind in x2_only]
            x2_only = PackedTensor(other._t[x2_only], other.batch_size, binds)

        if len(inds_both) == 0:
            x1_both = None
            x2_both = None
        else:
            x1_both = PackedTensor(self._t[x1_both], self.batch_size, inds_both)
            x2_both = PackedTensor(other._t[x2_both], other.batch_size, inds_both)

        return(
            x1_only,
            x2_only,
            x1_both,
            x2_both
        )


    def union(
            self,
            other: Union[torch.Tensor, "PackedTensor"]
    ) -> "PackedTensor":
        """
        Combine two packed sensors.

        Args:
            other: The other tensor.

        Return:
            A packed tensor containing combined samples from all
            tensors. If a sample is present in both tensors, the
            sample from ``self`` has priority.
        """
        if isinstance(other, PackedTensor):
            other_indices = other.batch_indices

            def get_slice(i):
                return other._t[i]

        else:
            other_indices = range(self.batch_size)

            def get_slice(i):
                return other[i]

        indices = sorted(list(set(self.batch_indices) | set(other_indices)))

        try:
            if len(self.batch_indices) > 0:
                pattern = self._t[0]
            else:
                pattern = get_slice(0)
        except IndexError:
            return None

        parts = []
        running_ind_self = 0
        running_ind_other = 0
        for i in range(self.batch_size):
            if i in indices:
                if i in self.batch_indices:
                    parts.append(torch.as_tensor(self._t[running_ind_self]))
                elif i in other.batch_indices:
                    parts.append(torch.as_tensor(get_slice(running_ind_other)))

            running_ind_self += i in self.batch_indices
            running_ind_other += i in other.batch_indices

        if len(parts) == 0:
            data = np.ones((0, 1, 1, 1))
        else:
            data = torch.stack(parts)
        return PackedTensor(data, self.batch_size, indices)

    def to(self, *args, **kwargs):
        return PackedTensor(
            self._t.to(*args, **kwargs), self.batch_size, self.batch_indices
        )


def forward(module, x, **kwargs):
    """
    Propagates a potentially input tensor through a module.

    Ensures that a PackedTensor is returned when a PackedTensor is provided
    as input so that the available samples are kept track of.

    Args:
        module: The torch.nn.Module through which to propagate the tensor.
        x: The tensor to propagate through the module,
        kwargs: Kwargs passed on to the forward function of the module.

    Return:
        A PackedTensor (if the input is a PackedTensor or a list of packed
        tensors) or a standard tensor (if the input is a standard tensor)
        containing the module applied to the given input tensor.
    """
    if isinstance(x, PackedTensor):
        if x.empty:
            return x
        y = module(x.tensor, **kwargs)
        if isinstance(y, list):
            return [
                PackedTensor(y_i, x.batch_size, x.batch_indices)
                for y_i in y
            ]
        return PackedTensor(
            y,
            x.batch_size,
            x.batch_indices
        )
    return module(x, **kwargs)
