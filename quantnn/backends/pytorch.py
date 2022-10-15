from quantnn.backends.tensor import TensorBackend


class PyTorch(TensorBackend):
    """
    TensorBackend implementation using torch tensors.
    """

    @classmethod
    def available(cls):
        try:
            import torch
        except ImportError:
            return False
        return True

    @classmethod
    def matches_tensor(cls, tensor):
        import torch

        return isinstance(tensor, torch.Tensor)

    @classmethod
    def from_numpy(cls, array, like=None):
        import torch

        tensor = torch.from_numpy(array)
        if like is not None:
            tensor = tensor.type(like.dtype).to(like.device)
        return tensor

    @classmethod
    def to_numpy(cls, array):
        return array.cpu().detach().numpy()

    @classmethod
    def as_type(cls, tensor, like):
        return tensor.type_as(like)

    @classmethod
    def sample_uniform(cls, shape=None, like=None):
        import torch

        if shape is None and like is None:
            raise ValueError(
                "'sample_uniform' requires at least one of the arguments "
                "'shape' and 'like'. "
            )
        dtype = None
        device = None
        if like is not None:
            dtype = like.dtype
            device = like.device
            if shape is None:
                shape = like.shape

        return torch.rand(shape, dtype=dtype, device=device)

    @classmethod
    def sample_gaussian(cls, shape=None, like=None):
        import torch

        if shape is None and like is None:
            raise ValueError(
                "'sample_uniform' requires at least one of the arguments "
                "'shape' and 'like'. "
            )
        dtype = None
        device = None
        if like is not None:
            dtype = like.dtype
            device = like.device
            if shape is None:
                shape = like.shape

        return torch.normal(0, 1, shape, dtype=dtype, device=device)

    @classmethod
    def size(cls, tensor):
        return tensor.numel()

    @classmethod
    def concatenate(cls, tensors, dimension):
        import torch

        return torch.cat(tensors, dimension)

    @classmethod
    def expand_dims(cls, tensor, dimension_index):
        return tensor.unsqueeze(dimension_index)

    @classmethod
    def exp(cls, tensor):
        return tensor.exp()

    @classmethod
    def log(cls, tensor):
        return tensor.log()

    @classmethod
    def pad_zeros(cls, tensor, n, dimension_index):
        import torch

        n_dims = len(tensor.shape)
        dimension_index = dimension_index % n_dims
        pad = [0] * 2 * n_dims
        pad[2 * n_dims - 2 - 2 * dimension_index] = n
        pad[2 * n_dims - 1 - 2 * dimension_index] = n
        return torch.nn.functional.pad(tensor, pad, "constant", 0.0)

    @classmethod
    def pad_zeros_left(cls, tensor, n, dimension_index):
        import torch

        n_dims = len(tensor.shape)
        dimension_index = dimension_index % n_dims
        pad = [0] * 2 * n_dims
        pad[2 * n_dims - 2 - 2 * dimension_index] = n
        pad[2 * n_dims - 1 - 2 * dimension_index] = 0
        return torch.nn.functional.pad(tensor, pad, "constant", 0.0)

    @classmethod
    def arange(cls, start, end, step, like=None):
        import torch

        device = None
        dtype = torch.float32
        if like is not None:
            dtype = like.dtype
            device = like.device
        return torch.arange(start, end, step, dtype=dtype, device=device)

    @classmethod
    def reshape(cls, tensor, shape):
        return tensor.reshape(shape)

    @classmethod
    def trapz(cls, y, x, dimension):
        import torch

        return torch.trapz(y, x, dim=dimension)

    @classmethod
    def cumsum(cls, y, dimension):
        import torch

        return torch.cumsum(y, dimension)

    @classmethod
    def zeros(cls, shape=None, like=None):
        import torch

        if shape is None and like is None:
            raise ValueError(
                "'zeros' requires at least one of the arguments " "'shape' and 'like'. "
            )
        dtype = None
        device = None
        if like is not None:
            if shape is None:
                return torch.zeros_like(like)
            dtype = like.dtype
            device = like.device

        return torch.zeros(shape, device=device, dtype=dtype)

    @classmethod
    def ones(cls, shape=None, like=None):
        import torch

        if shape is None and like is None:
            raise ValueError(
                "'ones' requires at least one of the arguments " "'shape' and 'like'. "
            )
        dtype = None
        device = None
        if like is not None:
            if shape is None:
                return torch.ones_like(like)
            dtype = like.dtype
            device = like.dtype

        return torch.ones(shape, device=device, dtype=dtype)

    @classmethod
    def softmax(cls, x, axis=None):
        import torch

        return torch.nn.functional.softmax(x, dim=axis)

    @classmethod
    def where(cls, condition, x, y):
        import torch

        return torch.where(condition, x, y)
