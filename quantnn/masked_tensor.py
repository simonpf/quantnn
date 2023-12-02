"""
quantnn.masked_tensor
=====================

Provides a masked tensor class that allows masking invalid elements.
"""
import functools

import numpy as np
import torch

HANDLED_FUNCTIONS = {}

MASK_HANDLERS = {
    torch.cat: torch.cat,
    torch.stack: torch.stack,
    torch.add: torch.logical_or,
    torch.Tensor.add: torch.logical_or,
    torch.Tensor.add_: torch.logical_or,
    torch.mul: torch.logical_or,
    torch.permute: torch.permute,
    torch.Tensor.reshape: torch.reshape,
    torch.Tensor.view: torch.Tensor.view,
    torch.sum: torch.any,
    torch.Tensor.sum: torch.any,
    torch.mean: torch.any,
    torch.Tensor.mean: torch.any,
    torch.unsqueeze: torch.unsqueeze,
    torch.Tensor.unsqueeze: torch.unsqueeze,
    torch.squeeze: torch.squeeze,
    torch.Tensor.squeeze: torch.squeeze,
    torch.gather: torch.gather,
}


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class MaskedTensor(torch.Tensor):
    """
    Extends the torch.Tensor class by adding a mask that identifies
    invalid elements. The masked tensor also provides functionality
    to compress the tensor along the batch axis to speed up
    calculations.
    """

    def __new__(cls, *args, **kwargs):
        mask = kwargs.pop("mask", None)
        compressed = kwargs.pop("compressed", None)
        tensor = super().__new__(cls, *args, **kwargs)

        # Keep reference to original tensor.
        if isinstance(args[0], MaskedTensor):
            tensor.base = args[0].base
        else:
            tensor.base = args[0]

        # Infer mask if not given.
        if mask is None:
            if isinstance(args[0], MaskedTensor):
                mask = args[0].mask
        if mask is None:
            tensor = args[0]
            mask = torch.zeros(tensor.shape, dtype=bool, device=tensor.device)
        tensor.mask = mask.detach().to(device=args[0].device)

        if compressed is None:
            if isinstance(args[0], MaskedTensor):
                compressed = args[0].compressed

        if isinstance(mask, MaskedTensor):
            mask = torch.tensor(mask)

        tensor.compressed = compressed
        return tensor

    # def __init__(self, *args, **kwargs):
    #    mask = kwargs.pop("mask", None)
    #    compressed = kwargs.pop("compressed", None)
    #    super().__init__()

    def strip(self):
        return self.base

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """ """
        if kwargs is None:
            kwargs = {}

        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **kwargs)

        if len(args) == 1 and len(kwargs) == 0:
            return func(args[0].base)

        if len(args) > 0 and isinstance(args[0], MaskedTensor):
            masked_args = [arg for arg in args[1:] if isinstance(arg, MaskedTensor)]
            masked_args += [arg for arg in kwargs.values() if isinstance(arg, MaskedTensor)]
            if len(masked_args) == 0:
                return func(args[0].base, *args[1:], **kwargs)

        raise Exception()
        return NotImplemented

    def compress(self):
        n_tot = self.shape[0]
        all_missing = self.mask.view((n_tot, -1)).all(dim=-1)
        valid = torch.nonzero(~all_missing)[:, 0]
        return MaskedTensor(self[valid], compressed=(n_tot, valid))

    def decompress(self, axis=0):
        if self.compressed is None:
            return self

        n_tot, valid = self.compressed
        full_shape = list(self.shape)
        full_shape[axis] = n_tot

        full = np.nan * torch.zeros(full_shape, dtype=self.dtype, device=self.device)
        full[valid] = self

        full_mask = torch.ones(full_shape, dtype=bool, device=self.device)
        full_mask[valid] = self.mask

        return MaskedTensor(full, mask=full_mask)

    def __getitem__(self, *args, **kwargs):
        """
        Slices tensor and corresponding mask.
        """
        return MaskedTensor(
            self.strip().__getitem__(*args, **kwargs),
            mask=self.mask.__getitem__(*args, **kwargs),
        )


def extract_mask(arg):
    if isinstance(arg, (tuple, list, set)):
        return [extract_mask(nested) for nested in arg]
    elif isinstance(arg, dict):
        return {key: extract_mask(nested) for key, nested in arg.items()}
    elif isinstance(arg, MaskedTensor):
        return arg.mask
    elif isinstance(arg, torch.Tensor):
        return torch.zeros_like(arg, dtype=bool, device=arg.device)
    elif isinstance(arg, float):
        return torch.zeros(1, dtype=bool)
    return arg


def strip_type(arg):
    if isinstance(arg, (tuple, list, set)):
        return [strip_type(nested) for nested in arg]
    elif isinstance(arg, dict):
        return {key: strip_type(nested) for key, nested in arg.items()}
    elif isinstance(arg, MaskedTensor):
        return torch.Tensor(arg)
    return arg


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


def get_base(tensor):
    if isinstance(tensor, MaskedTensor):
        return tensor.base
    return tensor


def get_mask(tensor):
    """
    Get maks from a tensor.

    Generic function to retrieve a mask identifying invalid elements in
    a standard tensor or a masked tensor.

    Args:
        tensor: If this is a MaskedTensor, its mask is returned. If it is
            a standard torch.Tensor a mask identifying NAN elements is
            returned.
    """
    if isinstance(tensor, MaskedTensor):
        return tensor.mask
    elif isinstance(tensor, torch.Tensor):
        return torch.zeros(tensor.shape, device=tensor.device, dtype=bool)
    torch.zeros(1, dtype=bool)


def get_compressed(tensor):
    """
    Extract compressed attribute from tensor.

    Args:
        tensor: A standard torch.Tensor or a MaskedTensor object.

    Return:
        If 'tensor' is a MaskedTensor, its 'compressed' attribute is returned.
        If it is not a MaskedTensor, None is returned.

    """
    if isinstance(tensor, MaskedTensor):
        return tensor.compressed
    return None


@implements(torch.cat)
def cat(tensors, dim=0, out=None):
    """
    Concatenate tensors and their masks.
    """
    compressed = [get_compressed(tensor) for tensor in tensors]
    if out is None:
        res = torch.cat([get_base(t) for t in tensors], dim=dim)
        mask = torch.cat([get_mask(t) for t in tensors], dim=dim)
        return MaskedTensor(res, mask=mask, compressed=compressed[0])
    else:
        res = torch.cat([get_base(t) for t in tensors], dim=dim, out=get_base(out))
        mask = torch.cat([get_mask(t) for t in tensors], dim=dim, out=out.mask)
        return res


@implements(torch.stack)
def stack(tensors, dim=0, out=None):
    """
    Stack tensors and their masks.
    """
    compressed = [get_compressed(tensor) for tensor in tensors]
    if out is None:
        res = torch.stack([get_base(t) for t in tensors], dim=dim)
        mask = torch.stack([get_mask(t) for t in tensors], dim=dim)
        return MaskedTensor(res, mask=mask, compressed=compressed[0])
    else:
        res = torch.stack([get_base(t) for t in tensors], dim=dim, out=get_base(out))
        mask = torch.stack([get_mask(t) for t in tensors], dim=dim, out=out.mask)
        return res


###############################################################################
# Addition
###############################################################################


@implements(torch.add)
def add(inpt, other, alpha=1, out=None):
    """
    Concatenate tensors and their masks.
    """
    compressed = [get_compressed(inpt), get_compressed(other)]
    if out is None:
        res = torch.add(
            get_base(inpt),
            get_base(other),
            alpha=alpha,
        )
        mask = torch.logical_or(
            get_mask(inpt),
            get_mask(other),
        )
        return MaskedTensor(res, mask=mask, compressed=compressed[0])
    else:
        res = torch.add(get_base(inpt), get_base(other), alpha=alpha, out=get_base(out))
        mask = torch.logical_or(get_mask(inpt), get_mask(other), out=out.mask)
        return res


@implements(torch.Tensor.add)
def tadd(inpt, other, alpha=1, out=None):
    """
    Concatenate tensors and their masks.
    """
    return add(inpt, other, alpha=alpha, out=out)


@implements(torch.Tensor.add_)
def iadd(inpt, other, alpha=1):
    """
    Concatenate tensors and their masks.
    """
    get_base(inpt).add_(get_base(other), alpha=alpha)
    if isinstance(inpt, MaskedTensor):
        inpt.mask = combine_masks(inpt, other)
    return inpt


@implements(torch.sub)
def sub(inpt, other, alpha=1, out=None):
    """
    Concatenate tensors and their masks.
    """
    compressed = [get_compressed(inpt), get_compressed(other)]
    if out is None:
        res = torch.sub(
            get_base(inpt),
            get_base(other),
            alpha=alpha,
        )
        mask = combine_masks(inpt, other, torch.logical_or)
        return MaskedTensor(res, mask=mask, compressed=compressed[0])
    else:
        res = torch.sub(get_base(inpt), get_base(other), alpha=alpha, out=get_base(out))
        mask = torch.logical_or(get_mask(inpt), get_mask(other), out=out.mask)
        return res


@implements(torch.Tensor.sub)
def tsub(inpt, other, alpha=1, out=None):
    """
    Concatenate tensors and their masks.
    """
    return sub(inpt, other, alpha=alpha, out=out)


@implements(torch.Tensor.sub_)
def isub(inpt, other, alpha=1):
    """
    Concatenate tensors and their masks.
    """
    get_base(inpt).sub_(other, alpha=alpha)
    if isinstance(inpt, MaskedTensor):
        inpt.mask = torch.logical_or(get_mask(inpt), get_mask(other))
    return inpt


###############################################################################
# Multiplication
###############################################################################


def combine_masks(
        tensor_1,
        tensor_2,
        op=torch.logical_or,
        out=None,
        shape=None
):
    """
    Combine masks of arguments, one of which must be a masked tensor.

    Args:
        tensor_1: The first tensor argument.
        tensor_2: The second tensor argument.
        op: The operation to use to combine the masks.
        out: Optional tensor to write the output to.
        shape: Tuple specifying the expected shape of the result.

    Return:
        A mask tensor obtained by combining the masks from 'tensor_1' and
        'tensor_2'.
    """
    if isinstance(tensor_1, MaskedTensor):
        if isinstance(tensor_2, MaskedTensor):
            if out is not None:
                return op(tensor_1.mask, tensor_2.mask)
            else:
                return op(tensor_1.mask, tensor_2.mask, out=out)
        if shape is not None and tensor_1.mask != shape:
            return torch.broadcast_to(tensor_1.mask, shape)
        return tensor_1.mask
    if shape is not None and tensor_2.mask != shape:
        return torch.broadcast_to(tensor_2.mask, shape)
    return tensor_2.mask





@implements(torch.mul)
def mul(inpt, other, out=None):
    """
    Concatenate tensors and their masks.
    """
    compressed = [get_compressed(inpt), get_compressed(other)]
    if out is None:
        res = torch.mul(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, shape=res.shape)
        return MaskedTensor(res, mask=mask, compressed=compressed[0])
    else:
        res = torch.mul(get_base(inpt), get_base(other), out=get_base(out))
        mask = combine_masks(inpt, other, out=out.mask, shape=res.shape)
        return res


@implements(torch.Tensor.mul)
def tmul(inpt, other, out=None):
    """
    Concatenate tensors and their masks.
    """
    return mul(inpt, other, out=out)


@implements(torch.Tensor.mul_)
def imul(inpt, other):
    """
    Concatenate tensors and their masks.
    """
    get_base(inpt).mul_(other)
    if isinstance(inpt, MaskedTensor):
        inpt.mask = torch.logical_or(get_mask(inpt), get_mask(other))


@implements(torch.permute)
def permute(inpt, dims):
    """
    Permutation of masked tensors.
    """
    base = torch.permute(inpt.base, dims)
    mask = torch.permute(inpt.mask, dims)
    compressed = inpt.compressed
    return MaskedTensor(base, mask=mask, compressed=compressed)


@implements(torch.reshape)
def reshape(inpt, new_shape):
    """
    Reshaping of masked tensors.
    """
    base = torch.reshape(inpt.base, new_shape)
    mask = torch.reshape(inpt.mask, new_shape)
    compressed = inpt.compressed
    return MaskedTensor(base, mask=mask, compressed=compressed)


@implements(torch.Tensor.reshape)
def reshape(inpt, new_shape):
    """
    Reshaping of masked tensors.
    """
    base = inpt.base.reshape(new_shape)
    mask = inpt.mask.reshape(new_shape)
    compressed = inpt.compressed
    return MaskedTensor(base, mask=mask, compressed=compressed)


@implements(torch.squeeze)
def squeeze(inpt, dim=None):
    """
    Squeezing of masked tensors.
    """
    if dim is not None:
        base = torch.squeeze(inpt.base, dim=dim)
        mask = torch.squeeze(inpt.mask, dim=dim)
    else:
        base = torch.squeeze(inpt.base)
        mask = torch.squeeze(inpt.mask)
    compressed = inpt.compressed
    return MaskedTensor(base, mask=mask, compressed=compressed)


@implements(torch.Tensor.squeeze)
def tsqueeze(inpt, dim=None):
    """
    Squeezing of masked tensors.
    """
    if dim is not None:
        base = inpt.base.squeeze(dim=dim)
        mask = inpt.mask.squeeze(dim=dim)
    else:
        base = inpt.base.squeeze()
        mask = inpt.mask.squeeze()
    compressed = inpt.compressed
    return MaskedTensor(base, mask=mask, compressed=compressed)


@implements(torch.unsqueeze)
def unsqueeze(inpt, dim=None):
    """
    Unsqueezing of masked tensors.
    """
    base = torch.unsqueeze(inpt.base, dim=dim)
    mask = torch.unsqueeze(inpt.mask, dim=dim)
    compressed = inpt.compressed
    return MaskedTensor(base, mask=mask, compressed=compressed)


@implements(torch.Tensor.unsqueeze)
def unsqueeze(inpt, dim=None):
    """
    Unsqueezing of masked tensors.
    """
    base = torch.unsqueeze(inpt.base, dim=dim)
    mask = torch.unsqueeze(inpt.mask, dim=dim)
    compressed = inpt.compressed
    return MaskedTensor(base, mask=mask, compressed=compressed)


@implements(torch.sum)
def sum(inpt, dim=None):
    """
    Test summing of tensors.
    """
    return torch.where(inpt.mask, 0.0, inpt.base).sum(dim=dim)


@implements(torch.Tensor.sum)
def tsum(inpt, dim=None):
    """
    Test summing of tensors.
    """
    return torch.where(inpt.mask, 0.0, inpt.base).sum(dim=dim)


@implements(torch.mean)
def mean(inpt, dim=None):
    """
    Test mean of masked tensors.
    """
    inpt_sum = inpt.sum()
    n_elem = (~inpt.mask).sum()
    return inpt_sum / n_elem


@implements(torch.Tensor.mean)
def tmean(inpt, dim=None):
    """
    Test mean of masked tensors.
    """
    inpt_sum = inpt.sum()
    n_elem = (~inpt.mask).sum()
    return inpt_sum / n_elem


@implements(torch.sum)
def tsum(inpt, dim=None):
    """
    Test summing of tensors.
    """
    return torch.where(inpt.mask, 0.0, inpt.base).sum(dim=dim)


@implements(torch.Tensor.view)
def view(inpt, new_shape):
    """
    Reshaping of masked tensors.
    """
    base = inpt.base.view(new_shape)
    mask = inpt.mask.view(new_shape)
    compressed = inpt.compressed
    return MaskedTensor(base, mask=mask, compressed=compressed)


@implements(torch.isclose)
def isclose(inpt, other, rtol=1e-5, atol=1e-8, equal_nan=False):
    """
    Implementation of is close for masked tensors.
    """
    return torch.isclose(
        get_base(inpt), get_base(other), rtol=rtol, atol=atol, equal_nan=equal_nan
    )


@implements(torch.Tensor.__repr__)
def repr(inpt, **kwargs):
    """
    Implementation of __repr__ operator.
    """
    return inpt.base.__repr__(**kwargs)


@implements(torch.Tensor.eq)
def eq(inpt, other, **kwargs):
    """
    Implementation of element-wise comparison.
    """
    if isinstance(inpt, MaskedTensor):
        return MaskedTensor(
            get_base(inpt).__eq__(get_base(other), **kwargs),
            mask=torch.logical_or(get_mask(inpt), get_mask(other)),
        )
    return get_base(inpt).__eq__(get_base(other), **kwargs)


@implements(torch.Tensor.to)
def to(inpt, *args, **kwargs):
    """
    Implementation of element-wise comparison.
    """
    other = inpt.base.to(*args, **kwargs)
    kwargs.pop("dtype", None)
    if len(args) > 0 and isinstance(args[0], torch.dtype):
        args = list(args)
        args[0] = bool

    mask = inpt.mask.to(*args, **kwargs)
    return MaskedTensor(other, mask=mask, compressed=inpt.compressed)


@implements(torch.Tensor.requires_grad.__set__)
def set_requires_grad(inpt, grad):
    inpt.base.requires_grad = grad


@implements(torch.where)
def where(cond, inpt, other, out=None):
    mask_inpt = get_mask(inpt)
    mask_other = get_mask(other)
    cond = get_base(cond)
    inpt = get_base(inpt)
    other = get_base(other)
    compressed = get_compressed(inpt)

    if out is None:
        base = torch.where(cond, inpt, other)
        mask = torch.where(cond, mask_inpt, mask_other)
        return MaskedTensor(base, mask=mask, compressed=compressed)

    base = torch.where(cond, inpt, other, out=out)
    mask = torch.where(cond, mask_inpt, mask_other)
    out.mask = mask
    return out


@implements(torch.ge)
def ge(inpt, other, out=None):
    """
    Element-wise comparison of tensor
    """
    compressed = [get_compressed(inpt), get_compressed(other)]
    if out is None:
        res = torch.ge(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, torch.logical_or, shape=res.shape)
        return MaskedTensor(res, mask=mask, compressed=compressed[0])
    else:
        res = torch.ge(get_base(inpt), get_base(other), out=get_base(out))
        mask = torch.logical_or(get_mask(inpt), get_mask(other), out=out.mask)
        return res


@implements(torch.Tensor.ge)
def tge(inpt, other, out=None):
    """
    Element-wise comparison of tensor
    """
    compressed = [get_compressed(inpt), get_compressed(other)]
    if out is None:
        res = torch.ge(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, torch.logical_or, shape=res.shape)
        return MaskedTensor(res, mask=mask, compressed=compressed[0])
    else:
        res = torch.ge(get_base(inpt), get_base(other), out=get_base(out))
        mask = combine_masks(inpt, other, torch.logical_or, out=out.mask, shape=res.shape)
        return res


@implements(torch.gt)
def gt(inpt, other, out=None):
    """
    Element-wise comparison of tensor
    """
    compressed = [get_compressed(inpt), get_compressed(other)]
    if out is None:
        res = torch.gt(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, torch.logical_or, shape=res.shape)
        return MaskedTensor(res, mask=mask, compressed=compressed[0])
    else:
        res = torch.gt(get_base(inpt), get_base(other), out=get_base(out))
        mask = torch.logical_or(get_mask(inpt), get_mask(other), out=out.mask)
        return res


@implements(torch.Tensor.gt)
def tgt(inpt, other, out=None):
    """
    Element-wise comparison of tensor
    """
    compressed = [get_compressed(inpt), get_compressed(other)]
    if out is None:
        res = torch.gt(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, torch.logical_or, shape=res.shape)
        return MaskedTensor(res, mask=mask, compressed=compressed[0])
    else:
        res = torch.gt(get_base(inpt), get_base(other), out=get_base(out))
        mask = torch.logical_or(get_mask(inpt), get_mask(other), out=out.mask)
        return res


@implements(torch.le)
def le(inpt, other, out=None):
    """
    Element-wise comparison of tensor
    """
    compressed = [get_compressed(inpt), get_compressed(other)]
    if out is None:
        res = torch.le(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, shape=res.shape)
        return MaskedTensor(res, mask=mask, compressed=compressed[0])
    else:
        res = torch.le(get_base(inpt), get_base(other), out=get_base(out))
        mask = combine_masks(inpt, other, out=out.mask, shape=res.shape)
        return res


@implements(torch.Tensor.le)
def tle(inpt, other, out=None):
    """
    Element-wise comparison of tensor
    """
    compressed = [get_compressed(inpt), get_compressed(other)]
    if out is None:
        res = torch.le(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, shape=res.shape)
        return MaskedTensor(res, mask=mask, compressed=compressed[0])
    else:
        res = torch.le(get_base(inpt), get_base(other), out=get_base(out))
        mask = combine_masks(inpt, other, out=out.mask, shape=res.shape)
        return res


@implements(torch.lt)
def lt(inpt, other, out=None):
    """
    Element-wise comparison of tensor
    """
    compressed = [get_compressed(inpt), get_compressed(other)]
    if out is None:
        res = torch.lt(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, shape=res.shape)
        return MaskedTensor(res, mask=mask, compressed=compressed[0])
    else:
        res = torch.lt(get_base(inpt), get_base(other), out=get_base(out))
        mask = combine_masks(inpt, other, out=out.mask, shape=res.shape)
        return res


@implements(torch.Tensor.lt)
def tlt(inpt, other, out=None):
    """
    Element-wise comparison of tensor
    """
    compressed = [get_compressed(inpt), get_compressed(other)]
    if out is None:
        res = torch.lt(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, shape=res.shape)
        return MaskedTensor(res, mask=mask, compressed=compressed[0])
    else:
        res = torch.lt(get_base(inpt), get_base(other), out=get_base(out))
        mask = combine_masks(inpt, other, out=out.mask, shape=res.shape)
        return res

@implements(torch.Tensor.type_as)
def type_as(inpt, other):
    """
    Element-wise comparison of tensor
    """
    return inpt.to(dtype=other.dtype)


@implements(torch.Tensor.__setitem__)
def setitem(inpt, *args, **kwargs):
    get_base(inpt).__setitem__(*[get_base(arg) for arg in args], **kwargs)


@implements(torch.nn.functional.relu)
def relu(inpt, **kwargs):
    return MaskedTensor(
        torch.nn.functional.relu(inpt.base, **kwargs),
        mask=inpt.mask,
        compressed=inpt.compressed,
    )

@implements(torch.pow)
def pow(inpt, exp, *args, out=None):
    return MaskedTensor(torch.pow(inpt.base, exp, *args, out=out), mask=inpt.mask)

@implements(torch.Tensor.pow)
def tpow(inpt, exp, *args, out=None):
    return pow(inpt, exp, *args, out=out)

@implements(torch._C._TensorBase.pow)
def tpow(inpt, exp, *args, out=None):
    return pow(inpt, exp, *args, out=out)

@implements(torch.Tensor.pow_)
def ipow(inpt, exp, *args):
    return pow(inpt, exp, *args, out=inpt.base)

#    torch.sum: torch.any,
#    torch.Tensor.sum: torch.any,
#    torch.mean: torch.any,
#    torch.Tensor.mean: torch.any,
#    torch.unsqueeze: torch.unsqueeze,
#    torch.Tensor.unsqueeze: torch.unsqueeze,
#    torch.squeeze: torch.squeeze,
#    torch.Tensor.squeeze: torch.squeeze,
#    torch.gather: torch.gather
