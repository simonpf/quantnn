import numpy as np
import torch


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
    torch.gather: torch.gather
}


class MaskedTensor(torch.Tensor):

    def __new__(cls, *args, **kwargs):
        mask = kwargs.pop("mask", None)
        compressed = kwargs.pop("compressed", None)
        tensor = super().__new__(cls, *args, **kwargs)
        if mask is None:
            if isinstance(args[0], MaskedTensor):
                mask = args[0].mask
        if compressed is None:
            if isinstance(args[0], MaskedTensor):
                compressed = args[0].compressed

        if mask is None:
            tensor = args[0]
            mask = torch.zeros(tensor.shape, dtype=bool, device=tensor.device)

        if isinstance(mask, cls):
            mask = mask.strip()

        tensor.mask = mask.detach()
        tensor.compressed = compressed
        tensor.empty = ~mask.any()
        return tensor


    def strip(self):
        return torch.Tensor(self)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):

        if kwargs is None:
            kwargs = {}

        masks = list(map(extract_mask, args))
        args_s = list(map(strip_type, args))
        kwargs_s = strip_type(kwargs)

        result = func(*args_s, **kwargs_s)

        if not isinstance(result, torch.Tensor):
            return result

        compressed = None
        for arg in args:
            if isinstance(arg, MaskedTensor):
                compressed = arg.compressed
                break
        if compressed is not None:
            for arg in kwargs.values():
                if isinstance(arg, MaskedTensor):
                    compressed = arg.compressed
                    break

        mask_func = MASK_HANDLERS.get(func, None)
        if mask_func is not None:
            masks = [
                mask.to(device=masks[0].device) if isinstance(mask, torch.Tensor) else mask
                for mask in masks
            ]
            if "out" in kwargs:
                kwargs.pop("out")
            mask_result = mask_func(*masks, **kwargs)
            return MaskedTensor(result, mask=mask_result, compressed=compressed)

        return MaskedTensor(result, mask=masks[0], compressed=compressed)


    def compress(self):
        n_tot = self.shape[0]
        all_missing = self.mask.view((n_tot, -1)).all(dim=-1)
        valid = torch.nonzero(~all_missing)[:, 0]
        return MaskedTensor(
            self[valid],
            compressed=(n_tot, valid)
        )

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
            mask=self.mask.__getitem__(*args, **kwargs)
        )


def extract_mask(arg):
    if isinstance(arg, (tuple, list, set)):
        return [extract_mask(nested) for nested in arg]
    elif isinstance(arg, dict):
        return {key: extract_mask(nested) for key, nested in arg.items()}
    elif isinstance(arg, MaskedTensor):
        return arg.mask
    elif isinstance(arg, torch.Tensor):
        return torch.zeros_like(arg, dtype=bool)
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
