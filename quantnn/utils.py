"""
=============
quantnn.utils
=============

This module providers Helper functions that are used in multiple other modules.
"""
def apply(f, *args):
    """
    Applies a function to sequence values or dicts of values.

    Args:
        f: The function to apply to ``x`` or all items in ``x``.
        *args: Sequence of arguments to be supplied to ``f``. If all arguments
            are dicts, the function ``f`` is applied key-wise to all elements
            in the dict. Otherwise the function is applied to all provided
            argument.s

    Return:
        ``{k: f(x_1[k], x_1[k], ...) for k in x}`` or ``f(x)`` depending on
        whether ``x_1, ...`` are a dicts or not.
    """
    if all([isinstance(x, dict) for x in args]):
        return {
            k: f(*[x[k] for x in args]) for k in args[0]
        }
    return f(*args)
