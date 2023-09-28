"""
quantnn.models.pytorch.base
===========================

Helper classes for pytorch models.
"""


class ParamCount:
    """
    Mixin class for pytorch modules that add a 'n_params' attribute
    to the class.
    """

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
