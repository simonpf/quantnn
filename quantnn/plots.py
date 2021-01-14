"""
quantnn.plots
=============

This module provides plotting routines for displaying QRNN predictions.
"""
from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
import pathlib
import numpy as np


_STYLE_FILE = pathlib.Path(__file__).parent / "data" / "matplotlib_style.rc"


def set_style():
    """
    Sets matplotlib style to a style file that I find visually more pleasing
    then the default settings.
    """
    plt.style.use(str(_STYLE_FILE))
