# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'quantnn'
copyright = '2020, Simon Pfreundschuh'
author = 'Simon Pfreundschuh'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["nbsphinx", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'smpl'
html_theme_options = {
    "navigation_bar_minimum_height": "15vh",
    "navigation_bar_targets": ["index.html",
                               "user_guide.html",
                               "examples.html",
                               "api_reference.html"],
    "navigation_bar_names": ["Home", "User guide", "Examples", "API Reference"],
    "navigation_bar_element_padding": "40px",
    "navigation_bar_background_color": "#333333",
    "navigation_bar_element_hover_color": "#ff5050",
    "navigation_bar_border_color": "#ff5050",
    "navigation_bar_border_style": "solid",
    "navigation_bar_border_width": "0px 0px 0px 0px",

    "link_color":  "#ff5050",
    "link_visited_color":  "#ff5050",
    "link_hover_color":  "#990000",

    "sidebars_right": [],
    "sidebars_left":["localtoc.html", "globaltoc.html"],
    "globaltoc_maxdepth": 1,
    "inline_code_border_radius": "2px",
    "sidebar_left_border_color": "#ff5050",

    "highlight_border_color": "#ff5050",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
