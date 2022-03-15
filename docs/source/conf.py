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

import os
import pathlib
import sys

# this path is pointing to project/docs/source
CURRENT_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
SKNEUROMSI_PATH = CURRENT_PATH.parent.parent

sys.path.insert(0, str(SKNEUROMSI_PATH))


import skneuromsi


# -- Project information -----------------------------------------------------

project = "scikit-neuromsi"
copyright = "2021, Paredes, Renato; Cabral, Juan"
author = "Paredes, Renato; Cabral, Juan"

# The full version, including alpha/beta/rc tags
release = skneuromsi.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    # skneuromsi
    "nbsphinx",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
]

# =============================================================================
# EXTRA CONF
# =============================================================================

autodoc_member_order = "bysource"

# =============================================================================
# BIB TEX
# =============================================================================

bibtex_default_style = "apa"  # pybtex-apa-style

bibtex_bibfiles = ["refs.bib"]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# =============================================================================
# INJECT README/CHANGELOG INTO THE RESTRUCTURED TEXT
# =============================================================================

import m2r

DYNAMIC_RST = {
    "README.md": "README.rst",
    "CHANGELOG.md": "CHANGELOG.rst",
}

for md_name, rst_name in DYNAMIC_RST.items():
    md_path = SKNEUROMSI_PATH / md_name
    with open(md_path) as fp:
        readme_md = fp.read().split("<!-- BODY -->")[-1]

    rst_path = CURRENT_PATH / "_dynamic" / rst_name

    with open(rst_path, "w") as fp:
        fp.write(".. FILE AUTO GENERATED !! \n")
        fp.write(m2r.convert(readme_md))
        print(f"{md_path} -> {rst_path} regenerated!")
