# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Add project root directory to the path (../../ goes up two levels from docs/source)
sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "song_analyzer_app_6"
copyright = "2025, Donovan Blair"  # Updated year based on current date
author = "Donovan Blair"
release = (
    "6.0.0"  # You can update this to a more standard version like '0.1.0' if you prefer
)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Automatically generate docs from docstrings
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.autosummary",  # Generate summary tables for modules
    "sphinx.ext.inheritance_diagram",  # Generate class inheritance diagrams (needs Graphviz)
    "sphinx.ext.graphviz",  # Embed Graphviz DOT language diagrams (needs Graphviz)
    # Add other extensions here if needed, e.g. 'sphinx.ext.githubpages'
]

# Autosummary settings
autosummary_generate = True

# Napoleon settings (optional, but good for specifying docstring style)
napoleon_google_docstring = True  # Set to True if using Google style docstrings
napoleon_numpy_docstring = False  # Set to False if using Google style docstrings
napoleon_include_init_with_doc = True  # Include docstrings for __init__ methods

# Autodoc default settings
# This ensures that members (methods, etc.) are documented by default
# when autosummary generates stub pages for classes/modules.
autodoc_default_options = {
    "members": True,  # Document members (methods, attributes, etc.)
    "member-order": "bysource",  # Order members as they appear in source code
    "undoc-members": True,  # Include members even if they lack docstrings (optional)
    "show-inheritance": True,  # Show base classes for classes
    # Add other autodoc options if needed
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for Graphviz ----------------------------------------------------
# You might need to tell Sphinx where the Graphviz executables are if they
# aren't in your system's PATH. Uncomment and set the path if needed.
# graphviz_dot = '/usr/local/bin/dot' # Example path for macOS with Homebrew
# graphviz_output_format = 'svg' # 'png' or 'svg', SVG is usually better for web
