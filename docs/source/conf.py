# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Imports -----------------------------------------------------
import os
import sys

# -- Path info -----------------------------------------------------
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyHLicron'
copyright = '2024, Graziella'
author = 'Graziella'
release = ''

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx_design',
    'numpydoc',
    'sphinx.ext.todo',
    # 'sphinx.ext.toc',
]

# For Auto doc

napoleon_numpy_docstring = True
napoleon_google_docstring = False  

autosummary_generate = True
autodoc_inherit_docstrings = True
add_function_parentheses = False



templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_theme = 'furo'
html_theme = 'pydata_sphinx_theme'

html_static_path = ['_static']


napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Autodoc settings to remove type hints and clean up the docs
autodoc_typehints = 'none'  # Removes type hints from the generated docs
autodoc_docstring_signature = True  # Ensures the signature is part of the docstring
autodoc_default_options = {
    'members': True,             # Include class/method members
    'undoc-members': True,       # Include undocumented members
    'show-inheritance': True,    # Show inheritance for classes
}

# # conf.py
# autodoc_default_options = {
#     'members': True,
#     'undoc-members': True,
#     'show-inheritance': True,
#     'special-members': '__init__',
# }

# # conf.py
# autodoc_typehints = 'description'  # or 'both' to show in both description and signature
# # conf.py
# autodoc_member_order = 'bysource'  # or 'alphabetical'
# # conf.py
# autodoc_docstring_signature = True
# # conf.py
# autodoc_mock_imports = ["some_optional_library"]

html_theme_options = {
    "navigation_depth": 0,  # Controls how deep the section navigation goes
    "collapse_navigation": True,  # Set to True if you want collapsible navigation
}