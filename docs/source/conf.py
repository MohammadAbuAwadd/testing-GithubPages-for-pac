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
    'sphinx.ext.autodoc', # 
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx_design',
    'numpydoc',
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
templates_path = ['_templates']
exclude_patterns = []
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']




# -- AUTO Documentation -------------------------------------------------
# Removes all paramater info in the autodoc
autodoc_typehints = 'none'  

# Style for google's docstring 
napoleon_google_docstring = True
napoleon_numpy_docstring = False



# autosummary_generate = True
# autodoc_inherit_docstrings = True
# add_function_parentheses = False
# autodoc_docstring_signature = True  # Ensures the signature is part of the docstring
# autodoc_member_order = 'bysource'  # or 'alphabetical'
# html_theme_options = {
#     "navigation_depth": 2,  # Controls how deep the section navigation goes
#     "collapse_navigation": True,  # Set to True if you want collapsible navigation
# }