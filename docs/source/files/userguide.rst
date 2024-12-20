User Guide
==========

.. image:: /images/carbon.png

There are two classes: one is responsible for the algorithm, and the other is for preprocessing data. First, import the library after installing it, and import the datasets required to run the algorithm:

::

    from PyHLicorn import HLicorn
    numerical_expression = pd.read_csv(file_path, index_col=0)
    discrete_expression = pd.read_csv(file_path, index_col=0)
    tf_list = pd.read_csv(file_path)

The Numerical Expression data, Discrete Expression data, and the transcription factor list are mandatory to run the algorithm. Alternatively, you can run the :ref:`discretization` to get the Discrete Expression data:
::

    from PyHLicorn import PreProcess
    discrete_expression = PreProcess.discretization(numerical_expression)

Running the :ref:`hlicorn` algorithm with the default parameters:

::

    GRN = HLicorn(numerical_expression, tf_list, discrete_expression)
    GRN = HLicorn(numerical_expression, tf_list.head(150), discrete_expression)
