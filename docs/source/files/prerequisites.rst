.. _prerequisites:

Prerequisites
=============

.. _requirements:

Requirements:
-------------
- Python 3.10.x
-  `Pip <https://www.python.org>`_
- Java 8/11/17 for running `Apache Spark <https://spark.apache.org/docs/3.5.2/#:~:text=Spark%20runs%20on%20Java%208,%2B%2C%20and%20R%203.5%2B.>`_ , we recommend using `Java 8 <https://www.java.com/en/download/help/java8.html>`_

.. note::
    It is advised to create a new python evironment spesifically for this package to avoid issues with dependencies:
        - Creating an evironment with `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
        - Creating an evironment with `venv <https://docs.python.org/3/library/venv.html>`_


.. _parallel:

Running on Parallel
-------------------
- **PySpark Environment:** PySpark is designed for distributed computing 
  and typically runs on clusters. Running PySpark on a standard laptop 
  **is not recommended due to potential performance issues and resource 
  constraints**. Proceed with caution if you choose to run it in such an 
  environment.

- **Core Allocation and n_jobs Parameter:** When using PySpark, if you do 
  not explicitly assign num_cores to 1, it may cause `n_jobs` in joblib to 
  be set to `-1`. This behavior can lead to unintended parallel processing 
  and potential performance degradation. To avoid this issue, ensure that
  you specify the number of `cores` to be used appropriately when 
  configuring A Spark Session.

- **Backend joblib Parameter:**
