Preprocessing
=============
.. N < mean - t x std = - 1
.. N > mean + t x std = 1

.. currentmodule:: pyProject.core

.. _discretization:

Discretization Expression Data Function
----------------------------------------

.. math::
   N < \mu - t \times \sigma  \Rightarrow N = -1
.. math::
   N > \mu + t \times \sigma  \Rightarrow N = 1

Given a continuous log2 gene expression matrix, this function discretizes the expression values into a binary matrix 
based on fold change comparisons. The behavior of the function depends on the input data:

1. **Negative Values**: If the matrix contains negative values, it is assumed to already be in the correct format.The 
   function applies a hard threshold directly to discretize the data.

2. **Positive Values**: If the matrix contains only positive values (common in normalized RNA-seq or single-color microarrays), 
   the function centers each gene’s expression based on its mean across all samples or the mean of a set of reference samples 
   (e.g., normal samples in a study of a specific disease).

In either case, a threshold is used to transform the data:
   - For values above or equal to the threshold, the function assigns a value of +1.
   - For values below the negative of the threshold, the function assigns a value of -1.
   - Values within this range are assigned a value of 0.

By default, the function computes the threshold based on the overall distribution of numerical values in the dataset, rather than using a predefined fold change (such as 1 or 2 corresponding to a two-fold or four-fold increase/decrease). This choice is made to accommodate the variability between different technologies. The decision to use a simple hard fold change threshold or a threshold as a multiplier of the global standard deviation remains flexible.

.. automethod:: PreProcess.discretization

.. _genelist:

Gene List
----------

.. automethod:: PreProcess.gene_list


