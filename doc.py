
"""
Hybrid Learning of Co-operative Regulation Network.

This function infers a gene regulatory network using hybrid learning techniques by analyzing gene expression data 
and transcription factors.

Parameters
----------
numerical_expression : pandas.DataFrame
    A numerical matrix containing the expression levels of genes and 
    transcription factors. This will be used to infer the network.
    
tf_list : pandas.DataFrame
    A list of genes designated as transcription factors (TFs) or regulators. 
    This should contain gene names.
    
discrete_expression : pandas.DataFrame
    A matrix with the same dimensions, columns, and indices as `numerical_expression`. 
    This should only contain values `{-1, 0, 1}`, where:
    - -1 indicates under-expression,
    - 0 indicates no change, and 
    - 1 indicates over-expression.

gene_list : pandas.DataFrame or list, optional, default: None
    A list of genes for which regulatory networks should be inferred. 
    These genes should appear in the row names of the expression data. 
    If not provided, all genes in the row names of the expression data 
    that are not in `tf_list` will be used.

min_gene_support : float, optional, default: 0.1
    A value between 0 and 1 indicating the minimum proportion of samples 
    in which a gene must have a non-zero discretized expression value 
    to be considered for network inference.

min_coreg_support : float, optional, default: 0.1
    A value between 0 and 1 indicating the minimum proportion of samples 
    in which a set of co-regulators must share the same non-zero discretized 
    expression value to be considered a potential set of co-regulators.

max_coreg : int, optional, default: None
    The maximum size of co-regulator groups to consider. 
    Defaults to the number of TFs. Can be reduced to save memory.

search_thresh : float, optional, default: 1/3
    A value between 0 and 1 specifying the minimum proportion of samples 
    in which a gene and a set of co-regulators must share the same non-zero 
    discretized expression value to be considered co-regulators of the gene.

nGRN : int, optional, default: 100
    The number of gene regulatory networks to infer.

parallel : {'multicore', 'spark', 'no'}, optional, default: 'multicore'
    The parallelization strategy to use. Options are:
    - 'multicore': Use multiple cores by joblib.
    - 'spark': Use Spark for distributed computation.
    - 'no': Do not parallelize.

num_cores : int, optional, default: -1
    The number of CPU cores to use for computation. 
    Set to `-1` to use all available cores.

joblib_backend : {'loky', 'multiprocessing', 'sequential', 'threading'}, optional, default: 'loky'
    The backend for Joblib parallelization.

lr_backend : {'loky', 'multiprocessing', 'sequential', 'threading'}, optional, default: 'sequential'
    The backend for the nested linear regression parallelization.

cluster : SparkSession, optional, default: None
    A Spark session to use for distributed computation.

linear_model : {'cvlm', 'lm', None}, optional, default: 'cvlm'
    The linear model to use:
    - 'cvlm': Cross-validated linear model.
    - 'lm': Ordinary linear model.
    - None: No linear model. GRN returned as default.

verbose : bool, optional, default: False
    If `True`, display additional logging information during execution.

Returns
-------
Class object
    The inferred gene regulatory network(s). The format and type of the return value 
    is a CoRegNet class object that contains the GRN as well as information
    
Attributes
----------
grn : pandas.DataFrame
    A dataframe describing the gene regulatory network (GRN), where:
    - 'Target Gene' : str
        Names of the target genes.
    - 'Co-activators' : list of str
        Co-activators for each target gene.
    - 'Co-repressors' : list of str
        Co-repressors for each target gene.
    - 'Estimated coefficients' : float
        Numerical values estimating the effect of co-activators and co-repressors on the target gene expression.

grn_info : dict
    Metadata for the gene regulatory network (GRN). This dictionary can include information such as:
    - Network description, version, or any other relevant data about the GRN.

adjacencyList : dict of dicts
    A dictionary representing the adjacency list of the GRN. Each key in the outer dictionary corresponds to a gene, and the corresponding value is another dictionary describing its interactions with other genes. For example:
    {
        'GeneA': {'GeneB': weight, 'GeneC': weight},
        'GeneB': {'GeneA': weight}
    }
    The value 'weight' denotes the strength or type of the interaction between genes.

bygene : dict of dicts
    A dictionary where each key is a target gene, and its corresponding value is another dictionary with:
    - 'act' : set of str
        Set of activators (transcription factors or other genes) that activate the target gene.
    - 'rep' : set of str
        Set of repressors that repress the target gene.
    Example:
    {
        'GeneA': {'act': {'TF1', 'TF2'}, 'rep': {'TF3'}},
        'GeneB': {'act': {'TF4'}, 'rep': {'TF5'}}
    }

byt : dict of dicts
    A dictionary where each key is a transcription factor (TF), and its corresponding value is another dictionary with:
    - 'act' : set of str
        Set of genes activated by this transcription factor.
    - 'rep' : set of str
        Set of genes repressed by this transcription factor.
    Example:
    {
        'TF1': {'act': {'GeneA', 'GeneC'}, 'rep': {'GeneB'}},
        'TF2': {'act': {'GeneA'}, 'rep': {'GeneD'}}
    }

inferenceParameters : dict
    A dictionary containing parameters used during the inference process, such as model configurations, thresholds, or algorithm settings. For example:
    {
        'threshold' : float
            A threshold value for inference (e.g., 0.5).
        'model' : str
            The model used for inference (e.g., 'Bayesian').
        'max_iter' : int
            The maximum number of iterations (e.g., 1000).
    }

coRegulators : pandas.DataFrame
    A dataframe specifying inferred co-regulators for each pair of genes or transcription factors. This dataframe can include:
    - 'GeneA' : list of str
        List of co-regulators and their associated measures (e.g., correlation, p-value, etc.).
    - 'GeneB' : list of str
        List of co-regulators for GeneB.
    Each row may include statistics like:
    - 'Co-regulator' : str
        The name of the co-regulator.
    - 'Measure/Statistic' : float
        Numerical values for the measure (e.g., correlation, p-value).
    Example:
    
    +--------+--------+--------------+---------+
    | GeneA  | GeneB  | Co-regulator | Measure |
    +--------+--------+--------------+---------+
    | GeneA  | GeneB  | TF1          | 0.75    |
    +--------+--------+--------------+---------+
    | GeneA  | GeneB  | TF2          | 0.68    |
    +--------+--------+--------------+---------+
"""
