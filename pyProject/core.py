import os
import pickle
import json
from typing import Literal, Optional

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession

from ._hlicorn import _hlicorn 
from .modules import _inference_parameters as _inference_parameters
from .modules import _coregulators as _coregulators
from .modules import _co_regulatory_net as _co_regulatory_net

class CoRegNet:
    def __init__(
        self,
        numerical_expression: pd.DataFrame = None,
        tf_list: pd.DataFrame = None,
        discrete_expression: pd.DataFrame = None,
        gene_list: pd.DataFrame = None,
        min_gene_support: float = 0.1,
        min_coreg_support: float = 0.1,
        max_coreg: int = None,
        search_thresh: float = 1 / 3,
        nGRN: int = 100,
        parallel: Optional[Literal['multicore', 'spark', 'no']] = 'multicore',
        num_cores: int = -1,
        joblib_backend: str = 'threading',
        lr_backend: str = 'threading',
        cluster: SparkSession = None,
        linear_model: Optional[Literal['cvlm', 'lm', None]] = 'cvlm',
        verbose: bool = False,
        data: dict = None,
    ) -> None:
        if data:
            self.grn = pd.DataFrame(data['grn'])
            self.grn_info = data['grn_info']
            self.adjacencyList = data['adjacencyList']
            self.bygene = data['bygene']
            self.bytf = data['bytf']
            self.inferenceParameters = data['inferenceParameters']
            self.coRegulators = pd.DataFrame(data['coRegulators'])
            
        elif isinstance(numerical_expression, pd.DataFrame):
            self.grn, self.grn_info = _hlicorn(
                numerical_expression, tf_list, discrete_expression, gene_list,
                min_gene_support, min_coreg_support, max_coreg, search_thresh,
                nGRN, parallel, num_cores, joblib_backend, lr_backend,
                cluster, linear_model, verbose
            )
            
            self = _co_regulatory_net(self)
            
            self.bygene = self.adjacencyList['bygene']
            self.bytf = self.adjacencyList['bytf']
            
            self.inferenceParameters= _inference_parameters(min_gene_support,min_coreg_support,max_coreg,search_thresh,nGRN)
            
            self.coRegulators = _coregulators(self)

    # Handles pickling
    def __reduce__(self):
        state = self.__dict__.copy()
        return (object.__new__, (self.__class__,), state)

    # Setting state for pickling
    def __setstate__(self, state):
        self.__dict__.update(state)

    # Saving GRN as pickle
    def save_grn(self, file_path : str, locally : bool = False) -> None:
        """
        Save GRN to pickle.

        Parameters
        ----------
        file_path : str
            Preferred path to save the file.
        locally : bool, optional
            If True, save the file in the package directory. Default is False.

        Returns
        -------
        None
        """
        
        dir = (os.path.dirname(os.path.dirname(__file__)))
        pth = os.path.join(dir, 'data/processed/grn.pkl')
        
        with open(file_path, "wb") as file:
                pickle.dump(self, file)
                
        if locally == True:
            with open(pth, "wb") as file:
                pickle.dump(self, file)

    # Retrieve GRN from local
    @classmethod
    def get_grn(self) -> object:
        """
        Gets saved GRN from the package directory

        Returns
        -------
        object: GRN 
        """
        
        dir = (os.path.dirname(os.path.dirname(__file__)))
        pth = os.path.join(dir, 'data/processed/grn.pkl')
        
        if os.path.exists(pth):
            with open(pth, 'rb') as file:
                data = pickle.load(file)
        else:
            raise FileNotFoundError("No GRN are saved locally")
        
        return data
    
    # To json
    def save_to_json(self, file_path : str) -> None:
        """
        Save GRN to json file

        Parameters
        ----------
        file_path : str
            Preferred path to save the file.
            
        """
        data = {
            "grn" : self.grn.to_dict(),
            "grn_info" : self.grn_info,
            "adjacencyList" : self.adjacencyList,
            "bygene" : self.bygene,
            "bytf" : self.bytf,
            "inferenceParameters" : self.inferenceParameters,
            "coRegulators" : self.coRegulators.to_dict()
        }
        
        with open(file_path, "w") as json_file:
            json_file.write(json.dumps(data, indent=4))
        
    # load from json
    @classmethod
    def load_grn_json(self , file_path : str) -> object:
        """
        Load GRN from json File.

        Parameters
        ----------
        file_path : str
            Preferred path to the JSON file containing the GRN data.

        Returns
        -------
        object
            The loaded GRN.
        """
        try:
            with open(file_path, 'r') as grn_json_file:
                data = json.load(grn_json_file)
        except FileNotFoundError:
            print(f"no json file at {file_path}")
        except json.JSONDecodeError:
            print("There was an error decoding the JSON file.")
        
        return CoRegNet(data=data)
    
    # Data Handling Methods
    
    # Regulators Extraction
    def get_regulators(self, target_gene : str) -> dict:
        """
        Retrieves regulators for a given target gene from the Gene Regulatory Network (GRN).

        Parameters
        ----------
        target_gene : str
            The name of the target gene for which regulators are to be fetched.

        Returns
        -------
        dict
            A dictionary with two keys:
            - 'act' : list
                A list of activators for the target gene.
            - 'rep' : list
                A list of repressors for the target gene.
        """

        if not hasattr(self, 'adjacencyList') or 'bygene' not in self.adjacencyList:
            self.adjacency_list()

        bygene = self.adjacencyList['bygene']
        
        if target_gene not in bygene:
            return {'act': [], 'rep': []}
        
        return bygene[target_gene]

    # Regulators Target Genes
    def get_targets(self, regulator : str)-> dict:
        """
        Retrieves the target genes regulated by a specific regulator from the Gene Regulatory Network (GRN).

        Parameters
        ----------
        regulator : str
            The name of the regulator (e.g., transcription factor) whose target genes are to be fetched.

        Returns
        -------
        dict
            A dictionary with two keys:
            - 'act' : list
                A list of target genes activated by the regulator.
            - 'rep' : list
                A list of target genes repressed by the regulator.
        """
        if not hasattr(self, 'adjacencyList') or 'bytf' not in self.adjacencyList:
            self.adjacency_list()

        bytf = self.adjacencyList['bytf']
        
        if regulator not in bytf:
            return {'act': [], 'rep': []}
        
        return bytf[regulator]


    
    
        

class HLicorn(CoRegNet):
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
            - 'Co-act' : list of str
                Co-activators for each target gene.
            - 'Co-rep' : list of str
                Co-repressors for each target gene.
            - 'Coef.Acts' : list of floats
                Coefficient for each Activators.
            - 'Coef.Reps' : list of floats
                Coefficient for each Repressors.
            - 'Coef.coActs' : list of floats
                Coefficient for each Co-activators.
            - 'Coef.coReps' : list of floats
                Coefficient for each Co-repressors.
            - 'R2' : float 
                Coefficient of determination (R2) of the network for the target gene 
            - 'RMSE' : float
                The root-mean-square-error (RMSE) of the network for the target gene 
                
    grn_info : dict
        Metadata for the gene regulatory network (GRN). This dictionary can include information such as:
        
            - {'gene_list' : len(gene_list), 'tf_list': len(tf_list), 'co_regs' : len(co_regs) }

    adjacencyList : dict of dicts
        A dictionary representing the adjacency list of the GRN. Each key in the outer dictionary corresponds to a gene, 
        and the corresponding value is another dictionary describing its interactions with other genes. Includes `GRN.bygene` and `GRN.bytf`
           

    bygene : dict of dicts
        A dictionary where each key is a target gene, and its corresponding value is another dictionary with:
            - 'act' : set of str
                Set of activators (transcription factors or other genes) that activate the target gene.
            - 'rep' : set of str
                Set of repressors that repress the target gene.

    bytf : dict of dicts
        A dictionary where each key is a transcription factor (TF), and its corresponding value is another dictionary with:
            - 'act' : set of str    
                Set of genes activated by this transcription factor.
            - 'rep' : set of str
                Set of genes repressed by this transcription factor.

    inferenceParameters : dict
        A dictionary containing parameters used during the inference process. For example:
            - min_gene_support, min_coreg_support, max_coreg, search_thresh, and nGRN.


    coRegulators : pandas.DataFrame
        A dataframe specifying inferred co-regulators for each pair of genes or transcription factors. This dataframe can include:
        
            - 'Reg1' : str
                Regulator One
            - 'Reg2' : str
                Regulator Two
            - 'support' : float
                The name of the co-regulator.
            - 'nGRN' : int
                The number of gene regulatory networks
            - 'fisherTest' : float
                Fisher's exact test p-value calculated using the `fisher_exact, alternative='greater'`  function from `scipy.stats`.
            - 'adjustedPvalue' : float
                Adjusted p-values calculated using the Holm method from `statsmodels.stats.multitest`.

        Example:
    
        +--------+--------+---------+-------+------------+----------------+
        | Reg1   | Reg2   | Support | nGRN  | fisherTest | adjustedPvalue |
        +--------+--------+---------+-------+------------+----------------+
        | GeneA  | GeneB  | float   | float | float      | float          |
        +--------+--------+---------+-------+------------+----------------+
        | GeneA  | GeneB  | float   | float | float      | float          |
        +--------+--------+---------+-------+------------+----------------+
    """
    
    def __init__(
        self,
        numerical_expression: pd.DataFrame,
        tf_list: pd.DataFrame,
        discrete_expression: pd.DataFrame,
        gene_list: pd.DataFrame | list = None,
        min_gene_support: float = 0.1,
        min_coreg_support: float = 0.1,
        max_coreg: int = None,
        search_thresh: float = 1 / 3,
        nGRN: int = 100,
        parallel: Optional[Literal['multicore', 'spark', 'no']] = 'multicore',
        num_cores: int = -1,
        joblib_backend: Optional[Literal['loky', 'multiprocessing', 'sequential', 'threading']] = 'loky',
        lr_backend: Optional[Literal['loky', 'multiprocessing', 'sequential', 'threading']] = 'sequential',
        cluster: SparkSession = None,
        linear_model: Optional[Literal['cvlm', 'lm', None]] = 'cvlm',
        verbose: bool = False,
    ) -> None:

        super().__init__(
            numerical_expression, tf_list, discrete_expression, gene_list,
            min_gene_support, min_coreg_support, max_coreg, search_thresh,
            nGRN, parallel, num_cores, joblib_backend, lr_backend,
            cluster, linear_model, verbose
        )

    # Handles pickling
    def __reduce__(self):
        state = self.__dict__.copy()
        return (object.__new__, (self.__class__,), state)

    # Setting state for pickling
    def __setstate__(self, state):
        self.__dict__.update(state)



class PreProcess:
    
    @classmethod
    def discretization(
        numericalExpression : pd.DataFrame,
        threshold : float | None ,
        refSamples = list| None ,
        sdBy = 'genes',
        center : bool = True
        ):
        """
        Discretization function for Numerical Expression data.

        Parameters
        ----------
        numericalExpression : pandas.DataFrame
            A numerical matrix containing the expression levels of genes and 
            transcription factors. This will be used to infer the network.
            
        threshold : float or None
            A numeric value used as a fixed fold-change threshold for discretization. 
            Can be None if another thresholding method is used.
            
        refSamples : list or None, optional
            A vector of column names used as a set of reference samples for computing fold changes.
            Can be None if the input data is already centered on reference values (indicated by 
            the presence of negative values) or if the mean of each gene should be used 
            to center (but not scale) the expression values. Default is None.
            
        sdBy : str or None, optional
            Specifies the level of standard deviation calculation. 
            Use `'genes'` to compute the standard deviation for each gene individually. 
            If None, this step is skipped. Default is None.
            
        center : bool, optional
            Whether to center the data. Default is True.

        Returns
        -------
        pd.DataFrame
            The discretized numerical expression data.
        """
        numericalExpression = np.matrix(numericalExpression)

        if center:
            if sdBy == 'genes':  #By genes (rows)
                if refSamples is not None:
                    ref_means = np.mean(numericalExpression[:, refSamples], axis=1)
                    centered = numericalExpression - ref_means
                else:
                    centered = numericalExpression - np.mean(numericalExpression, axis=1)
            else:  #By samples (columns)
                if refSamples is not None:
                    ref_means = np.mean(numericalExpression[:, refSamples], axis=0)
                    centered = numericalExpression - ref_means
                else:
                    centered = numericalExpression - np.mean(numericalExpression, axis=0)

            centered[np.isnan(centered)] = 0  
        else:
            centered=numericalExpression.copy()

        if sdBy == 'genes':
            sds = np.std(centered, axis=1)
        else:
            sds = np.std(centered, axis=0)
    
        if threshold is None:
            threshold = 1
        
        standardDeviationThreshold = threshold * sds
    
        discretized = np.zeros_like(centered)
        discretized[centered > standardDeviationThreshold] = 1
        discretized[centered < -standardDeviationThreshold] = -1

        return discretized

    # Get gene list 
    @classmethod
    def gene_list(numerical_expression : pd.DataFrame, tf_list: pd.DataFrame):
        """
        A list of genes for which regulatory networks should be inferred. 
        These genes should appear in the row names of the expression data. 
        If not provided, all genes in the row names of the expression data 
        that are not in `tf_list` will be used.
        
        Parameters
        ----------
        numerical_expression : pandas.DataFrame
            A numerical matrix containing the expression levels of genes and 
            transcription factors. This will be used to infer the network.
            
        tf_list : pandas.DataFrame
            A list of genes designated as transcription factors (TFs) or regulators. 
            This should contain gene names.
            
        Returns
        -------
        list
        """
        num_row_names = numerical_expression.index.tolist()
        gene_list = list(set(num_row_names) - set(tf_list))
        gene_list = list(set(numerical_expression.index).intersection(gene_list))
        
        return gene_list
