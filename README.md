# PyCoRegNet

A python package that is a re-implementation of an existing deprecated R Package 

The CoRegNet package aims at inferring a large scale transcription co-regulatory network from transcriptomic data and at integrating external data on gene regulation to infer and analyze transcriptional programs. The particularity of the network inference algorithm proposed in the package is to learn co-regulation network in which gene regulation is modeled by transcription factors acting in a cooperative manner to synergistically regulate target genes.

## About the Project
The package was used in a study of Bladder Cancer to identify the driver transcriptional programs from a set of 183 samples. Throughout this Vignette, a smaller version of the transcriptomic dataset is used to illustrate the use of the package.


## Prerequisites

### Ensure you have the following prerequisites installed:

- Python 3.10.x
- [pip](https://pip.pypa.io/en/stable/) (Python package manager)

### pySpark Notes

- **PySpark Environment:** PySpark is designed for distributed computing and typically runs on clusters. Running PySpark on a standard laptop **``is not recommended due to potential performance issues and resource constraints``**. Proceed with caution if you choose to run it in such an environment.

- **Java Requirements:** PySpark requires Java 8 to function properly. Ensure that Java 8 is installed and correctly configured on your system.

- **Core Allocation and n_jobs Parameter:** When using PySpark, if you do not explicitly assign num_cores to 1, it may cause `n_jobs` in joblib to be set to `-1`. This behavior can lead to unintended parallel processing and potential performance degradation. To avoid this issue, ensure that you specify the number of `cores` to be used appropriately when configuring `PySpark`.



## Installation

**Clone the repository:**

```bash
pip install PyCoRegNet
```
<!-- 
```bash
python3 install PyCoRegNet
``` -->
















## Usage

```python
  import pandas as pd
  from PyCoRegNet import coregnet
```
```python
#index_col = 0 is MANDATORY for all paramters
 numerical_expression = pd.read_csv('path',index_col = 0)
 tf_list = pd.read_csv('path',index_col = 0) 
 discrete_expression = pd.read_csv('path',index_col = 0) 
```

```python
  GRN = coregnet(numerical_expression,tf_list,discrete_expression)
```

## Default Input
| Parameter | Type     | Value                |
| :-------- | :------- | :------------------------- |
| `numerical_expression` | `pandas.DataFrame` | **Required** |
| `tf_list` | `pandas.DataFrame` | **Required**  |
| `discrete_expression` | `pandas.DataFrame` | **Required**  |
| `gene_list` | `pandas.DataFrame` | **None**  |
| `min_gene_support` | `float` | **0.1**  |
| `min_coreg_support` | `float` | **0.1**  |
| `search_thresh` | `int` | **1/3**  |
| `nGRN` | `int` | **100** |
| `parallel` | `str` | **"multicore"**  |
| `num_cores` | `int` | **-1**  |
| `cluster` | `SparkSession` | **None**  |
| `linear_model` | `str` | **"cvlm"**  |
| `verbose` | `bool` | **Flase**  |

<br>

## Coregnet Class


<!-- default=None -->
### Parameters

- **numerical_expression : _pd.DataFrame_**: 

    A numerical Matrix containing the expression of genes and of
    transcription factors that will be used to inferred the network.
    Rownames should contain the Gene names/identifiers. 
    Samples should be in columns but Colnames are not important.
    The data will be gene-centered but not scaled.

- **tf_list : _pd.DataFrame_**: 

    A character vector containing the names of the genes that
        are designated as Transcription Factor or Regulators.

- **discrete_expression : _pd.DataFrame_**: 

    Should be in exactly the same format as numerical_expression
        Same Dimensions, columns and index 
        Should only contain value of `{-1,0,1} `with -1 for under-expressed, 0 for no change and 1 for over expressed.
 


- **gene_list : _pd.DataFrame, defualt=None_**: 

   The list of genes for which Gene Regulatory Networks
        should be inferred. Should be in the rownames of the expression data. 
        If not provided will be taken as all the genes in the rownames
        of the expression data that are not annotated as TF in  `tf_list` 
    

- **minGeneSupport :_float, defualt=0.1_**: 

   A float between `0 and 1`. 
        Minimum number of samples in which a gene has to have non-zero discretized expression value to be considered for regulatory network inference.

- **minGeneSupport :_float, defualt=0.1_**: 

   A float between` 0 and 1`. 
        Minimum number of samples in which a set of co-regulators have the same non-zero discretized expression value to be considered as a potential set of co-regulator. Default is 0.1 . Can be increased in case of limitations on memmory.

- **maxCoreg :_int, defualt=None_**: 

    An integer. 
        Maximum size of co-regulator to consider. Default is set to the number of TF. 
        Can be decreased in case of limitations on memmory.

- **searchThresh :_float, defualt=1/3_**: 

   A float between `0 and 1`. 
        Minimum proportion of sample in which a gene and a set of co-regulators must have the same 
        non-zero discretized value in order to be considered as a potential co-regulator of the gene.


- **searchThresh :_float, defualt=1/3_**: 

   A float between `0 and 1`. 
        Minimum proportion of sample in which a gene and a set of co-regulators must have the same 
        non-zero discretized value in order to be considered as a potential co-regulator of the gene.


- **nGRN :_int, defualt=None_**: 

   A float between `0 and 1`. 
        Minimum proportion of sample in which a gene and a set of co-regulators must have the same 
        non-zero discretized value in order to be considered as a potential co-regulator of the gene.

- **parallel :_str,{multicore,spark,no}, defualt=`multicore`_**:      
        Defualt : `multicore` <br>
        `multicore` = joblib <br>
        `spark` = pyspark <br>
        `no` = default for loop 

- **num_cores :_int,defualt=-1_**: 
       Number of desired cores for multithreading 
       

- **cluster :_SparkSession,defualt=None_**: 
        pySpark SparkSession

- **linear_model :_str,{"cvlm','lm',None},defualt=`cvlm`_**:<br>
    cvlm: Linear Regression's Cross Verification :
    from sklearn.model_selection import cross_val_predict<br>
    lm : Defualt Linear Regression <br>
    None : returns licorn algorithm output, which is the GRN without coefficients, r2 score, and rsme score. 

## Return
### A python class object
- After running the code you can comment `GRN = coregnet(n...` and type `GRN = get_grn()`, this returns last run GRN, if you want to get a specific grn, type the name of the GRN you saved by `GRN.save_grn('example_grn')` in get_grn(`example_grn`)
```python
GRN = get_grn()
```



## Attributes

- Co-Regulary Network

```python
GRN.grn
```

- co-Regulators 

```python
GRN.coRegulators
```

- Adjacency List
```python
GRN.adjacencyList
GRN.bygene 
GRN.bytf
```

- Inference Parameters
```python
GRN.inferenceParameters
```

- Save as desired name
```python
GRN.save_grn('grn')
```





## Exporting The Coregnet 

- To csv
```python
import pandas as pd
GRN.grn.to_csv(path)
```
- To your directory of choice
```python
GRN.export_grn(self, path : str) -> pickle:
```

## Future Updates
- Automatic Discrete Expression
- Dashboard for results
- More methods


## License

Distributed under the GNU License. See `LICENSE.txt` for more information.

## Contact

Your Name - [your-email@example.com](mailto:your-email@example.com)

Project Link: [https://github.com/your-username/project-name](https://github.com/your-username/project-name)

## Acknowledgements

- [Resource 1](https://example.com)
- [Resource 2](https://example.com)
- [Resource 3](https://example.com)


