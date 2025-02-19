# Heuristic Methods for $\Gamma$-Robust Mixed-Integer Linear Bilevel Problems

## Description  
This repository contains the code accompanying the paper [Heuristic Methods for Gamma-Robust Mixed-Integer Linear Bilevel Problems](https://optimization-online.org/?p=26186) by Yasmine Beck, Ivana Ljubić, and Martin Schmidt.

## Prerequisites  
The methods are implemented in `Python 3.7.11` and `Gurobi 11.0.0` is used to solve all arising optimization problems. Visit [Gurobi's official website](https://www.gurobi.com/academia/academic-program-and-licenses) for details on how to obtain a license. In addition to Gurobi, the following Python packages and modules are required:

* argparse
* json
* numpy
* os
* subprocess
* time

Moreover, the heuristics use the `bkpsolver` presented in [Weninger and Fukasawa (2023)](https://link.springer.com/chapter/10.1007/978-3-031-32726-1_31). To install the `bkpsolver`, follow the instructions at https://github.com/nwoeanhinnogaehr/bkpsolver. 

## Usage  
In the following, we elaborate on how to use the exact and heuristic approaches considered in the computational study of [the paper](https://optimization-online.org/?p=26186). We distinguish between the min-max and the more general bilevel setting.  

## Approaches for $\Gamma$-Robust Min-Max Problems Applied to the $\Gamma$-Robust Knapsack Interdiction Problem  

### 1. Heuristics Presented in [the Paper](https://optimization-online.org/?p=26186)  
From [the main directory](./), run

```
python3 -m src.min_max_heuristic --instance_file file.txt --conservatism conservatism_value --deviations deviation_values --output_file outfile.json
```

to build a $\Gamma$-robust knapsack interdiction problem and solve it heuristically by solving a linear number of knapsack interdiction problems of the nominal type.

#### Necessary arguments:
`--instance_file`  
The file containing the nominal instance data.

`--conservatism`  
Level of conservatism (in percent) must be a scalar between 0 and 1.

`--output_file`  
The .json file to write the output to.

and either  

`--deviations`  
The deviations for the objective function coefficients, e.g., `1 2 1` for a problem of size 3.

or  

`--uncertainty`  
Uncertainty value (in percent) must be a scalar between 0 and 1.  

A detailed description of the instance data format and the uncertainty parameterization can be found in the [data directory](data).

#### Optional arguments:  
`--solver`  
The solver to use for the solution of the problems of the nominal type. The default is the combinatorial approach (`--solver bkp`) by [Weninger and Fukasawa (2023)](https://link.springer.com/chapter/10.1007/978-3-031-32726-1_31). To use the branch-and-cut approach based on [Fischetti et al. (2019)](https://pubsonline.informs.org/doi/10.1287/ijoc.2018.0831), specify `--solver ic`.  

To install the `bkpsolver` by Weninger and Fukasawa (2023), follow the instructions at https://github.com/nwoeanhinnogaehr/bkpsolver. The best performance is achieved if the `bkpsolver` is located in the parent directory of this repository. Alternatively, you can modify the path to the solver in the `__init__` section of [min_max_heuristic.py](src/min_max_heuristic.py) to match its location.

`--modify`  
Use the modified variant of the heuristic in which all bilevel sub-problems are solved first (`True`) or use the variant that alternates between solving bilevel and single-level problems (`False`). The default is `False`.

`--time_limit`  
The time limit in seconds. The default is 3600 seconds.

### 2. Greedy Interdiction Heuristic  
To apply a "Greedy Interdiction" heuristic in the spirit of the one presented in Algorithm 4.2 in the [PhD thesis by S. DeNegre](https://coral.ise.lehigh.edu/~ted/files/papers/ScottDeNegreDissertation11.pdf) to the $\Gamma$-robust knapsack interdiction problem, run

```
python3 -m src.greedy_interdiction --instance_file file.txt --conservatism conservatism_value --deviations deviation_values --output_file outfile.json
```

from [the main directory](./) using the same arguments as specified in the necessary arguments section above.

### 3. Exact and Problem-Tailored Branch-and-Cut Approach
To apply the exact and problem-tailored branch-and-cut approach presented in [our earlier work](https://link.springer.com/article/10.1007/s12532-023-00244-6), follow the instructions at https://github.com/YasmineBeck/gamma-robust-knapsack-interdiction-solver (DOI: 10.5281/zenodo.7965281).

## Approaches for General $\Gamma$-Robust Bilevel Problems Applied to the Generalized $\Gamma$-Robust Knapsack Interdiction Problem  

### 1. Heuristics Presented in [the Paper](https://optimization-online.org/?p=26186)  
From [the main directory](./), run

```
python3 -m src.general_heuristic --instance_file file.txt --conservatism conservatism_value --deviations deviation_values --output_file outfile.json
```

to build a generalized $\Gamma$-robust knapsack interdiction problem and solve it heuristically by solving a linear number of generalized knapsack interdiction problems of the nominal type.

#### Necessary arguments:
Same as for the min-max setting, see above.

#### Optional arguments:
`--refine`  
Include a refinement step (`True`) to account for an optimistic follower or not (`False`). The default is `True`.

`--time_limit`  
The time limit in seconds. The default is 3600 seconds.

### 2. ONE-SHOT and ITERATE Heuristics  
To apply the ITERATE heuristic presented in [Fischetti et al. (2018)](https://doi.org/10.1016/j.ejor.2017.11.043) to an instance of the generalized $\Gamma$-robust knapsack interdiction problem, run

```
python3 -m src.iterate_heuristic --instance_file file.txt --conservatism conservatism_value --deviations deviation_values --output_file outfile.json
```

from [the main directory](./) using the same arguments as specified in the necessary arguments section above. The default time limit is 3600 seconds. You can change it using the optional argument `--time_limit TL`, where TL is a scalar specifying the time limit in seconds.

To run the ONE-SHOT variant of the method presented in [Fischetti et al. (2018)](https://doi.org/10.1016/j.ejor.2017.11.043), add the optional argument

```
--one_shot True
```

### 3. Exact and Problem-Tailored Branch-and-Cut Approach
To apply the exact and problem-tailored branch-and-cut approach outlined in [the paper](https://optimization-online.org/?p=26186), run

```
python3 -m src.gamma_robust_extended_model --instance_file file.txt --conservatism conservatism_value --deviations deviation_values --output_file outfile.json
```

from [the main directory](./) using the same arguments as specified in the necessary arguments section above. The default time limit is 3600 seconds. You can change it using the optional argument `--time_limit TL`, where TL is a scalar specifying the time limit in seconds.  

Further details on this approach can also be found in Section 3.5 of [the PhD thesis by Yasmine Beck](https://ubt.opus.hbz-nrw.de/frontdoor/index/index/docId/2432).