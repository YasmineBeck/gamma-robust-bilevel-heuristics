# Introduction
The repository contains the code accompanying the paper

&emsp; _Heuristic Methods for Mixed-Integer, Linear, and Gamma-Robust Bilevel Problems_

by Yasmine Beck, Ivana Ljubic, and Martin Schmidt (2024)
(to appear).

# Dependencies
The methods are implemented in Python 3.7.11 and Gurobi 11.0.0 
is used to solve all arising optimization problems.
Further, the following Python packages and modules are required:

* argparse
* json
* logging
* numpy
* os
* subprocess
* time

# Usage
### Min-Max Problems
Run
    
`python3 min_max_heuristic.py --instance_file file.txt --conservatism conservatism_value --deviations deviation_values --output_file outfile.json`

to build a Gamma-robust knapsack interdiction problem and solve it heuristically by solving a linear number of knapsack interdiction problems of the nominal type.

#### Necessary arguments:
**--instance_file**  
The file containing the nominal instance data.

**--conservatism**  
Level of conservatism (in percent) must be between 0 and 1.

**--output_file**  
The file to write the output to.

and either  

**--deviations**  
The deviations for the objective function coefficients, e.g., 1 2 1 for a problem of size 3.

or  

**--uncertainty**  
Uncertainty value (in percent) must be between 0 and 1.  

A detailed description of the instance data format and the uncertainty parameterization can be found below.

#### Optional arguments:
**--solver**  
The solver to use for the solution of the problems of the nominal type. Default is the combinatorial approach (_bkp_) by Fukasawa and Weninger (2023) (details can be found at https://github.com/nwoeanhinnogaehr/bkpsolver). To use the branch-and-cut approach based on Fischetti et al. (2019), specify _ic_.

**--modify**  
Use the modified variant of the heuristic in which all bilevel sub-problems are solved first (_True_) or use the variant that alternates between solving bilevel and single-level problems (_False_). Default is _False_.

<br/>  

### General Bilevel Problems
Run
    
`python3 general_heuristic.py --instance_file file.txt --conservatism conservatism_value --deviations deviation_values --output_file outfile.json`

to build a generalized Gamma-robust knapsack interdiction problem and solve it heuristically by solving a linear number of generalized knapsack interdiction problems of the nominal type.

#### Necessary arguments:
Same as for the min-max setting, see above.

#### Optional arguments:
**--refine**  
Include a refinement step (_True_) to account for an optimistic follower or not (_False_). Default is _True_.

# Instance Format
Nominal instance data must be given in form of a dictionary.
For example, the nominal instance considered in Example 1 of
Caprara et al. (2016) would be given in a simple text file containing:

{  
"size": 3,  
"profits": [4, 3, 3],  
"leader weights": [2, 1, 1],  
"follower weights": [4, 3, 2],  
"leader budget": 2,  
"follower budget": 4  
}

The deterministic instances that are used in the computational study of the paper are included in the [nominal-instance-data](./nominal-instance-data) directory.

To account for uncertain objective function coefficients, the following specifications may be used:

**conservatism**  
A value between 0 and 1 is required, which specifies the percentage that the parameter Gamma takes of the instance size. In the case of a fractional value for Gamma, the closest integer is considered.

**uncertainty**  
The percentage for the deviations in the objective function coefficients (all coefficients are equally perturbed). The value must be between 0 and 1.

**deviations**  
Absolute values for the deviations in the objective function coefficients.

Either _uncertainty_ or _deviations_ must be specified.

# Counterexample
As discussed in the paper, the main result by Bertsimas and Sim (2003) cannot be carried over to the bilevel setting. For the case of general mixed-integer, linear, and Gamma-robust bilevel problems, it may even be the case that none of the solutions to the deterministic bilevel sub-problems solved in Line 3 of Algorithm 3 in the paper is feasible for the Gamma-robust bilevel problem. We observe this behavior for the nominal instance of size `n = 40` given in [counterexample.txt](counterexample/counterexample.txt) with the uncertainty parameterization given by `uncertainty = 0.1` and `conservatism = 0.5`. The latter implies that Gamma takes a value of 20 and all lower-level objective function coefficients are equally perturbed by 10 percent of the nominal value. In the [counterexample](./counterexample) directory, we provide the numerical results for all deterministic bilevel sub-problems that need to be solved when applying the primal heuristic for Gamma-robust bilevel problems to this instance. For example, `counterexample_20.json` contains the numerical results for the bilevel sub-problem with sub-problem index 20. To verify that indeed none of the solutions to the deterministic bilevel sub-problems solved when applying the primal heuristic is Gamma-robust feasible, simply run

`python3 check_counterexample.py`.

Here, for each fixed leader's decision obtained from solving a bilevel sub-problem of the nominal type, all lower-level sub-problems are solved (cf. Lemma 1 of the paper) to determine whether the pair (x,y) that is output as a solution of the bilevel sub-problem is Gamma-robust feasible. Running the above script returns the message that there is a bilevel sub-problem for which the solution is Gamma-robust feasible or, otherwise, that no such bilevel sub-problem exists.

# Contents
**check_counterexample.py**  
This script is used to verify that there may be instances for which none of the solutions to the general deterministic bilevel sub-problems is feasible for the overall Gamma-robust bilevel problem using a specific example; see Section 4 in the paper.

**counterexample**  
Directory containing the nominal instance data of the counterexample as well as the numerical results for the general deterministic bilevel sub-problems solved using our primal heuristic. The data is used to verify that there may be instances for which none of the solutions to the sub-problems is feasible for the overall Gamma-robust bilevel problem; see Section 4 in the paper.

**combinatorial_approach.py**  
Solve a bilevel knapsack interdiction problem using the combinatorial approach (bkpsolver) by Fukasawa and Weninger (2023). To install _bkpsolver_, follow the instructions at https://github.com/nwoeanhinnogaehr/bkpsolver.

**gamma_robust_extended_model.py**  
Solves an extended formulation of the generalized Gamma-robust bilevel knapsack interdiction problem using a branch-and-cut approach. The latter exploits the ideas of Beck et al. (2023) and Fischetti et al. (2019).

**general_deterministic_model.py**  
Solves a generalized knapsack interdiction problem of the nominal type using a branch-and-cut approach that exploits the ideas of Fischetti et al. (2019).

**general_heuristic.py**  
Primal heuristic for general mixed-integer, linear, and Gamma-robust bilevel problems on the example of generalized bilevel knapsack interdiction problems. The method exploits the solution of a linear number of bilevel problems of the nominal type.

**help_functions.py**  
Contains the following functions that are used in the presented branch-and-cut approaches for bilevel problems of the nominal type (see also the paper by Fischetti et al. (2019) for further details):
* get_dominance  
Determines the set of items that satisfy certain dominance properties such that additional constraints on the leader's decision can be added.

* lifted_cut_separation  
Determines the set of items that satisfy the requirements for lifting interdiction cuts.

* make_maximal  
Completes a feasible decision to a maximal packing.

* solve_lower_level  
Solves a parameterized lower-level problem of the nominal type.

* solve_refinement_problem  
Solves a parameterized refinement problem to account for an optimistic follower.

**instance_data_builder.py**  
Takes a nominal (generalized) bilevel knapsack interdiction instance and returns a robustified instance based on the uncertainty parameterization given by _conservatism_ and _uncertainty_ or _deviations_.

**nominal-instance-data**  
Contains all 280 nominal instances that are used for the computational study of the paper.

**min_max_deterministic_model.py**  
Solves a knapsack interdiction problem of the nominal type using a branch-and-cut approach based on Fischetti et al. (2019). In the paper, several enhancement techniques are discussed. Here, lifted interdiction cuts, maximal packings of the follower, and dominance inequalities are incorporated.

**min_max_heuristic.py**  
Primal heuristics for mixed-integer, linear, and Gamma-robust min-max problems on the example of bilevel knapsack interdiction problems. The method exploits the solution of a linear number of interdiction problems of the nominal type. For the solution of the deterministic problems, two options are available: a branch-and-cut approach based on Fischetti et al. (2019) or the combinatorial approach by Fukasawa and Weninger (2023). In addition, two variants of the heuristic are implemented: one that alternates between solving bilevel and single-level problems and one that solves all bilevel sub-problems first and, afterward, performs a correction step by solving single-level problems.

**optimality_checker.py**  
Class containing all necessary functions (compute upper bound, check ex-post conditions) to prove optimality of heuristically obtained solutions to Gamma-robust min-max problems.
