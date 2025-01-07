# Data

## Instance Format  
Nominal instance data is given as a dictionary in a simple text file. For example, the nominal instance considered in Example 1 of [Caprara et al. (2016)](https://pubsonline.informs.org/doi/10.1287/ijoc.2015.0676) would be given as

```
{  
"size": 3,  
"profits": [4, 3, 3],  
"leader weights": [2, 1, 1],  
"follower weights": [4, 3, 2],  
"leader budget": 2,  
"follower budget": 4  
}
```

The deterministic instances that are used in the computational study of the paper are included in [this directory](./). Two classes of problems are considered:

**Knapsack Interdiction Problems**  
For each instance size $n = 35, 40, 45, 50, 55, \dots, 100$, 10 instances of the knapsack interdiction problem have been generated according to [Martello et al. (1999)](https://pubsonline.informs.org/doi/10.1287/mnsc.45.3.414). The instances are named using the following format:

```
BKIP_<n>_<i>.txt,
```

where
- `BKIP` stands for bilevel knapsack interdiction problem,  
- `<n>` indicates the number of items (size of the problem),  
- `<i>` is the instance number (an integer between 1 and 10).  

For example, the first out of 10 instances of the knapsack interdiction problem with 35 items would be called

```
BKIP_35_1.txt.
```  

**Generalized Knapsack Interdiction Problems**  
Instances of the generalized knapsack interdiction problem have been obtained from the classic knapsack interdiction problem above by randomly generating and adding upper-level objective function coefficients for the leader's (`leader costs`) and the follower's variables (`follower costs`). Instances of the generalized knapsack interdiction problem follow the same naming convention as the classic knapsack interdiction instances, with the addition of the prefix `generalized_`. An instance of the generalized knapsack interdiction problem could look as follows:

```
{  
"size": 3,  
"profits": [4, 3, 3],  
"leader weights": [2, 1, 1],  
"follower weights": [4, 3, 2],  
"leader budget": 2,  
"follower budget": 4,
"leader costs": [1, 2, 3],  
"follower costs": [2, 2, 1]  
}
```

## Accounting for Uncertainty
To account for uncertain objective function coefficients, the following specifications may be used:

**conservatism**  
A value between 0 and 1 is required, which specifies the percentage that the parameter $\Gamma$ takes of the instance size. In the case of a fractional value for $\Gamma$, the closest integer is considered.

**uncertainty**  
The percentage for the deviations in the objective function coefficients (all coefficients are equally perturbed). The value must be between 0 and 1.

**deviations**  
Absolute values for the deviations in the objective function coefficients.

Either _uncertainty_ or _deviations_ must be specified.