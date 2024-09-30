# This file is part of the code used for the computational study
# in the paper
#
#     "Heuristic Methods for Gamma-Robust Mixed-Integer Linear
#      Bilevel Problems"
#
# by Yasmine Beck, Ivana Ljubic, and Martin Schmidt (2024).

# Global imports
import json
import numpy as np
import os

# Local imports
from help_functions import solve_lower_level
from instance_data_builder import InstanceDataBuilder
from general_deterministic_model import GeneralDeterministicModel
            
def build_robustified_instance(instance_file, conservatism, uncertainty,
                               deviations, tol=1e-04):
    # Build a robustified instance from the nominal data, the specified
    # uncertainty, the given deviations, and the level of conservatism.
    builder = InstanceDataBuilder(
        instance_file,
        conservatism=conservatism,
        uncertainty=uncertainty,
        deviations=deviations
    )

    instance_dict = builder.build_robustified_instance()
    gamma = instance_dict['gamma']
    size = instance_dict['size']
    devs = instance_dict['deviations']

    # Determine the set of sub-problems to be solved.
    if gamma < 1:
        subprobs = [0]
    else:
        # Reduction of sub-problems according to Lee and Kwon (2014).
        max_idx = int(np.ceil((size - gamma)/2)) + 1
        subprobs = [gamma + 2*idx - 1 for idx in range(1, max_idx)]
        subprobs.append(size + 1)

        # Account for Python indexing starting at 0.
        subprob_cnt = len(subprobs)
        temp_subprobs = [subprobs[idx] - 1 for idx in range(subprob_cnt)]

        # Only the sub-problems with pairwise distinct associated
        # deviations need to be considered, i.e., remove one of the
        # sub-problems with a duplicate value for the deviations
        # (see Proposition 3).
        subprobs = [temp_subprobs[0]]
        for idx in range(1, subprob_cnt):
            curr_subprob = temp_subprobs[idx]
            curr_dev = devs[curr_subprob]
            prev_subprob = temp_subprobs[idx - 1]
            prev_dev = devs[prev_subprob]
            if abs(prev_dev - curr_dev) > tol:
                subprobs.append(curr_subprob)
                    
        # Determine the total number of sub-problems to be considered.
        subprob_cnt = len(temp_subprobs)

    return instance_dict, subprobs

def get_subprob_data(subprob, instance_dict):
    # Get the data of one specific (deterministic) bilevel sub-problem.
    subprob_data = {}
    for key in instance_dict:
        if 'profit' not in key:
            subprob_data[key] = instance_dict[key]
        subprob_data['profits'] = instance_dict['modified profits'][subprob]
    return subprob_data

def solve_bilevel_subprob(subprob, subprob_data):
    # Solve one deterministic bilevel sub-problem.
    model = GeneralDeterministicModel(subprob_data)
    results_dict = model.solve()
    return results_dict

def solve_follower_subprob(subprob, leader_sol, instance_dict):
    # Solve one deterministic lower-level sub-problem.
    sol, obj, node_cnt = solve_lower_level(
        leader_sol,
        instance_dict['modified profits'][subprob],
        instance_dict['follower weights'],
        instance_dict['follower budget'],
    )

    # Add constant term from robustification to the objective.
    obj -= (instance_dict['gamma']
            *instance_dict['deviations'][subprob])
    return sol, obj

def solve_gamma_robust_lower_level(leader_sol, instance_dict, subprobs):
    follower_sols = []
    follower_objs = []
    for subprob in subprobs:
        # Solve the deterministic lower-level sub-problem.
        sol, obj, node_cnt = solve_lower_level(
            leader_sol,
            instance_dict['modified profits'][subprob],
            instance_dict['follower weights'],
            instance_dict['follower budget'],
        )
        
        # Add constant term from robustification to the objective.
        obj -= (instance_dict['gamma']
                *instance_dict['deviations'][subprob])

        # Update.
        follower_sols.append(sol)
        follower_objs.append(obj)

    # Determine Gamma-robust response of the follower.
    max_idx = np.argmax(follower_objs)
    rob_val = follower_objs[max_idx]
    rob_sol = follower_sols[max_idx]
    return rob_sol, rob_val

def check_feasibility(instance_file, conservatism, uncertainty,
                      deviations, tol=1e-04):
    instance_dict, subprobs = build_robustified_instance(
        instance_file,
        conservatism,
        uncertainty,
        deviations
    )
    size = instance_dict['size']
    gamma = instance_dict['gamma']

    for subprob in subprobs:
        # Solve a bilevel sub-problem unless the problem as already
        # been solved.
        out_file = 'counterexample/counterexample_{}.json'.format(subprob)
        if os.path.exists(out_file):
            results = open(out_file)
            results_dict = json.load(results)
        else:
            subprob_data = get_subprob_data(subprob, instance_dict)
            results_dict = solve_bilevel_subprob(subprob, subprob_data)
            with open(out_file, 'w') as out_file:
                json.dump(results_dict, out_file, indent=4)

        leader_sol = results_dict['leader decision']
        follower_sol = results_dict['follower decision']

        # Determine the lower-level objective function value.
        deviation = instance_dict['deviations'][subprob]
        profits = instance_dict['modified profits'][subprob]
        follower_obj = (-gamma*deviation
                        + sum(profits[idx]*follower_sol[idx]
                              for idx in range(size)))

        # Compute the Gamma-robust lower-level objective function value.
        rob_sol, rob_val = solve_gamma_robust_lower_level(
            leader_sol,
            instance_dict,
            subprobs
        )

        # Check whether the current solution to the deterministic bilevel
        # sub-problem is feasible for the original problem.
        if abs(rob_val - follower_obj) < tol:
            print('Solution to sub-problem {} is feasible!'.format(subprob))
            return True

    print('None of the solutions to the sub-problems is feasible!')
    return False

if __name__ == "__main__":
    instance_file = 'counterexample/counterexample.txt'
    uncertainty = 0.1
    conservatism = 0.5
    deviations = None

    # Check whether there exists a deterministic bilevel sub-problem
    # solved in the primal heuristic for Gamma-robust bilevel problems that
    # produces a feasible point for the original problem.
    is_feasible = check_feasibility(
        instance_file,
        conservatism,
        uncertainty,
        deviations
    )

