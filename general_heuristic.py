# This file is part of the code used for the computational study
# in the paper
#
#     "Heuristic Methods for Mixed-Integer, Linear,
#      and Gamma-Robust Bilevel Problems"
#
# by Yasmine Beck, Ivana Ljubic, and Martin Schmidt (2024).

# Global imports
import argparse
import gurobipy as gp
from gurobipy import GRB
import json
import numpy as np
import os
from time import time

# Local imports
from help_functions import solve_lower_level
from help_functions import solve_refinement_problem
from instance_data_builder import InstanceDataBuilder
from general_deterministic_model import GeneralDeterministicModel


class GeneralHeuristic:
    """
    Class for the primal heuristic for general mixed-integer, linear
    bilevel problems with a Gamma-robust follower.
    """
    def __init__(self,
                 instance_file,
                 output_file,
                 conservatism,
                 uncertainty,
                 deviations,
                 refine):
        self.instance_file = instance_file
        self.output_file = output_file
        self.conservatism = conservatism
        self.uncertainty = uncertainty
        self.deviations = deviations
        self.tol = 1e-06
        if not isinstance(refine, bool):
            if 'False' in refine:
                refine = False
            elif 'True' in refine:
                refine = True
            else:
                raise ValueError('Refinement must be specified by Boolean.')
        self.refine = refine

        # Set directory for log-files.
        self.log_dir = 'general-log-files'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Build a robustified instance.
        self.build_robustified_instance()

        # Initialize results data for solution of lower-level problems.
        self.follower_nodes = 0
        self.single_level_times = []
        self.ideal_single_level_times = []
        self.solved_single_level = 0
        if refine:
            self.refined = 0

    def build_robustified_instance(self):
        # Build a robustified instance from the nominal data, the specified
        # uncertainty, the given deviations, and the level of conservatism.
        builder = InstanceDataBuilder(
            self.instance_file,
            conservatism=self.conservatism,
            uncertainty=self.uncertainty,
            deviations=self.deviations
        )

        self.instance_dict = builder.build_robustified_instance()
        gamma = self.instance_dict['gamma']
        size = self.instance_dict['size']
        devs = self.instance_dict['deviations']

        # Determine the set of sub-problems to be solved.
        if gamma < 1:
            self.subprobs = [0]
        else:
            # Reduction of sub-problems according to Lee and Kwon (2014).
            max_idx = int(np.ceil((size - gamma)/2)) + 1
            subprobs = [gamma + 2*idx - 1 for idx in range(1, max_idx)]
            subprobs.append(size + 1)

            # Account for Python indexing starting at 0.
            subprob_cnt = len(subprobs)
            subprobs = [subprobs[idx] - 1 for idx in range(subprob_cnt)]

            # Only the sub-problems with pairwise distinct associated
            # deviations need to be considered, i.e., remove one of the
            # sub-problems with a duplicate value for the deviations
            # (see Proposition 3).
            self.subprobs = [subprobs[0]]
            for idx in range(1, subprob_cnt):
                curr_subprob = subprobs[idx]
                curr_dev = devs[curr_subprob]
                prev_subprob = subprobs[idx - 1]
                prev_dev = devs[prev_subprob]
                if abs(prev_dev - curr_dev) > self.tol:
                    self.subprobs.append(curr_subprob)
                    
        # Determine the total number of sub-problems to be considered.
        self.subprob_cnt = len(self.subprobs)

    def get_subprob_data(self, subprob):
        # Get the data of one specific (deterministic) bilevel sub-problem.
        subprob_data = {}
        for key in self.instance_dict:
            if 'profit' not in key:
                subprob_data[key] = self.instance_dict[key]
        subprob_data['profits']\
            = self.instance_dict['modified profits'][subprob]
        return subprob_data

    def solve_bilevel_subprob(self, subprob, subprob_data):
        # Solve one deterministic bilevel sub-problem.
        model = GeneralDeterministicModel(subprob_data)
        results_dict = model.solve()
        instance = os.path.splitext(os.path.basename(self.output_file))[0]
        out_file = '{}/{}_{}.json'.format(self.log_dir, instance, subprob)
        with open(out_file, 'w') as out_file:
            json.dump(results_dict, out_file, indent=4)
        return results_dict

    def solve_follower_subprob(self, subprob, leader_dec):
        # Solve one deterministic lower-level sub-problem.
        timer = time()
        sol, obj, node_cnt = solve_lower_level(
            leader_dec,
            self.instance_dict['modified profits'][subprob],
            self.instance_dict['follower weights'],
            self.instance_dict['follower budget'],
        )
        obj -= (self.instance_dict['gamma']
                *self.instance_dict['deviations'][subprob])
        subprob_time = time() - timer
        return sol, obj, node_cnt, subprob_time
    
    def extract_bilevel_times_and_nodes(self):
        # Extract runtime and node count data of the solved bilevel problems.
        times = []
        nodes = 0
        for idx, subprob in enumerate(self.subprobs):
            instance = os.path.splitext(os.path.basename(self.output_file))[0]
            out_file = '{}/{}_{}.json'.format(self.log_dir, instance, subprob)
            results_file = open(out_file,)
            results_dict = json.load(results_file)
            if results_dict:
                times.append(results_dict['runtime'])
                nodes += results_dict['node count']
        runtime = sum(times)
        ideal_runtime = max(times)
        return runtime, ideal_runtime, nodes

    def same_solution(self, sols):
        # Check if the same decision is chosen in every bilevel sub-problem.
        # Return True in the positive case, and False otherwise.
        for idx in range(1, len(sols)):
            if not all(abs(sol1 - sol2) < self.tol
                       for sol1, sol2 in zip(sols[0], sols[idx])):
                return False
        return True

    def check_dominance(self, follower_decs):
        # Check whether the dominance properties regarding the objective
        # function values in Theorem 5 are satisifed.
        # In the positive case, we return True, an optimal follower's
        # decision and the index of the "best" sub-problem.
        ul_vals = []
        ll_vals = []
        for subprob_idx, subprob in enumerate(self.subprobs):
            sol = follower_decs[subprob_idx]

            # Compute the upper-level objective term depending on
            # the follower's variables.
            ul_val = sum(self.instance_dict['follower costs'][idx]*sol[idx]
                         for idx in range(len(sol)))
            ul_vals.append(ul_val)

            # Compute the lower-level objective function value of
            # the current sub-problem.
            ll_val = self.compute_follower_value(subprob, sol)
            ll_vals.append(ll_val)

        for sol_idx, sol in enumerate(follower_decs):
            dominant = True
            subprob_idx = 0
            while dominant and subprob_idx < self.subprob_cnt:
                subprob = self.subprobs[subprob_idx]
                if sol_idx != subprob_idx:
                    if ul_vals[sol_idx] > ul_vals[subprob_idx] + self.tol:
                        dominant = False
                        continue
                    val = self.compute_follower_value(subprob, sol)
                    if val + self.tol < ll_vals[subprob_idx]:
                        dominant = False
                        continue
                subprob_idx += 1
                
            if dominant:
                # The dominance properties are satisifed.
                return True, sol, self.subprobs[sol_idx]
        return False, None, None

    def compute_follower_value(self, subprob, sol):
        # Compute the objective function value of a lower-level
        # sub-problem for a given follower's decision.
        size = self.instance_dict['size']
        gamma = self.instance_dict['gamma']
        deviation = self.instance_dict['deviations'][subprob]
        profits = self.instance_dict['modified profits'][subprob]
        val = (-gamma*deviation
               + sum(profits[idx]*sol[idx] for idx in range(size)))
        return val

    def compute_follower_costs(self, sol):
        # Compute the upper-level objective term depending on a given
        # follower's decision (follower's costs).
        size = self.instance_dict['size']
        val = sum(self.instance_dict['follower costs'][idx]*sol[idx]
                  for idx in range(size))
        return val

    def correct_and_refine(self, subprob_idx, leader_decs, follower_decs):
        # Compute a follower's response (optimistic approach) for a
        # given leader's decision.
        fix_subprob = self.subprobs[subprob_idx]
        fix_subprob_val = self.compute_follower_value(
            fix_subprob,
            follower_decs[subprob_idx]
        )
        
        # Fix the leader's decision and solve the parameterized
        # lower-level sub-problems.
        size = self.instance_dict['size']
        sol = leader_decs[subprob_idx]
        leader_costs = sum(self.instance_dict['leader costs'][idx]*sol[idx]
                           for idx in range(size))
        ul_objs = []
        ll_objs = []
        new_follower_decs = []
        times = []
        for subprob_idx, subprob in enumerate(self.subprobs):
            if subprob == fix_subprob:
                # No need to solve the fixed sub-problem (again).
                correction_time = 0
                refinement_time = 0
                follower_dec = follower_decs[subprob_idx]
                val = fix_subprob_val
                follower_costs = self.compute_follower_costs(follower_dec)
            else:
                # Check whether bilevel sub-problems have the same solution.
                diffs = [abs(sol[idx] - leader_decs[subprob_idx][idx])
                         for idx in range(size)]
                    
                if sum(diffs) < self.tol:
                    # Both sub-problems have the same solution, i.e., no
                    # need to solve the current lower-level sub-problem.
                    correction_time = 0
                    refinement_time = 0
                    follower_dec = follower_decs[subprob_idx]
                    val = self.compute_follower_value(subprob, follower_dec)
                    follower_costs = self.compute_follower_costs(follower_dec)
                else:
                    # Correction step: Solve the lower-level sub-problem.
                    follower_dec, val, nodes, correction_time\
                        = self.solve_follower_subprob(subprob, sol)
                    
                    # Update results.
                    self.follower_nodes += nodes
                    self.solved_single_level += 1
                    
                    if follower_dec is None:
                        # Lower level could not be solved, i.e.,
                        # we cannot compute a feasible pair.
                        times.append(correction_time)
                        self.single_level_times.append(sum(times))
                        self.ideal_single_level_times.append(max(times))
                        return None, np.inf, -np.inf
                    
                    follower_costs = self.compute_follower_costs(follower_dec)
                    
                    # Refinement step.
                    if not self.refine:
                        refinement_time = 0
                    else:
                        gamma = self.instance_dict['gamma']
                        deviation = self.instance_dict['deviations'][subprob]
                        const = gamma*deviation + val
                        refinement_timer = time()
                        follower_dec, obj, nodes = solve_refinement_problem(
                            sol,
                            self.instance_dict['follower costs'],
                            self.instance_dict['modified profits'][subprob],
                            const,
                            self.instance_dict['follower weights'],
                            self.instance_dict['follower budget']
                        )
                        refinement_time = time() - refinement_timer
                        
                        # Update results.
                        self.solved_single_level += 1
                        if obj + self.tol < follower_costs:
                            # Refinement improved upper-level objective value.
                            self.refined += 1
                        follower_costs = obj
                        
            # Update.
            ul_objs.append(leader_costs + follower_costs)
            ll_objs.append(val)
            new_follower_decs.append(follower_dec)
            times.append(correction_time + refinement_time)

        rob_val = max(ll_objs)
        self.single_level_times.append(sum(times))
        self.ideal_single_level_times.append(max(times))
                
        # Determine the sub-problems that yield the best objective function
        # value for the follower.
        candidates = [idx for idx in range(self.subprob_cnt)
                      if abs(rob_val - ll_objs[idx]) < self.tol]
        best_candidate = np.argmin([ul_objs[candidate]
                                    for candidate in candidates])
        best_idx = candidates[best_candidate]
        return new_follower_decs[best_idx], ul_objs[best_idx], rob_val
    
    def main(self):
        # Solve all bilevel sub-problems and, afterward, solve
        # lower-level sub-problems.
        lb = -np.inf
        ub = np.inf
        leader_sol = None
        follower_sol = None
        objs = []
        leader_decs = []
        follower_decs = []
        start_time = time()
        
        # Solve all bilevel sub-problems.
        for subprob in self.subprobs:
            subprob_data = self.get_subprob_data(subprob)
            results_dict = self.solve_bilevel_subprob(subprob, subprob_data)

            if results_dict is None:
                # The bilevel sub-problem could not be solved, i.e.,
                # we cannot compute a valid lower bound.
                return {'status': 'unsolved'}

            # Extract a solution.
            if ('optimality gap' in results_dict
                and 'leader decision' in results_dict):
                opt_gap = results_dict['optimality gap']
                if opt_gap > self.tol:
                    # The bilevel sub-problem could not be solved, i.e.,
                    # we cannot compute a valid lower bound.
                    return {'status': 'unsolved'}
                
                subprob_obj = results_dict['objective']
                leader_dec = results_dict['leader decision']
                follower_dec = results_dict['follower decision']
            else:
                # The bilevel sub-problem could not be solved, i.e.,
                # we cannot compute a valid lower bound.
                return {'status': 'unsolved'}

            objs.append(subprob_obj)
            leader_decs.append(leader_dec)
            follower_decs.append(follower_dec)

        # Update the lower bound.
        lb = min(objs)

        # Check for ex-post optimality, determine a feasible pair,
        # and compute an upper bound.
        results_dict = {}
        dominant = False
        if self.same_solution(leader_decs):
            dominant, follower_dec, best_subprob\
                = self.check_dominance(follower_decs)
            if dominant:
                # Optimality guarantee of Theorem 5 is satisfied.
                results_dict['status'] = 'optimal'
                results_dict['condition']= 'same solutions'
                results_dict['gap'] = 0.0
                results_dict['objective'] = lb
                results_dict['best subprob'] = best_subprob
                results_dict['leader decision'] = leader_decs[0]
                results_dict['follower decision'] = follower_dec
                
        if not dominant:
            # Sort the sub-problems such that the objective function values
            # are given in non-decreasing order and solve additional
            # lower-level problems.
            sorted_idxs = np.argsort(objs)
            for idx in sorted_idxs:
                sol, leader_obj, rob_obj = self.correct_and_refine(
                    idx,
                    leader_decs,
                    follower_decs
                )
                
                if sol is not None and leader_obj + self.tol < ub:
                    ub = leader_obj
                    leader_sol = leader_decs[idx]
                    follower_sol = sol
                    follower_obj = rob_obj
                    best_subprob = self.subprobs[idx]
                    
                # Terminate if the gap is closed.
                gap = ub - lb
                if abs(gap) < self.tol:
                    results_dict['status'] = 'optimal'
                    results_dict['leader objective'] = ub
                    break

            results_dict['condition']= 'solved lower level'
            results_dict['gap'] = gap
            results_dict['lower bound'] = lb
            results_dict['upper bound'] = ub
            results_dict['follower objective'] = follower_obj
            results_dict['best subprob'] = best_subprob
            results_dict['leader decision'] = leader_sol
            results_dict['follower decision'] = follower_sol
            if 'status' not in results_dict:
                results_dict['status'] = 'feasible'
            
        # Extract runtime results.
        runtime = time() - start_time
        results_dict['runtime'] = runtime
        bilevel_time, ideal_bilevel_time, nodes\
            = self.extract_bilevel_times_and_nodes()
        results_dict['bilevel time'] = bilevel_time
        results_dict['ideal bilevel time'] = ideal_bilevel_time
        results_dict['total node count'] = nodes + self.follower_nodes
        results_dict['single-level problems solved'] = self.solved_single_level
        if self.solved_single_level > 0:
            single_level_time = sum(self.single_level_times)
            ideal_single_level_time = sum(self.ideal_single_level_times)
        else:
            single_level_time = 0
            ideal_single_level_time = 0

        results_dict['single-level time'] = single_level_time
        results_dict['ideal single-level time'] = ideal_single_level_time
        ideal_runtime = (runtime - bilevel_time + ideal_bilevel_time
                         - single_level_time + ideal_single_level_time)
        results_dict['ideal runtime'] = ideal_runtime
        if self.refine:
            results_dict['refined subprobs'] = self.refined
        return results_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_file', required=True,
                        help='The file containing the nominal instance data as dictionary.')
    parser.add_argument('--conservatism', required=True, type=float,
                        help='Level of conservatism (in percent) must be between 0 and 1.')
    parser.add_argument('--uncertainty', type=float, default=None,
                        help='Uncertainty (in percent) must be between 0 and 1.')
    parser.add_argument('--deviations', nargs='+', type=float, default=None,
                        help='The deviations, e.g., 1 2 1 for a problem of size 3.')
    parser.add_argument('--output_file', required=True,
                        help='The file to write the output to.')
    parser.add_argument('--refine', default='True',
                        help='Include a refinement step (True) or not (False). Default is True.')
    arguments = parser.parse_args()

    instance_file = arguments.instance_file
    conservatism = arguments.conservatism
    uncertainty = arguments.uncertainty
    deviations = arguments.deviations
    output_file = arguments.output_file
    refine = arguments.refine

    model = GeneralHeuristic(
        instance_file,
        output_file,
        conservatism,
        uncertainty,
        deviations,
        refine
    )
    results_dict = model.main()

    with open(output_file, 'w') as outfile:
        json.dump(results_dict, outfile, indent=4)
