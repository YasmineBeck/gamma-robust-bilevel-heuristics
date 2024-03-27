# This file is part of the code used for the computational study
# in the paper
#
#     "Heuristic Methods for Mixed-Integer, Linear,
#      and Gamma-Robust Bilevel Problems"
#
# by Yasmine Beck, Ivana Ljubic, and Martin Schmidt (2024).

# Global imports
import argparse
import json
import logging
import numpy as np
import os
import subprocess
from time import time

# Local imports
from combinatorial_approach import CombModel
from min_max_deterministic_model import DeterministicModel
from help_functions import solve_lower_level
from instance_data_builder import InstanceDataBuilder
from optimality_checker import OptimalityChecker

# Initialize logger.
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MinMaxHeuristic:
    """
    Class for the primal heuristic for mixed-integer, linear
    min-max problems with a Gamma-robust follower.
    """
    def __init__(self,
                 instance_file,
                 output_file,
                 conservatism,
                 uncertainty,
                 deviations,
                 solver):
        self.instance_file = instance_file
        self.output_file = output_file
        self.conservatism = conservatism
        self.uncertainty = uncertainty
        self.deviations = deviations
        self.tol = 1e-06

        # Two options for the solution of the deterministic bilevel problems:
        # 1. combinatorial approach (bkpsolver) 'bkp'
        #    by Fukasawa and Weninger (2023) (default)
        # 2. branch-and-cut approach using interdiction cuts 'ic'
        #    based on Fischetti et al. (2019)
        if not isinstance(solver, str):
            raise TypeError('Solver must be specified by a string.')
        elif solver not in ['bkp', 'ic']:
            raise ValueError('Invalid solver: Choose from "bkp" or "ic".')
        self.solver = solver

        if solver == 'bkp':
            # Determine the path to the bkpsolver.
            path_to_bkpsolver = '../bkpsolver/build/bkpsolver'
            if os.path.exists(path_to_bkpsolver):
                self.path_to_bkpsolver = path_to_bkpsolver
            else:
                # Avoid going here, adapt the path to bkpsolver accordingly!
                cmd = 'find ~/ -type f -name bkpsolver'
                out = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
                lines = out.stdout.decode().splitlines()
                for idx, line in enumerate(lines):
                    if 'build/bkpsolver' in line:
                        self.path_to_bkpsolver = lines[idx]

        # Build a robustified instance.
        self.build_robustified_instance()
                    
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

    def presolve(self, subprob_data):
        # Eliminate items 1) due to negative (modified) profits and
        # 2) for which the associated weight exceeds the budget;
        # see Pisinger and Toth (1998).
        LOGGER.debug('Entering presolve...')
        size = subprob_data['size']
        profits = subprob_data['profits']
        ll_weights = subprob_data['follower weights']
        ll_budget = subprob_data['follower budget']
        ul_weights = subprob_data['leader weights']
        ul_budget = subprob_data['leader budget']

        fix_leader = []
        fix_both = []
        for idx in range(size):
            if ((profits[idx] < 0) or (ll_weights[idx] >= ll_budget)):
                fix_both.append(idx)
                subprob_data['size'] -= 1
            if ((ul_weights[idx] >= ul_budget) and (idx not in fix_both)):
                fix_leader.append(idx)

        keys = ['profits', 'leader weights', 'follower weights']
        for key in keys:
            subprob_data[key]\
                = np.asarray([subprob_data[key][idx]
                              for idx in range(size) if idx not in fix_both])
                
        return subprob_data, fix_leader, fix_both

    def get_file_paths(self, subprob):
        # Generate the paths to the instance data and the output files.
        instance_dir = os.path.dirname(os.path.abspath(self.instance_file))
        instance = os.path.splitext(os.path.basename(self.output_file))[0]
        data_file = '{}/{}_{}.ki'.format(instance_dir, instance, subprob)

        if self.solver == 'bkp':
            output_dir = 'bkp-log-files/'
        else:
            output_dir = 'ic-log-files/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        log_file = '{}/{}_{}.log'.format(output_dir, instance, subprob)
        out_file = '{}/{}_{}.json'.format(output_dir, instance, subprob)
        return data_file, log_file, out_file

    def solve_subprob(self, subprob, subprob_data, fix_leader):
        # Solve one deterministic bilevel sub-problem.
        data_file, log_file, out_file = self.get_file_paths(subprob)
        if subprob_data['size'] < 1:
            # The problem has already been solved during presolve.
            LOGGER.debug('Sub-problem has been solved during presolve.')
            results_dict = {
                'objective': 0.0,
                'leader decision': [0]*self.instance_dict['size'],
                'runtime': 0.0,
                'node count': 0,
                'optimality gap': 0.0
                }

        elif subprob_data['size'] == len(fix_leader):
            # The problem can be solved as a standard knapsack problem
            # since presolve fixed all variables of the leader to zero.
            LOGGER.debug('Solving sub-problem as single-level problem...')
            profits = subprob_data['profits']
            weights = subprob_data['follower weights']
            budget = subprob_data['follower budget']
            timer = time()
            var, obj, nodes = solve_lower_level([], profits, weights, budget)
            runtime = time() - timer

            if obj is None:
                results_dict = None
            else:
                results_dict = {
                    'objective': obj,
                    'leader decision': [0]*self.instance_dict['size'],
                    'runtime': runtime,
                    'node count': nodes,
                    'optimality gap': 0.0
                }

        else:
            # Solve a deterministic bilevel problem (of reduced size).
            if self.solver == 'bkp' and fix_leader:
                # bkpsolver requires that the leader's weights do not exceed
                # the leader's budget, which is violated.
                # The solver option is changed to the branch-and-cut approach.
                LOGGER.info('Requirements of bkp violated, using ic instead.')
                self.solver = 'ic'
                data_file, log_file, out_file = self.get_file_paths(subprob)
                
            if self.solver == 'bkp':
                LOGGER.debug('Using bkpsolver...')
                model = CombModel(
                    self.path_to_bkpsolver,
                    data_file,
                    log_file,
                    subprob_data
                )
                results_dict = model.run()
            else:
                LOGGER.debug('Using branch-and-cut approach...')
                model = DeterministicModel(subprob_data)
                results_dict = model.solve()

        # Adapt objective values: add constant term from robustification.
        keys = ['objective', 'upper bound', 'greedy bound', 'root relaxation']
        for key in keys:
            if key in results_dict:
                if results_dict[key] is not None:
                    results_dict[key]\
                        -= (self.instance_dict['gamma']
                            *self.instance_dict['deviations'][subprob])

        with open(out_file, 'w') as out_file:
            json.dump(results_dict, out_file, indent=4)
        return results_dict

    def extract_time_and_nodes(self, max_subprob_idx, nodes_lst):
        # Extract runtime and node count data.
        times = []
        # Get the number of nodes considered for solving lower-level problems.
        nodes = sum(cnt for cnt in nodes_lst if cnt is not None)
        for idx in range(max_subprob_idx + 1):
            subprob = self.subprobs[idx]
            data_file, log_file, out_file = self.get_file_paths(subprob)
            results_file = open(out_file,)
            results_dict = json.load(results_file)
            times.append(results_dict['runtime'])
            # Add the nodes considered for solving the bilevel sub-problem.
            nodes += results_dict['node count']
        runtime = sum(times)
        ideal_runtime = max(times)
        return runtime, ideal_runtime, nodes

    def extract_solution(self, subprob, presolved):
        # Extract the solution and the optimal objective function value of
        # a specific bilevel sub-problem.
        data_file, log_file, out_file = self.get_file_paths(subprob)
        results_file = open(out_file,)
        results_dict = json.load(results_file)
        try:
            subprob_sol = results_dict['leader decision']
            size = self.instance_dict['size']
            sol = [0]*size
            cnt = 0
            for idx in range(size):
                if idx not in presolved:
                    sol[idx] = subprob_sol[cnt]
                    cnt += 1
                    
            if results_dict['optimality gap'] < self.tol:
                obj = results_dict['objective']
                return sol, obj
            else:
                # The problem has not been solved to global optimality.
                # Still, the leader's decision is feasible for the overall
                # problem and can thus be used to compute a valid upper bound.
                return sol, -np.inf
        except:
            # The problem could not be solved.
            return None, -np.inf
        
    def alternating_heuristic(self):
        # Alternate between solving bilevel and lower-level problems.
        single_level_times = []
        ideal_single_level_times = []
        sols = []
        solved_ll = 0
        node_lst = []
        lb = -np.inf
        ub = np.inf
        idx = 0
        start_time = time()
        while idx < self.subprob_cnt:
            # Solve a bilevel sub-problem.
            subprob = self.subprobs[idx]
            subprob_data = self.get_subprob_data(subprob)
            subprob_data, fix_leader, fix_both = self.presolve(subprob_data)
            raw_results_dict\
                = self.solve_subprob(subprob, subprob_data, fix_leader)

            if raw_results_dict is None:
                # The bilevel sub-problem could not be solved.
                idx += 1
                continue
            
            subprob_sol, subprob_obj\
                = self.extract_solution(subprob, fix_both)

            if subprob_sol is None:
                # The bilevel sub-problem could not be solved.
                idx += 1
                continue
            
            # Update the lower bound.
            LOGGER.debug('Updating lower bound...')
            lb = max(lb, subprob_obj)

            # Terminate if the gap is closed.
            gap = ub - lb
            if abs(gap) < self.tol:
                idx += 1                
                break

            # Only solve lower-level problems if the solution to the current
            # bilevel sub-problem differs from those already considered and
            # is not dominated; see Theorems 2 and 3.
            checker = OptimalityChecker(self.instance_dict, self.subprobs)
            if (subprob_sol not in sols
                and not checker.is_dominated(subprob_sol, sols)):
                sols.append(subprob_sol)
                single_level_timer = time()
                subprob_ub, node_cnt, times\
                    = checker.compute_upper_bound(subprob_sol, subprob)
                elapsed = time() - single_level_timer
                subprob_ub = max(subprob_ub, subprob_obj)
                single_level_times.append(elapsed)
                ideal_time = elapsed - sum(times) + max(times)
                ideal_single_level_times.append(ideal_time)
                node_lst.append(node_cnt)
                solved_ll += self.subprob_cnt - 1
                
                # Update.
                if subprob_ub - self.tol < ub:
                    LOGGER.debug('Updating upper bound...')
                    ub = subprob_ub
                    sol = subprob_sol
                    best_subprob = subprob

            # Terminate if the gap is closed.
            gap = ub - lb
            if abs(gap) < self.tol:
                best_subprob = subprob
                idx += 1                
                break
            else:
                idx += 1

        if abs(gap) > self.tol:
            # Check for a dominating solution.
            dom_sol, dominating = checker.dominating_solution(sols)
            if dominating:
                sol = dom_sol
                gap = 0.0
                ub = lb
            
        runtime = time() - start_time
        single_level_time = sum(single_level_times)
        ideal_single_level_time = sum(ideal_single_level_times)

        # Extract results.
        LOGGER.debug('Extracting results...')
        results_dict = {}
        results_dict['runtime'] = runtime
        results_dict['ideal runtime']\
            = runtime - single_level_time + ideal_single_level_time
        results_dict['single-level time'] = single_level_time
        results_dict['ideal single-level time'] = ideal_single_level_time
        bilevel_time, ideal_bilevel_time, nodes\
            = self.extract_time_and_nodes(idx - 1, node_lst)
        results_dict['bilevel time'] = bilevel_time
        results_dict['bilevel problems considered']\
            = '{} of {}'.format(idx, self.subprob_cnt)
        results_dict['single-level problems solved'] = solved_ll
        results_dict['total node count'] = nodes
        results_dict['gap'] = gap
        if abs(gap) < np.inf:
            # A feasible leader's decision has been found.
            results_dict['best subprob'] = best_subprob
            results_dict['leader decision'] = sol
            if abs(gap) < self.tol:
                results_dict['status'] = 'optimal'
                results_dict['objective'] = ub
            else:
                results_dict['status'] = 'feasible'
                results_dict['lower bound'] = lb
                results_dict['upper bound'] = ub
        else:
            # None of the bilevel sub-problems has been solved.
            results_dict['status'] = 'unsolved'
        return results_dict
    
    def modified_heuristic(self):
        # Solve all bilevel sub-problems and, afterward,
        # perform a correction step by solving lower-level sub-problems.
        sols = []
        objs = []
        start_time = time()
        
        # Solve all bilevel sub-problems.
        for subprob in self.subprobs:
            subprob_data = self.get_subprob_data(subprob)
            subprob_data, fix_leader, fix_both = self.presolve(subprob_data)
            raw_results_dict\
                = self.solve_subprob(subprob, subprob_data, fix_leader)

            if raw_results_dict is None:
                # The bilevel sub-problem could not be solved.
                idx += 1
                continue
            
            sol, obj = self.extract_solution(subprob, fix_both)
            
            if sol is None:
                # The bilevel sub-problem could not be solved.
                idx += 1
                continue
            
            sols.append(sol)
            objs.append(obj)

        if objs:
            # Update the lower bound.
            LOGGER.debug('Updating lower bound...')
            lb = max(objs)

            # Check for ex-post optimality and compute an upper bound.
            optimality_check_timer = time()
            checker = OptimalityChecker(self.instance_dict, self.subprobs)
            check_dict = checker.ex_post_checks(sols, objs, lb)
            optimality_check_time = time() - optimality_check_timer
        else:
            # None of the bilevel sub-problems has been solved.
            optimality_check_time = 0
            
        runtime = time() - start_time

        # Extract results.
        LOGGER.debug('Extracting results...')
        results_dict = {}
        results_dict['runtime'] = runtime
        bilevel_time, ideal_bilevel_time, nodes\
            = self.extract_time_and_nodes(self.subprob_cnt - 1, [0])
        ideal_runtime = runtime - bilevel_time + ideal_bilevel_time
        results_dict['bilevel time'] = bilevel_time
        results_dict['ideal bilevel time'] = ideal_bilevel_time
        results_dict['total node count'] = nodes
        results_dict['time for ex-post checks'] = optimality_check_time
        
        if objs:
            if 'node count' in check_dict:
                results_dict['total node count'] += check_dict['node count']

            if 'single-level time' in check_dict:
                seq_time = check_dict['single-level time']
                ideal_time = check_dict['ideal single-level time']
                ideal_runtime +=  ideal_time - seq_time
                results_dict['single-level time'] = seq_time
                results_dict['ideal single-level time'] = ideal_time
            results_dict.update(check_dict)
        else:
            results_dict['status'] = 'unsolved'
        results_dict['ideal runtime'] = ideal_runtime
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
    parser.add_argument('--solver', default='bkp', help='The solver to be used: "bkp" or "ic".')
    parser.add_argument('--output_file', required=True,
                        help='The file to write the output to.')
    parser.add_argument('--modify', default='False',
                        help='Use the modified heuristic (True) or not (False). Default is False.')
    arguments = parser.parse_args()

    instance_file = arguments.instance_file
    conservatism = arguments.conservatism
    uncertainty = arguments.uncertainty
    deviations = arguments.deviations
    solver = arguments.solver
    output_file = arguments.output_file
    modify = arguments.modify

    if modify not in ['True', 'False']:
        raise TypeError('Use of the modified heuristic must be specified by Boolean.')

    model = MinMaxHeuristic(
        instance_file,
        output_file,
        conservatism,
        uncertainty,
        deviations,
        solver
    )

    if 'True' in modify:
        results_dict = model.modified_heuristic()
    else:
        results_dict = model.alternating_heuristic()

    with open(output_file, 'w') as out_file:
        json.dump(results_dict, out_file, indent=4)
