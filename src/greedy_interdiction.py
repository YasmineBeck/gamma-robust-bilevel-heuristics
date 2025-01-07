##################################################################
# This file is part of the code used for the computational study #
# in the paper                                                   #
#                                                                #
#  "Heuristic Methods for Gamma-Robust Mixed-Integer Linear      #
#   Bilevel Problems"                                            #
#                                                                #
# by Yasmine Beck, Ivana Ljubic, and Martin Schmidt (2025).      #
##################################################################

# Global imports
import argparse
import json
import numpy as np
from time import time

# Local imports
from src.instance_data_builder import InstanceDataBuilder
from src.help_functions import solve_lower_level

class GreedyInterdiction:
    """
    Class for determining a heuristic solution to
    the Gamma-robust knapsack interdiction problem
    using a greedy approach aking to the one in the
    PhD thesis by S. DeNegre (2011).
    """
    def __init__(self,
                 instance_file,
                 output_file,
                 conservatism,
                 uncertainty,
                 deviations):
        self.instance_file = instance_file
        self.output_file = output_file
        self.conservatism = conservatism
        self.uncertainty = uncertainty
        self.deviations = deviations
        self.tol = 1e-06

        # Build a robustified instance.
        self.build_robustified_instance()

    def build_robustified_instance(self):
        """Build a robustified instance from the nominal data,
        the specified uncertainty, the given deviations, and
        the level of conservatism."""
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

    def get_lower_bound(self):
        """Determine a lower bound for the optimal objective
        function value ("solve" the high-point relaxation)."""
        size = self.instance_dict['size']
        gamma = self.instance_dict['gamma']
        devs = self.instance_dict['deviations']
        profits = self.instance_dict['modified profits']
        self.lb = -np.inf
        for subprob in self.subprobs:
            lb = -gamma*devs[subprob]
            for idx in range(size):
                lb += min(0, profits[subprob][idx])
            if lb + self.tol > self.lb:
                self.lb = lb

    def get_upper_bound(self):
        """Determine a feasible interdiction policy and an
        upper bound for the optimal objective function value."""
        self.sol = self.get_greedy_interdiction()
        weights = self.instance_dict['follower weights']
        budget = self.instance_dict['follower budget']
        devs = self.instance_dict['deviations']
        gamma = self.instance_dict['gamma']
        self.ub = np.inf

        # Solve the Gamma-robust lower-level problem that is
        # parameterized in the greedy interdiction policy.
        self.node_cnt = 0
        self.runtimes = []
        objs = []
        for subprob in self.subprobs:
            timer = time()
            var, obj, nodes = solve_lower_level(
                self.sol,
                self.instance_dict['modified profits'][subprob],
                weights,
                budget
            )
            obj -= gamma*devs[subprob]
            objs.append(obj)
            self.runtimes.append(time() - timer)
            self.node_cnt += nodes
        self.ub = max(objs)

    def get_greedy_interdiction(self):
        """Determine an interdiction policy in a greedy way;
        cf. Algorithm 4.2 in the PhD thesis by S. DeNegre."""
        size = self.instance_dict['size']
        weights = self.instance_dict['leader weights']
        budget = self.instance_dict['leader budget']
        profits = self.instance_dict['modified profits'][-1]
        ordered_idxs = np.argsort(profits)

        # Initialize interdiction policy.
        var = [0]*size

        # Interdict in a greedy way.
        checked = []
        packed = []
        while len(checked) < size and budget > self.tol:
            check_idx = [idx for idx in ordered_idxs
                         if idx not in checked and idx not in packed][0]
            if budget - weights[check_idx] > self.tol:
                var[check_idx] = 1
                packed.append(check_idx)
                budget -= weights[check_idx]
            checked.append(check_idx)
        return var
    
    def main(self):
        """Run the greedy heuristic."""
        start_time = time()
        self.get_lower_bound()
        self.get_upper_bound()
        runtime = time() - start_time
        ideal_runtime = runtime - sum(self.runtimes) + max(self.runtimes)
        gap = abs(self.ub - self.lb)/abs(self.ub + self.tol)
        if abs(gap) < self.tol:
            status = 'optimal'
        else:
            status = 'feasible'
        
        results_dict = {
            'runtime': runtime,
            'ideal runtime': ideal_runtime,
            'total node count': self.node_cnt,
            'status': status,
            'lower bound': self.lb,
            'upper bound': self.ub,
            'optimality gap': gap,
            'leader decision': self.sol
        }
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
    arguments = parser.parse_args()

    instance_file = arguments.instance_file
    conservatism = arguments.conservatism
    uncertainty = arguments.uncertainty
    deviations = arguments.deviations
    output_file = arguments.output_file

    greedy = GreedyInterdiction(
        instance_file,
        output_file,
        conservatism,
        uncertainty,
        deviations
    )
    results_dict = greedy.main()
    with open(output_file, 'w') as out_file:
        json.dump(results_dict, out_file, indent=4)
