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
import json
import os
import subprocess

class CombModel(object):
    """
    Class for solving an instance of the bilevel knapsack interdiction
    problem using the combinatorial solver by Fukasawa and Weninger (2023).
    """
    def __init__(self,
                 path_to_solver,
                 instance_file,
                 out_file,
                 instance_dict=None,
                 time_limit=3600):
        self.path_to_solver = path_to_solver
        self.out_file = out_file
        self.instance_file = instance_file
        self.tol = 1e-06
        self.time_limit = time_limit

        if instance_file is None and instance_dict is None:
            raise ValueError('No instance data given!')

        if os.path.exists(self.instance_file):
            with open(self.instance_file, 'r') as file:
                scaling_factor = 1
                for line in file:
                    if 'scaling factor' in line:
                        scaling_factor = int(line.split(' ')[-1].strip())
                        break
                self.scaling_factor = scaling_factor
        
        if (not os.path.exists(self.instance_file)
            and instance_dict is not None):
            # To use the solver, all instance data must be integer.
            # If the data is not integer, we scale the data accordingly.
            self.build_integer_instance(instance_dict)
            self.build_instance_file()

    def make_integer(self, data, scaling_factor=1):
        data = round(data, 5)
        scaled_data = data
        if not isinstance(data, int):
            if abs(data - int(data)) < self.tol:
                scaled_data = int(data)
            else:
                scaling_factor = 10**len(str(data).split('.')[1])
                scaled_data = int(scaling_factor*data)
        return scaled_data, scaling_factor

    def build_integer_instance(self, instance_dict):
        if not isinstance(instance_dict['size'], int):
            raise ValueError('Instance size must be integer!')
        self.size = instance_dict['size']

        profits = instance_dict['profits']
        profit_scaling_factor = 1
        for item in range(self.size):
            scaled_profit, scaling_factor = self.make_integer(profits[item])
            profits\
                = [scaling_factor*profits[idx] for idx in range(self.size)]
            profits[item] = scaled_profit
            profit_scaling_factor *= scaling_factor
        self.profits = [int(profits[idx]) for idx in range(self.size)]
        self.scaling_factor = profit_scaling_factor
        
        players = ['leader', 'follower']
        for player in players:
            budget_key = '{} budget'.format(player)
            weights_key = '{} weights'.format(player)
            
            scaled_budget, scaling_factor\
                = self.make_integer(instance_dict[budget_key])
            scaled_weights = [scaling_factor*instance_dict[weights_key][idx]
                              for idx in range(self.size)]
            
            constr_scaling_factor = 1
            for item in range(self.size):
                scaled_weight, scaling_factor\
                    = self.make_integer(instance_dict[weights_key][item])
                scaled_weights = [scaling_factor*scaled_weights[idx]
                                  for idx in range(self.size)]
                scaled_weights[item] = scaled_weight
                constr_scaling_factor *= scaling_factor
            scaled_weights = [int(scaled_weights[idx]) for idx in range(self.size)]
            scaled_budget = constr_scaling_factor*scaled_budget
            
            if player == 'leader':
                self.leader_budget = scaled_budget
                self.leader_weights = scaled_weights
            else:
                self.follower_budget = scaled_budget
                self.follower_weights = scaled_weights

    def build_instance_file(self):
        """Build a text file with extension .ki containing 7 lines of data
        to give to the bkpsolver."""
        lines = []

        # First line contains the number of items.
        lines.append(str(self.size))

        # Second line contains the lower-level knapsack capacity, i.e.,
        # the follower's budget.
        lines.append(str(self.follower_budget))

        # Third line contains the upper-level knapsack capacity, i.e.,
        # the leader's budget.
        lines.append(str(self.leader_budget))

        # Fourth line contains the lower-level/follower's weights.
        lines.append(' '.join(str(weight) for weight in self.follower_weights))
        
        # Fifth line contains the upper-level/leader's weights.
        lines.append(' '.join(str(weight) for weight in self.leader_weights))
        
        # Sixth line contains the profits.
        lines.append(' '.join(str(profit) for profit in self.profits))

        # Seventh line contains the scaling factor applied to obtain
        # integer-valued profits.
        lines.append('scaling factor {}'.format(self.scaling_factor))
        
        # Last line contains the instance name.
        lines.append(self.instance_file)

        with open(self.instance_file, 'w+') as file:
            for line in lines:
                file.write(line)
                file.write('\n')
        file.close()

    def write_results(self):
        results_dict = {}

        with open(self.out_file, 'r') as file:
            for line in file:
                # Extract number of nodes.
                if 'nodes' in line:
                    nodes = int(line.split(' ')[-1].strip())
                    results_dict['node count'] = nodes

                # Extract runtime.
                if 'total_time' in line:
                    time = float(line.split(' ')[-1].strip())
                    # Account for runtime results being given in milliseconds.
                    results_dict['runtime'] = time/1000.0

                # Extract objective function value.
                if 'profit' in line:
                    obj = float(line.split(' ')[-1].strip())
                    # Account for scaling of profits if they are not integer.
                    results_dict['objective'] = (1.0/self.scaling_factor)*obj

                # Extract upper (!) bound value (dp_lb).
                if 'dp_lb' in line and 'dp_lb_time' not in line:
                    bound = float(line.split(' ')[-1].strip())
                    # Account for scaling of profits if they are not integer.
                    results_dict['upper bound']\
                        = (1.0/self.scaling_factor)*bound

                # Extract upper-level/leader's decision.
                if 'upper' in line:
                    leader_str = line.split(' ')[-1].strip()
                    leader_decision = [int(idx) for idx in leader_str]
                    results_dict['leader decision'] = leader_decision
                    
                # Extract lower-level/follower's decision.
                if 'lower' in line:
                    follower_str = line.split(' ')[-1].strip()
                    follower_decision = [int(idx) for idx in follower_str]
                    results_dict['follower decision'] = follower_decision

                # Extract greedy upper bound.
                if 'greedy_ub' in line:
                    greedy_bound = float(line.split(' ')[-1].strip())
                    # Account for scaling of profits if they are not integer.
                    results_dict['greedy bound']\
                        = (1.0/self.scaling_factor)*greedy_bound

        # Get DCS lower bound in case DP lower bound is not given.
        if 'upper bound' not in results_dict:
            with open(self.out_file, 'r') as file:
                for line in file:                    
                    if 'dcs_lb' in line:
                        dcs_bound = float(line.split(' ')[-1].strip())
                        # Account for scaling of profits if they are
                        # not integer.
                        results_dict['upper bound']\
                            = (1.0/self.scaling_factor)*dcs_bound
            
        if 'node count' not in results_dict:
            # Problem has been solved at the root node.
            results_dict['node count'] = 0

        if 'objective' in results_dict and 'upper bound' in results_dict:
            obj = results_dict['objective']
            bound = results_dict['upper bound']
            gap = abs(obj - bound)/(abs(bound) + self.tol)
            if 'greedy bound' in results_dict:
                greedy_bound = results_dict['greedy bound']
                if greedy_bound - self.tol <= obj:
                    gap = 0
            if 'runtime' in results_dict:
                if results_dict['runtime'] < self.time_limit:
                    # Confirmed by Weninger: if algorithm terminates,
                    # the instance is solved to global optimality.
                    gap = 0
            results_dict['optimality gap'] = gap
            
        if len(results_dict) == 0:
            return None        
        return results_dict
        
    def run(self, results_dict=None):
        cmd = '{} comb -p -l0 -q {} > {}'.format(
            self.path_to_solver,
            self.instance_file,
            self.out_file
        )
        subprocess.run(cmd, shell=True)
        if os.path.exists(self.out_file):
            results_dict = self.write_results()
        return results_dict
