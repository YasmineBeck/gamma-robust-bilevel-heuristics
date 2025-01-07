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
import gurobipy as gp
from gurobipy import GRB
import json
import numpy as np
from time import time

# Local imports
from src.instance_data_builder import InstanceDataBuilder

class IterateHeuristic:
    """
    Class for solving generalized Gamma-robust knapsack
    interdiction problems using the ITERATE heuristic 
    proposed in Fischetti et al. (2018).
    The ONE-SHOT variant of the method is also covered.
    """
    def __init__(self,
                 instance_file,
                 output_file,
                 conservatism,
                 uncertainty,
                 deviations,
                 one_shot,
                 time_limit):
        self.instance_file = instance_file
        self.output_file = output_file
        self.conservatism = conservatism
        self.uncertainty = uncertainty
        self.deviations = deviations
        self.time_limit = time_limit
        self.tol = 1e-06
        if not isinstance(one_shot, bool):
            if 'False' in one_shot:
                one_shot = False
            elif 'True' in one_shot:
                one_shot = True
            else:
                raise ValueError('ONE-SHOT variant must be specified by Boolean.')
        self.one_shot = one_shot

        # Build robustified instance data.
        builder = InstanceDataBuilder(
            self.instance_file,
            conservatism=self.conservatism,
            uncertainty=self.uncertainty,
            deviations=self.deviations
        )
        instance_data = builder.build_robustified_instance()
        self.profits = instance_data['profits']
        self.leader_costs = instance_data['leader costs']
        self.leader_weights = instance_data['leader weights']
        self.leader_budget = instance_data['leader budget']
        self.follower_costs = instance_data['follower costs']
        self.follower_weights = instance_data['follower weights']
        self.follower_budget = instance_data['follower budget']
        self.size = instance_data['size']
        self.devs = instance_data['deviations']
        self.gamma = instance_data['gamma']

    def solve_high_point_relaxation(self):
        """Build and solve the high-point relaxation of the bilevel problem."""
        model = gp.Model()
        leader_vars = model.addVars(self.size, vtype=GRB.BINARY)
        follower_vars = model.addVars(self.size, vtype=GRB.BINARY)

        # Set objective.
        model.setObjective(
            gp.quicksum(
                self.leader_costs[idx]*leader_vars[idx]
                + self.follower_costs[idx]*follower_vars[idx]
                for idx in range(self.size)
            ),
            GRB.MINIMIZE
        )

        # Add upper-level knapsack constraint.
        model.addConstr(
            gp.quicksum(self.leader_weights[idx]*leader_vars[idx]
                        for idx in range(self.size))
            <= self.leader_budget
        )

        # Add lower-level knapsack constraint.
        model.addConstr(
            gp.quicksum(self.follower_weights[idx]*follower_vars[idx]
                        for idx in range(self.size))
            <= self.follower_budget
        )

        # Add interdiction constraints.
        for idx in range(self.size):
            model.addConstr(follower_vars[idx] <= 1 - leader_vars[idx])

        # Optimize.
        model.Params.OutputFlag = False
        model.Params.TimeLimit = self.time_limit
        model.optimize()
        
        # Extract solution.
        nodes = model.getAttr('NodeCount')
        if model.status == GRB.OPTIMAL:
            return model.objVal, nodes
        else:
            return None, nodes

    def main(self):
        """Use the heuristic to obtain a bilevel-feasible point and
        an upper bound. A lower bound is obtained by solving the
        high-point relaxation of the bilevel problem."""
        self.lb = -np.inf
        self.ub = np.inf
        self.leader_sol = None
        self.follower_sol = None
        solved_subprobs = 0
        time_to_best = 0
        start_time = time()

        # Compute lower bound.
        val, nodes = self.solve_high_point_relaxation()
        self.node_cnt = nodes
        if val is not None:
            self.lb = val

        # Initialize MILP subproblem.
        self.build_milp_model()

        # Apply ITERATE (ONE-SHOT) heuristic.
        runtime = time() - start_time
        while runtime < self.time_limit:
            remaining_time = self.time_limit - runtime
            leader_sol, nodes = self.solve_subprob(0.95*remaining_time)
            self.node_cnt += nodes
            if leader_sol is None:
                break
            solved_subprobs += 1
            follower_sol, nodes = self.solve_lower_level(
                leader_sol,
                0.05*remaining_time
            )
            self.node_cnt += nodes
            if follower_sol is None:
                break
            solved_subprobs += 1
            obj = sum(self.leader_costs[idx]*leader_sol[idx]
                      + self.follower_costs[idx]*follower_sol[idx]
                      for idx in range(self.size))
            if obj + self.tol < self.ub:
                self.ub = obj
                self.leader_sol = leader_sol
                self.follower_sol = follower_sol
                time_to_best = time() - start_time
                
            # Terminate after one iteration if the ONE-SHOT variant
            # of the heuristic is applied.
            if self.one_shot:
                break
                
            self.add_cut(leader_sol)
            runtime = time() - start_time

        # Extract results.
        gap = abs(self.ub - self.lb)/abs(self.ub + self.tol)
        if abs(gap) < self.tol:
            status = 'optimal'
        else:
            status = 'feasible'

        if self.one_shot:
            variant = 'ONE-SHOT'
        else:
            variant = 'ITERATE'
        
        results_dict = {
            'runtime': time() - start_time,
            'time to best': time_to_best,
            'total node count': self.node_cnt,
            'status': status,
            'variant': variant,
            'solved subproblems': solved_subprobs,
            'lower bound': self.lb,
            'upper bound': self.ub,
            'optimality gap': gap,
            'leader decision': self.leader_sol,
            'follower decision': self.follower_sol
        }
        return results_dict

    def build_milp_model(self):
        """Build the initial MILP subproblem."""
        self.model = gp.Model()

        # Build variables.
        self.leader_vars = {}
        self.follower_vars = {}
        for idx in range(self.size):
            self.leader_vars[idx] = self.model.addVar(
                vtype=GRB.BINARY,
                name=f"leader_{idx}"
            )
            self.follower_vars[idx] = self.model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0
            )
            
        self.aux_vars = self.model.addVars(
            self.size + 1,
            vtype=GRB.CONTINUOUS
        )
        
        self.dual_vars = self.model.addVars(
            2*self.size + 1,
            vtype=GRB.CONTINUOUS
        )

        # Set objective.
        self.model.setObjective(
            gp.quicksum(
                self.leader_costs[idx]*self.leader_vars[idx]
                + self.follower_costs[idx]*self.follower_vars[idx]
                for idx in range(self.size)
            ),
            GRB.MINIMIZE
        )

        # Add upper-level knapsack constraint.
        self.model.addConstr(
            gp.quicksum(self.leader_weights[idx]*self.leader_vars[idx]
                        for idx in range(self.size))
            <= self.leader_budget
        )

        # Add lower-level knapsack constraint.
        self.model.addConstr(
            gp.quicksum(self.follower_weights[idx]*self.follower_vars[idx]
                        for idx in range(self.size))
            <= self.follower_budget
        )

        # Add interdiction constraints.
        for idx in range(self.size):
            self.model.addConstr(
                self.follower_vars[idx]
                <= 1 - self.leader_vars[idx]
            )

        # Add constraints for robustification.
        for idx in range(self.size):
            self.model.addConstr(
                self.aux_vars[idx] + self.aux_vars[self.size]
                >= self.devs[idx]*self.follower_vars[idx]
            )

        # Add dual feasibility constraints.
        for idx in range(self.size):
            self.model.addConstr(
                self.follower_weights[idx]*self.dual_vars[2*self.size]
                + self.devs[idx]*self.dual_vars[idx]
                + self.dual_vars[self.size + idx]
                >= self.profits[idx]*(1 - self.leader_vars[idx])
            )

            self.model.addConstr(self.dual_vars[idx] <= 1.0)

        self.model.addConstr(
            gp.quicksum(self.dual_vars[idx] for idx in range(self.size))
            <= self.gamma
        )

        # Add strong duality constraint.
        self.model.addConstr(
            - self.gamma*self.aux_vars[self.size]
            + gp.quicksum(
                self.profits[idx]*self.follower_vars[idx] - self.aux_vars[idx]
                for idx in range(self.size)
            )
            >= (self.follower_budget*self.dual_vars[2*self.size]
                + gp.quicksum(self.dual_vars[idx]
                              for idx in range(self.size, 2*self.size)))
        )

    def solve_subprob(self, time_limit):
        """Solve an MILP subproblem."""
        self.model.update()
        self.model.Params.TimeLimit = time_limit
        self.model.Params.OutputFlag = False
        self.model.optimize()
        nodes = self.model.getAttr('NodeCount')
        try:
            leader_sol = [var.x for var in self.leader_vars.values()]
            return leader_sol, nodes
        except:
            return None, nodes

    def solve_lower_level(self, leader_sol, time_limit):
        """Solve the MILP reformulation of the parameterized
        robustified lower-level problem."""
        model = gp.Model()
        follower_vars = model.addVars(self.size, vtype=GRB.BINARY)
        aux_vars = model.addVars(self.size + 1, vtype=GRB.CONTINUOUS)

        # Set objective.
        model.setObjective(
            - self.gamma*aux_vars[self.size]
            + gp.quicksum(
                self.profits[idx]*follower_vars[idx]
                - aux_vars[idx]
                for idx in range(self.size)
            ),
            GRB.MAXIMIZE
        )

        # Add knapsack constraint.
        model.addConstr(
            gp.quicksum(self.follower_weights[idx]*follower_vars[idx]
                        for idx in range(self.size))
            <= self.follower_budget
        )

        # Add interdiction constraints.
        for idx in range(self.size):
            model.addConstr(follower_vars[idx] <= 1 - leader_sol[idx])

        # Add constraints for robustification.
        for idx in range(self.size):
            model.addConstr(
                aux_vars[idx] + aux_vars[self.size]
                >= self.devs[idx]*follower_vars[idx]
            )

        # Optimize.
        model.Params.OutputFlag = False
        model.Params.TimeLimit = time_limit
        model.optimize()
        
        # Extract solution.
        nodes = model.getAttr('NodeCount')
        if model.status == GRB.OPTIMAL:
            follower_sol = [var.x for var in follower_vars.values()]
            return follower_sol, nodes
        else:
            return None, nodes
        
    def add_cut(self, leader_sol):
        """Add a no-good cut to the current subproblem to separate
        the previous leader's decision."""
        ones = [idx for idx in range(self.size) if leader_sol[idx] > 0.5]
        zeros = [idx for idx in range(self.size) if leader_sol[idx] < 0.5]

        # Retrieve leader variables.
        leader_vars = [self.model.getVarByName(f"leader_{idx}")
                       for idx in range(self.size)]

        # Add no-good cut.
        self.model.addConstr(
            gp.quicksum(leader_vars[idx] for idx in zeros)
            + gp.quicksum(1 - leader_vars[idx] for idx in ones)
            >= 1
        )
        self.model.update()

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
    parser.add_argument('--time_limit', type=float, default=3600,
                        help='The time limit (in s). Default is 3600s.')
    parser.add_argument('--one_shot', default='False',
                        help='Use the ONE-SHOT variant (True) or not (False). Default is False.')
    arguments = parser.parse_args()

    instance_file = arguments.instance_file
    conservatism = arguments.conservatism
    uncertainty = arguments.uncertainty
    deviations = arguments.deviations
    output_file = arguments.output_file
    time_limit = arguments.time_limit
    one_shot = arguments.one_shot

    model = IterateHeuristic(
        instance_file,
        output_file,
        conservatism,
        uncertainty,
        deviations,
        one_shot,
        time_limit
    )
    results_dict = model.main()
    with open(output_file, 'w') as out_file:
        json.dump(results_dict, out_file, indent=4)
