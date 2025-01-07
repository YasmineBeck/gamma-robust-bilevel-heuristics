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
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Local imports
from src.help_functions import *

def interdiction_cuts_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        leader_vars = [None]*model._size
        current_follower_obj = 0
        for idx in range(model._size):
            leader_vars[idx] = model.cbGetSolution(model._leader_vars[idx])
            follower_var = model.cbGetSolution(model._follower_vars[idx])
            current_follower_obj += model._profits[idx]*follower_var

        # Solve the lower-level problem.
        follower_sol, follower_obj, nodes = solve_lower_level(
            leader_vars,
            model._profits,
            model._follower_weights,
            model._follower_budget
        )
        
        assert follower_sol is not None, 'Cannot solve lower level!'

        # Complete the follower's decision to a maximal packing.
        follower_sol = make_maximal(
            follower_sol,
            model._profits,
            model._follower_weights,
            model._follower_budget
        )

        # Add a cut.
        coeffs = np.multiply(model._profits, follower_sol)
        if current_follower_obj + model._tol < follower_obj:
            model.cbLazy(
                gp.quicksum(
                    model._profits[idx]*model._follower_vars[idx]
                    - coeffs[idx]*(1 - model._leader_vars[idx])
                    for idx in range(model._size)
                )
                >= 0
            )

class GeneralDeterministicModel:
    """
    Class to solve generalized knapsack interdiction problems
    in which the upper- and lower-level objective function
    coefficients for the follower's variables may differ.
    """
    def __init__(self, instance_data, time_limit=3600):
        self.profits = instance_data['profits']
        self.leader_costs = instance_data['leader costs']
        self.leader_weights = instance_data['leader weights']
        self.leader_budget = instance_data['leader budget']
        self.follower_costs = instance_data['follower costs']
        self.follower_weights = instance_data['follower weights']
        self.follower_budget = instance_data['follower budget']
        self.size = len(self.profits)
        self.time_limit = time_limit
        self.tol = 1e-04

    def solve(self):
        # Suppress Gurobi output.
        env = gp.Env(empty=True)
        env.setParam('OutputFlag',0)
        env.start()

        # Build the model.
        model = gp.Model('deterministic model', env=env)
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

        # Add upper-level budget constraint.
        model.addConstr(
            gp.quicksum(self.leader_weights[idx]*leader_vars[idx]
                        for idx in range(self.size))
            <= self.leader_budget
        )

        # Add lower-level budget constraint.
        model.addConstr(
            gp.quicksum(self.follower_weights[idx]*follower_vars[idx]
                        for idx in range(self.size))
            <= self.follower_budget
        )

        # Add interdiction constraints.
        for idx in range(self.size):
            model.addConstr(follower_vars[idx] <= 1 - leader_vars[idx])

        # Prepare for callback.
        model._leader_vars = leader_vars
        model._follower_vars = follower_vars
        model._size = self.size
        model._profits = self.profits
        model._follower_weights = self.follower_weights
        model._follower_budget = self.follower_budget
        model._tol = self.tol

        model.Params.LazyConstraints = 1
        model.Params.TimeLimit = self.time_limit

        # Solve the model.
        model.optimize(interdiction_cuts_callback)

        # Extract solution.
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            results_dict = {
                'objective': model.objVal,
                'runtime': model.Runtime,
                'node count': model.getAttr('NodeCount')
            }
            try:
                results_dict['optimality gap'] = model.MIPGap
                results_dict['leader decision']\
                    = [var.x for var in leader_vars.values()]
                results_dict['follower decision']\
                    = [var.x for var in follower_vars.values()]
                return results_dict
            except:
                return results_dict
        else:
            return None
