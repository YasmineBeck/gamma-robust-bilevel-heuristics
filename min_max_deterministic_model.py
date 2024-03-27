# This file is part of the code used for the computational study
# in the paper
#
#     "Heuristic Methods for Mixed-Integer, Linear,
#      and Gamma-Robust Bilevel Problems"
#
# by Yasmine Beck, Ivana Ljubic, and Martin Schmidt (2024).

# Global imports
import logging
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from time import time

# Local imports
from help_functions import *
    
def interdiction_cuts_callback(model, where):
        # Get root relaxation.
        if where == GRB.Callback.MIPNODE:
            node_cnt = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
            if (node_cnt < 1
                and (model.cbGet(GRB.Callback.MIPNODE_STATUS)
                     == GRB.Status.OPTIMAL)):
                    model._root_relax = model.cbGetNodeRel(model._aux_var)
                        
        if where == GRB.Callback.MIPSOL:            
            leader_var = [None]*model._size
            for idx in range(model._size):
                leader_var[idx] = model.cbGetSolution(model._leader_var[idx])
            aux_var = model.cbGetSolution(model._aux_var)

            # Solve the lower-level problem.
            follower_var, follower_obj, nodes = solve_lower_level(
                    leader_var,
                    model._profits,
                    model._follower_weights,
                    model._follower_budget
            )
            assert follower_var is not None, 'Cannot solve lower level!'

            # Complete the follower's decision to a maximal packing.
            follower_var = make_maximal(
                    follower_var,
                    model._profits,
                    model._follower_weights,
                    model._follower_budget
            )

            # Separate a lifted cut.
            coef = np.multiply(model._profits, follower_var)
            set_a, set_b = lifted_cut_separation(
                    leader_var,
                    follower_var,
                    model._profits,
                    model._follower_weights
            )
            
            for idx in range(model._size):
                for item in range(len(set_b)):
                    if idx == set_b[item]:
                        coef[idx] += (model._profits[idx]
                                      - model._profits[set_a[item]])

            # Add a cut.
            if aux_var + model._tol < follower_obj:
                model.cbLazy(
                        model._aux_var
                        >= gp.quicksum(
                                coef[idx]*(1 - model._leader_var[idx])
                                for idx in range(model._size)
                        )
                )

class DeterministicModel:
    """
    Class to solve deterministic knapsack interdiction problems.
    The method is based on Fischetti et al. (2019).
    """
    def __init__(self, instance_dict, time_limit=3600):
        self.profits = instance_dict['profits']
        self.leader_weights = instance_dict['leader weights']
        self.follower_weights = instance_dict['follower weights']
        self.leader_budget = instance_dict['leader budget']
        self.follower_budget = instance_dict['follower budget']
        self.size = len(self.profits)
        self.time_limit = time_limit
        self.tol = 1e-06
                
    def solve(self):
        # Suppress Gurobi output.
        env = gp.Env(empty=True)
        env.setParam('OutputFlag',0)
        env.start()
        
        start_time = time()
        model = gp.Model('knapsack interdiction model', env=env)
        leader_var = model.addVars(self.size, vtype=GRB.BINARY)
        aux_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
        model.setObjective(aux_var, GRB.MINIMIZE)
        model.addConstr(
                gp.quicksum(
                        self.leader_weights[idx]*leader_var[idx]
                        for idx in range(self.size)
                )
                <= self.leader_budget
        )

        # Add dominance inequalities.
        idx_set_1, idx_set_2 = get_dominance(
                self.leader_weights,
                self.follower_weights,
                self.profits
        )

        for idx, item in enumerate(idx_set_1):
            model.addConstr(
                    leader_var[idx_set_2[idx]]
                    <= leader_var[item]
            )

        # Prepare for callback.
        model._leader_var = leader_var
        model._aux_var = aux_var
        model._size = self.size
        model._profits = self.profits
        model._follower_weights = self.follower_weights
        model._follower_budget = self.follower_budget
        model._root_relax = None
        model._tol = self.tol

        model.Params.LazyConstraints = 1
        model.Params.TimeLimit = self.time_limit
        model.Params.MIPGap = 0.0
        
        model.optimize(interdiction_cuts_callback)

        runtime = time() - start_time

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            results_dict = {
                'objective': model.objVal,
                'runtime': runtime,
                'node count': model.getAttr('NodeCount'),
                'optimality gap': model.MIPGap,
                'root relaxation': model._root_relax,
            }
            try:
                sol = [var.x for var in model.getVars()][:-1]
                results_dict['leader decision'] = sol
                return results_dict
            except:
                return results_dict
        else:
             return None       
