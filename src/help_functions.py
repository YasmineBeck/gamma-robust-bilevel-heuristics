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

def get_dominance(leader_weights, follower_weights, profits):
    """Determine the index sets for dominance inequalities."""
    size = len(leader_weights)
    idx_set_1 = []
    idx_set_2 = []
    for idx1 in range(size):
        for idx2 in range(size):
            if idx1 == idx2:
                pass
            else:
                if((leader_weights[idx1] <= leader_weights[idx2])
                   and (follower_weights[idx1] <= follower_weights[idx2])
                   and (profits[idx1] >= profits[idx2])):
                    idx_set_1.append(idx1)
                    idx_set_2.append(idx2)
    return idx_set_1, idx_set_2

def lifted_cut_separation(leader_var, follower_var, profits,
                          follower_weights, tol=1e-06):
    """Determine the sets of items for lifted cuts."""
    set_a = []
    set_b = []
    zeros = np.where(follower_var < 0.5)[0]
    ones = np.where(follower_var > 0.5)[0]
    for one in ones:
        candidates = []
        for zero in zeros:
            if zero not in set_b:
                if((follower_weights[one] >= follower_weights[zero])
                   and (profits[one] + tol < profits[zero])):
                    candidates.append(zero)
        if candidates:
            coefs = []
            for candidate in candidates:
                coef = ((profits[candidate] - profits[one])
                        *(1 - leader_var[candidate]))
                coefs.append(coef)
            max_coef = np.argmax(coefs)
            set_a.append(one)
            set_b.append(candidates[max_coef])
    return set_a, set_b

def make_maximal(var, profits, weights, budget):
    """Complete a feasible decision to a maximal packing."""
    size = len(var)
    residual = budget - sum(weights[idx]*var[idx] for idx in range(size))
    
    # Order items in decreasing order according to profit-to-weight ratio.
    order = np.argsort(-np.divide(profits, weights))
    var = np.asarray(var)[order]
    weights = weights[order]
    
    idx = 0
    while((idx < size) and (residual > 0)):
        if((var[idx] < 0.5) and (residual - weights[idx] >= 0)):
            residual -= weights[idx]
            var[idx] = 1
        idx += 1
    
    # Revert ordering.
    revert_order = np.argsort(order)
    var = var[revert_order]
    return var 

def solve_lower_level(leader_var, profits, weights, budget, time_limit=3600):
    """Solve a deterministic lower-level problem that is parameterized
    by a given leader's decision."""
    size = len(profits)
    model = gp.Model()
    var = model.addVars(size, vtype=GRB.BINARY)
    model.setObjective(
        gp.quicksum(
            profits[idx]*var[idx] for idx in range(size)
        ),
        GRB.MAXIMIZE
    )

    # Add knapsack constraint.
    model.addConstr(
        gp.quicksum(
            weights[idx]*var[idx] for idx in range(size)
        )
        <= budget
    )
    
    # Add interdiction constraints.
    if leader_var:
        for idx in range(size):
            model.addConstr(
                var[idx] <= 1 - leader_var[idx]
            )
        
    model.Params.OutputFlag = False
    model.Params.TimeLimit = time_limit
    model.optimize()

    # Extract solution.
    if model.status == GRB.OPTIMAL:
        var = [var.x for var in model.getVars()]
        obj = model.objVal
        nodes = model.getAttr('NodeCount')
    else:
        var = None
        obj = np.inf
        nodes = model.getAttr('NodeCount')
    return var, obj, nodes

def solve_refinement_problem(leader_var, costs, profits, const, weights,
                             budget, time_limit=3600):
    """Solve the refinement problem of a lower-level sub-problem to obtain
    an optimistic follower's respone."""
    size = len(costs)
    model = gp.Model()
    var = model.addVars(size, vtype=GRB.BINARY)
    
    # Set objective to account for an optimistic follower.
    model.setObjective(
        gp.quicksum(
            costs[idx]*var[idx] for idx in range(size)
        ),
        GRB.MINIMIZE
    )
    
    # Add knapsack constraint.
    model.addConstr(
        gp.quicksum(
            weights[idx]*var[idx] for idx in range(size)
        )
        <= budget
    )
    
    # Add interdiction constraints.
    if leader_var:
        for idx in range(size):
            model.addConstr(
                var[idx] <= 1 - leader_var[idx]
            )
            
    # Add constraint for lower-level optimality.
    model.addConstr(
        gp.quicksum(
            profits[idx]*var[idx] for idx in range(size)
        )
        >= const
    )
    
    model.Params.OutputFlag = False
    model.Params.TimeLimit = time_limit
    model.optimize()

    # Extract solution.
    if model.status == GRB.OPTIMAL:
        var = [var.x for var in model.getVars()]
        obj = model.objVal
        nodes = model.getAttr('NodeCount')
    else:
        var = None
        obj = None
        nodes = model.getAttr('NodeCount')
    return var, obj, nodes
