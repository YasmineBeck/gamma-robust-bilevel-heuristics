# This file is part of the code used for the computational study
# in the paper
#
#     "Heuristic Methods for Gamma-Robust Mixed-Integer Linear
#      Bilevel Problems"
#
# by Yasmine Beck, Ivana Ljubic, and Martin Schmidt (2024).

# Global imports
import argparse
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Local imports
from instance_data_builder import InstanceDataBuilder

def solve_extended_follower(leader_vars, gamma, profits, deviations,
                            weights, budget):
    # Solve the extended lower-level problem that is parameterized in
    # a given leader's decision.
    size = len(profits)   

    model = gp.Model()

    follower_vars = model.addVars(size, vtype=GRB.BINARY)
    rob_vars = model.addVars(size + 1, vtype=GRB.CONTINUOUS, lb=0.0)
    var_t = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
    
    # Set the objective function.
    model.setObjective(
        -gamma*rob_vars[size]
        + gp.quicksum(profits[idx]*follower_vars[idx]
                      - rob_vars[idx] for idx in range(size)),
        GRB.MAXIMIZE
    )

    # Add the budget constraint.
    model.addConstr(
        gp.quicksum(
            weights[idx]*follower_vars[idx] for idx in range(size)
        )
        <= budget
    )

    for idx in range(size):
        # Add interdiction constraints.
        model.addConstr(follower_vars[idx] <= 1 - leader_vars[idx])
        
        # Add constraints for the robustification.
        model.addConstr(rob_vars[size] + rob_vars[idx]
                        >= deviations[idx]*follower_vars[idx])

    model.Params.OutputFlag = False

    # Optimize.
    model.optimize()
    
    # Extract solution.
    if model.status == GRB.OPTIMAL:
        follower_sol = np.array([follower_vars[idx].X for idx in range(size)])
        rob_sol = np.array([rob_vars[idx].X for idx in range(size + 1)])
        obj = model.objVal
        return obj, follower_sol, rob_sol
    else:
        return np.inf, None, None

def extended_interdiction_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        leader_vars = [None]*model._size
        follower_vars = [None]*model._size
        aux_vars = [None]*(model._size + 1)
        current_follower_obj = 0
        for idx in range(model._size):
            leader_vars[idx] = model.cbGetSolution(model._leader_vars[idx])
            follower_vars[idx]\
                = model.cbGetSolution(model._follower_vars[idx])
            aux_vars[idx] = model.cbGetSolution(model._aux_vars[idx])
            current_follower_obj += (model._profits[idx]*follower_vars[idx]
                                     - aux_vars[idx])
        aux_vars[model._size]\
            = model.cbGetSolution(model._aux_vars[model._size])
        current_follower_obj -= model._gamma*aux_vars[model._size]

        # Solve the lower-level problem.
        obj, sol, aux_sol = solve_extended_follower(
            leader_vars,
            model._gamma,
            model._profits,
            model._deviations,
            model._follower_weights,
            model._follower_budget
        )
        
        assert sol is not None, 'Cannot solve lower level!'
        
        const = (-model._gamma*aux_sol[model._size]
                 - sum(aux_sol[idx] for idx in range(model._size)))

        # Add a cut.
        if current_follower_obj + model._tol < obj:
            coeffs = np.multiply(model._profits, sol)
            
            model.cbLazy(
                gp.quicksum(
                    model._profits[idx]*model._follower_vars[idx]
                    - model._aux_vars[idx]
                    for idx in range(model._size)
                )
                - model._gamma*model._aux_vars[model._size]
                >= const
                + gp.quicksum(
                    coeffs[idx]*(1 - model._leader_vars[idx])
                    for idx in range(model._size)
                )
            )

class GammaRobustExtendedModel:
    """
    Class for solving generalized discrete linear knapsack interdiction
    problems with a Gamma-robust follower using an extended reformulation.
    """
    def __init__(self,
                 instance_file,
                 conservatism,
                 uncertainty,
                 deviations,
                 time_limit=3600):
        self.instance_file = instance_file
        self.conservatism = conservatism
        self.uncertainty = uncertainty
        self.deviations = deviations
        self.tol = 1e-06
        self.time_limit = time_limit

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
        self.gamma = self.instance_dict['gamma']
        self.size = self.instance_dict['size']
        self.deviations = self.instance_dict['deviations']

    def main(self):
        # Suppress Gurobi output.
        env = gp.Env(empty=True)
        env.setParam('OutputFlag',0)
        env.start()

        model = gp.Model('extended model', env=env)

        leader_vars = model.addVars(self.size, vtype=GRB.BINARY)
        follower_vars = model.addVars(self.size, vtype=GRB.BINARY)
        aux_vars = model.addVars(self.size + 1, vtype=GRB.CONTINUOUS, lb=0.0)

        # Set upper-level objective.
        model.setObjective(
            gp.quicksum(
                self.instance_dict['leader costs'][idx]*leader_vars[idx]
                + self.instance_dict['follower costs'][idx]*follower_vars[idx]
                for idx in range(self.size)
            ),
            GRB.MINIMIZE
        )

        # Add upper-level budget constraint.
        model.addConstr(
            gp.quicksum(
                self.instance_dict['leader weights'][idx]*leader_vars[idx]
                for idx in range(self.size)
            )
            <= self.instance_dict['leader budget']
        )

        # Add lower-level budget constraint.
        model.addConstr(
            gp.quicksum(
                self.instance_dict['follower weights'][idx]*follower_vars[idx]
                for idx in range(self.size)
            )
            <= self.instance_dict['follower budget']
        )

        # Add interdiction constraints.
        for idx in range(self.size):
            model.addConstr(
                follower_vars[idx]
                <= 1 - leader_vars[idx]
            )

        # Add constraints for robustification.
        for idx in range(self.size):
            model.addConstr(
                aux_vars[idx] + aux_vars[self.size]
                >= self.deviations[idx]*follower_vars[idx]
            )

        # Prepare for callback.
        model._leader_vars = leader_vars
        model._follower_vars = follower_vars
        model._aux_vars = aux_vars
        model._size = self.size
        model._profits = self.instance_dict['profits']
        model._follower_weights = self.instance_dict['follower weights']
        model._follower_budget = self.instance_dict['follower budget']
        model._deviations = self.deviations
        model._gamma = self.gamma
        model._tol = self.tol

        model.Params.LazyConstraints = 1
        model.Params.TimeLimit = self.time_limit

        # Solve the model.
        model.optimize(extended_interdiction_callback)

        # Extract solution.
        if (model.status == GRB.OPTIMAL) or (model.status == GRB.TIME_LIMIT):
            try:
                leader_sol = [var.x for var in leader_vars.values()]
                follower_sol = [var.x for var in follower_vars.values()]
                aux_sol = [var.x for var in aux_vars.values()]

                follower_obj = (
                    sum(self.instance_dict['profits'][idx]*follower_sol[idx]
                        - aux_sol[idx]
                        for idx in range(self.size))
                    - self.gamma*aux_sol[self.size]
                )
            
                results_dict = {
                    'leader objective': model.objVal,
                    'leader solution': leader_sol,
                    'follower objective': follower_obj,
                    'follower solution': follower_sol,
                    'runtime': model.Runtime,
                    'node count': model.getAttr('NodeCount'),
                    'optimality gap': model.MIPGap
                }
            except:
                results_dict = None
        else:
            return None
        return results_dict
        
if __name__ == "__main__":
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

    model = GammaRobustExtendedModel(
        instance_file,
        conservatism,
        uncertainty,
        deviations
    )
    results_dict = model.main()
    
    with open(output_file, 'w') as outfile:
        json.dump(results_dict, outfile, indent=4)
