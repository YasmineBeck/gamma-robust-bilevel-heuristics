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
from time import time

# Local imports
from src.help_functions import solve_lower_level

class OptimalityChecker:
    """
    Class for optimality checks including the computation of
    upper bounds and checks for ex-post optimality conditions.
    """
    def __init__(self, instance_dict, subprobs):
        self.instance_dict = instance_dict
        self.subprobs = subprobs
        self.subprob_cnt = len(subprobs)
        self.tol = 1e-06

    def solve_follower_subprob(self, subprob, leader_sol):
        timer = time()
        sol, obj, node_cnt = solve_lower_level(
            leader_sol,
            self.instance_dict['modified profits'][subprob],
            self.instance_dict['follower weights'],
            self.instance_dict['follower budget'],
        )
        obj -= (self.instance_dict['gamma']
                *self.instance_dict['deviations'][subprob])
        subprob_time = time() - timer
        return sol, obj, node_cnt, subprob_time
        
    def compute_upper_bound(self, leader_sol, excluded_subprob):
        """Solve all lower-level sub-problems parameterized in the
        current fixed leader's decision (except for the one already
        considered while solving the corresponding bilevel sub-problem;
        see Remark 3)."""
        try:
            times = []
            node_lst = []
            objs = []
            for subprob in self.subprobs:
                if subprob != excluded_subprob:
                    sol, obj, node_cnt, subprob_time\
                        = self.solve_follower_subprob(subprob, leader_sol)
                    objs.append(obj)
                    times.append(subprob_time)
                    node_lst.append(node_cnt)
            max_obj = max(objs)
            nodes = sum(node_lst)
        except:
            max_obj = np.inf
            nodes = 0
            subprob_times = [0]
        return max_obj, nodes, times

    def ex_post_checks(self, sols, objs, lb):
        """Determine all sub-problems that yield the maximal objective
        function value."""
        best_idxs = [idx for idx in range(self.subprob_cnt)
                     if abs(lb - objs[idx]) < self.tol]
        
        # Check if the leader plays the same decision in every bilevel
        # sub-problem; see Theorem 2.
        if self.same_solution(sols):
            check_dict = {
                'status': 'optimal',
                'condition': 'same solutions',
                'gap': 0.0,
                'objective': lb,
                'best subprob': self.subprobs[best_idxs[0]],
                'leader decision': sols[0]
            }
            return check_dict

        # Check if there is a solution that dominates all sub-problem
        # solutions; see Theorem 3.
        dom_sol, dominating = self.dominating_solution(sols)
        if dominating:
            check_dict = {
                'status': 'optimal',
                'condition': 'dominating solution',
                'gap': 0.0,
                'objective': lb,
                'best subprob': self.subprobs[best_idxs[0]],
                'leader decision': dom_sol
            }
            return check_dict

        # Sort the sub-problems such that the objective function values
        # are given in non-decreasing order and solve additional
        # lower-level problems.
        sorted_idxs = np.argsort(objs)
        times = []
        ideal_times = []
        ub = np.inf
        nodes = 0
        solved = 0
        dominated = 0
        for idx1 in sorted_idxs:
            subprob_ub = objs[idx1]
            # Check whether solving additional lower-level problems can
            # improve the best known solution.
            if subprob_ub + self.tol >= ub:
                continue
            
            subprob_sol = sols[idx1]
            single_level_times = []
            for idx2 in sorted_idxs:
                if idx1 != idx2:
                    # Check whether Proposition 4 can be exploited.
                    subprob_sol, dominating\
                        = self.complete_solution(subprob_sol, sols[idx2])
                    if dominating:
                        subprob_ub = max(subprob_ub, objs[idx2])
                        dominated += 1
                        continue
                
                    # Solve the lower-level sub-problem for the fixed
                    # leader's decision.
                    follower_sol, obj, node_cnt, subprob_time\
                        = self.solve_follower_subprob(self.subprobs[idx2],
                                                      subprob_sol)
                    subprob_ub = max(subprob_ub, obj)
                    single_level_times.append(subprob_time)
                    nodes += node_cnt
                    solved += 1
                        
            # Update.
            if subprob_ub - self.tol < ub:
                ub = subprob_ub
                sol = subprob_sol
                best_subprob = self.subprobs[idx1]
                
            times.append(sum(single_level_times))
            if not single_level_times:
                ideal_times.append(0)
            else:
                ideal_times.append(max(single_level_times))
            
            # Terminate if the gap is closed.
            gap = ub - lb
            if abs(gap) < self.tol:
                if dominated > 0:
                    cond = '{}x dominance, {}x solved lower level'.format(
                        dominated,
                        solved
                    )
                else:
                    cond = 'solved lower level'
                    
                check_dict = {
                    'status': 'optimal',
                    'condition': cond,
                    'single-level problems solved': solved,
                    'single-level time': sum(times),
                    'ideal single-level time': sum(ideal_times),
                    'node count': nodes,
                    'gap': gap,
                    'objective': ub,
                    'best subprob': best_subprob,
                    'leader decision': sol
                }
                return check_dict

        # The bilevel problem has not been solved to global optimality.
        if dominated > 0:
            cond = '{}x dominance, {}x solved lower level'.format(
                dominated,
                solved
            )
        else:
            cond = 'solved lower level'

        check_dict = {
            'status': 'feasible',
            'condition': cond,
            'single-level problems solved': solved,
            'single-level time': sum(times),
            'ideal single-level time': sum(ideal_times),
            'node count': nodes,
            'gap': gap,
            'lower bound': lb,
            'upper bound': ub,
            'best subprob': best_subprob,
            'leader decision': sol
        }
        return check_dict
        
    def same_solution(self, sols):
        """Check whether the leader plays the same decision in every
        bilevel sub-problem. Return True in the positive case, and
        False otherwise."""
        for idx in range(1, len(sols)):
            if not all(abs(sol1 - sol2) < self.tol
                       for sol1, sol2 in zip(sols[0], sols[idx])):
                return False
        return True

    def dominating_solution(self, sols):
        """Check whether there is a leader decision that dominates all
        sub-problem solutions. Return True in the positive case, and
        False otherwise."""
        new_sol = sols[0].copy()
        dominating = True
        for idx in range(1, len(sols)):
            new_sol, dominating = self.complete_solution(new_sol, sols[idx])
            if not dominating:
                return None, dominating
        return new_sol, dominating

    def complete_solution(self, sol1, sol2):
        """Check whether sol1 can be completed to a solution that dominates
        sol2. Return True in the positive case, and False otherwise."""
        size = len(sol1)
        new_sol1 = sol1.copy()
        zeros = [idx for idx in range(size) if sol1[idx] < 0.5]
        for idx in zeros:
            if sol2[idx] > 0.5:
                # Pack the items that are packed in sol2 but not in sol1.
                new_sol1[idx] = sol2[idx]

        # Check whether the new leader's decision is feasible.
        budget = self.instance_dict['leader budget']
        weights = self.instance_dict['leader weights']
        residual = (budget
                    - sum(weights[idx]*new_sol1[idx] for idx in range(size)))
        if residual < -self.tol:
            return sol1, False        
        return new_sol1, True

    def is_dominated(self, new_sol, sols):
        """Check whether one of the solutions given by sols can be completed
        such that it dominates new_sol. Return True in the positive case,
        and False otherwise."""
        for sol in sols:
            dom_sol, dominating = self.complete_solution(sol, new_sol)
            if dominating:
                return True
        return False
