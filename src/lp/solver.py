### ADAPTED FROM JACK HAWTHORNE MASTER THESIS
### Changed solve function for modified POET

import logging
from copy import deepcopy
import time
from pyomo.opt import SolverFactory
import numpy as np

logging.getLogger('pyomo.core').setLevel(logging.ERROR)


def SOLVE(solver, instance, objectives):
    solver.solve(instance)
    obj_vector = []
    for obj in objectives:
        obj_vector += [getattr(instance, obj).value]
    return obj_vector


def solve(instance,
          objectives,
          solver='gurobi',
          verbose=False,
          n_points=2):
    """Simple in series solving for one network, no history"""
    assert n_points >= len(
        objectives), f"'n_points' must be larger than or equal to {len(objectives)}. Given: {n_points}"
    solver = SolverFactory(solver, solver_io="python")
    # instance = model.set_up(network, objectives)
    results = []
    instances = []

    # reset objective vals
    [setattr(instance, obj, 0) for obj in objectives]
    [getattr(instance, o + '_objective').deactivate() for o in ["cost", 'co2']]

    # Solve extremes
    for obj in objectives:
        # if verbose:
        #     print('\tsolving for objective: %s'%obj)
        other_objs = [o for o in objectives if o != obj]
        getattr(instance, obj + '_objective').activate()
        for o in other_objs:
            getattr(instance, o + '_objective').deactivate()

        # tuple_network = network.network_to_tuple()
        start = time.time()
        obj_vector = SOLVE(solver, instance, objectives)
        end = time.time()
        # if verbose:
        #     print('\t\t solve time: ', end - start, ', success: ', bool(obj_vector[0]))
        results.append(obj_vector)
        instances.append(deepcopy(instance))

    # Solve between points
    # For now, epislon constraint along co2, like ehub tool
    if len(objectives) == 1:
        pass
    elif n_points > len(objectives) and results[0][0]:
        assert len(
            objectives) == 2, "Currently only support two objectives for epsilon-constraint method"
        obj_solve = objectives[0]
        obj_constrain = objectives[-1]

        obj_solve_idx = 0
        obj_constrain_idx = 1

        getattr(instance, obj_solve + '_objective').activate()
        getattr(instance, obj_constrain + '_objective').deactivate()

        constraints = np.linspace(results[0][obj_constrain_idx],
                                  results[-1][obj_constrain_idx],
                                  num=n_points)[1:-1]

        for i, c in enumerate(constraints):
            setattr(instance, obj_constrain + '_upper_bound', c)
            start = time.time()
            obj_vector = SOLVE(solver, instance, objectives)
            end = time.time()
            # if verbose:
            #     print('\t\t solve time: ', end - start, ', success: ', bool(obj_vector[0]))
            results.insert(i+1, obj_vector)

    return results, instances


if __name__ == "__main__":
    pass
