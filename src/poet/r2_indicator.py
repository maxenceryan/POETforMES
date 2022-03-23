from numpy import linspace

def get_r2_indicator(pareto_front,
                      weights_num=3,
                      weights_start=None,
                      weights_end=None,
                      utopia=None):
    
    """
        Calculates R2 indicator for pareto front.
        Default utopia point is at origin.
        Default number of weight vectors is 3.
        Default start and 
    """

    if None in pareto_front:
        return None
    
    num_objectives = len(pareto_front[0])
    if weights_start is None:
        weights_start = pareto_front[0]
    if weights_end is None:
        weights_end = pareto_front[-1]
    if utopia is None:
        utopia = [0]*num_objectives

    if not (len(weights_end) == len(weights_start) == len(utopia) == num_objectives):
        raise ValueError(
            "Size of weight vectors and utopia points must match number of objectives.")

    if not utopia:
        utopia = [0]*num_objectives
        
    if weights_num < 2:
        raise ValueError("Must have at least 2 weight vectors.")
    
    weights = _get_weight_vectors(weights_start, weights_end, weights_num, num_objectives)

    # z = []
    # for w in weights:
    #     y = []
    #     for pt in pareto_front:
    #         x = []
    #         for i in range(num_objectives):
    #             x.append(w[i] * abs(utopia[i] - pt[i]))
    #         y.append(max(x))
    #     z.append(min(y))

    # r2 = 1 / weights_num * sum(z)

    return _r2_subfunction_sum(weights, pareto_front, num_objectives, utopia)

def get_CPR(point: tuple, 
            pareto_front: list,
            weights_num=3,
            weights_start=None,
            weights_end=None,
            utopia=None):
    
    """
        Calculates the contribution of a given solution to the R2 indicator of the solution set.
    """
    if not pareto_front:
        raise ValueError("Pareto front is empty.")
    num_objectives = len(pareto_front[0])

    if not (len(weights_end) == len(weights_start) == len(utopia) == num_objectives):
        raise ValueError(
            "Size of weight vectors and utopia points must match number of objectives.")

    if not utopia:
        utopia = [0]*num_objectives
        
    if weights_num < 2:
        raise ValueError("Must have at least 2 weight vectors.")
    
    new_front = pareto_front.copy()
    new_front.remove(point)
    
    weights = _get_weight_vectors(weights_start, weights_end, weights_num, num_objectives)
    
    return 1 / weights_num * sum(_r2_subfunction_min(w, pareto_front, num_objectives, utopia) 
                                  - _r2_subfunction_min(w, new_front, num_objectives, utopia)
                                 for w in weights)


def _get_weight_vectors(start, end, num, num_objectives):
    # return [(1,0), (0.5, 0.5), (0,1)]
    return list(zip(*(linspace(start[i], end[i] , num) for i in range(num_objectives))))

# Separate parts of the function to enable caching or jit
# Can be refactored...

def _r2_subfunction_max(w, pt, num_objectives, utopia):
    return max(w[i] * abs(utopia[i] - pt[i]) for i in range(num_objectives))

def _r2_subfunction_min(w, pareto_front, num_objectives, utopia):
    return min(_r2_subfunction_max(w, pt, num_objectives, utopia) for pt in pareto_front)
    
def _r2_subfunction_sum(weights, pareto_front, num_objectives, utopia):
    return 1 / len(weights) * sum(_r2_subfunction_min(w, pareto_front, num_objectives, utopia) for w in weights)


if __name__ == "__main__":
    pareto_front = [(1,3), (2,2), (3,1)]
    print(f"Pareto: {pareto_front}\tR2: {get_r2_indicator(pareto_front, weights_num=100)}")
    pareto_front = [(2,4), (3,3), (4,2)]
    print(f"Pareto: {pareto_front}\tR2: {get_r2_indicator(pareto_front, weights_num=100)}")
    pareto_front = [(1,4), (1,2), (2,1), (3,1), (4,1)]
    print(f"Pareto: {pareto_front}\tR2: {get_r2_indicator(pareto_front, weights_num=100)}")
