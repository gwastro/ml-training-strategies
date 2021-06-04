import numpy as np
from scipy.optimize import minimize
from pycbc.io.record import FieldArray
from pycbc.boundaries import Bounds

def uniform_from_bounds(bounds, size=1):
    return np.random.uniform(bounds.min, bounds.max, size=size)

def estimate_transformed_bounds(param_bounds, transform, draws=100):
    #param_bounds : dict {param_name: bounds}
    names = []
    missing_params = []
    for name in transform.inputs:
        if name not in param_bounds:
            missing_params.append(name)
        bound = param_bounds[name]
        names.append(name)
    if len(missing_params) > 0:
        msg  = 'To estimate the bounds of a transformation the bounds '
        msg += 'for all input parameters need to be provided. Missing '
        msg += 'bounds for {}.'.format(missing_params)
        raise ValueError(msg)
    
    data = [uniform_from_bounds(param_bounds[name], size=draws) for name in names]
    to_trans = FieldArray.from_arrays(data, dtype=[(name, float) for name in names])
    
    res = transform.transform(to_trans)
    ret_bounds = {}
    for name in transform.outputs:
        ret_bounds[name] = Bounds(np.min(res[name]), np.max(res[name]))
    
    return ret_bounds
