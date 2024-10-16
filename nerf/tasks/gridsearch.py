from itertools import product

import flytekit
from rich.pretty import pprint

from nerf.orchestration.constants import image
from nerf.core.structs import Hyperparameters, SearchSpace

@flytekit.task(container_image=image, cache=True, cache_version="#cache-v1",)
def gridsearch(searchspace: SearchSpace) -> list[Hyperparameters]:
    """Generate a grid of hyperparameters"""
    
    overrides = {key: value for key, value in vars(searchspace).items() if value is not None}
    
    keys = list(overrides.keys())
    values = list(overrides.values())
    
    grid = [Hyperparameters(**dict(zip(keys, combination))) for combination in product(*values)]
    
    for params in grid:
        pprint(params)
    
    return grid