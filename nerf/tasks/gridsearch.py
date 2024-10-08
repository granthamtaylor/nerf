from itertools import product

import flytekit as fk
from rich.pretty import pprint

from nerf.orchestration.images import image
from nerf.core.structs import Hyperparameters, SearchSpace

@fk.task(container_image=image, cache=True, cache_version="#cache-v1",)
def gridsearch(searchspace: SearchSpace) -> list[Hyperparameters]:
    
    
    overrides = {key: value for key, value in vars(searchspace).items() if value is not None}
    
    keys = list(overrides.keys())
    values = list(overrides.values())
    
    grid = [Hyperparameters(**dict(zip(keys, combination))) for combination in product(*values)]
    
    for params in grid:
        pprint(params)
    
    return grid