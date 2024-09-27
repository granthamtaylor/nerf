from itertools import product

import flytekit as fk
from rich.pretty import pprint

from nerf.orchestration import image
from nerf.core.structs import Hyperparameters

@fk.task(container_image=image)
def gridsearch(overrides: dict[str, list[float|int]]) -> list[Hyperparameters]:
    
    combination = product(*[v if isinstance(v, (list, tuple)) else [v] for v in overrides.values()])
    
    configs = [dict(zip(overrides.keys(), values)) for values in combination]

    grid = [Hyperparameters(**config) for config in configs]
    
    for params in grid:
        pprint(params)
    
    return grid
