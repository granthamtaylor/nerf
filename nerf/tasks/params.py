import typing
import flytekit as fk

from nerf.orchestration import image
from nerf.core.structs import Hyperparameters
from nerf.core.utils import permutate_params


@fk.task(container_image=image)
def parameterize() -> Hyperparameters:
    return Hyperparameters()


@fk.task(container_image=image)
def hp_grid_search(base: Hyperparameters) -> typing.List[Hyperparameters]:
    new_params = permutate_params(
        base=base,
        d_model=[64, 128],
        # n_bands=[7, 8, 9],
        # n_layers=[4, 8],
        # offset=[1, 2],
        learning_rate=[0.001, 0.01],
    )
    return new_params
