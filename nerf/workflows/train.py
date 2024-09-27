import typing
from functools import partial

import flytekit as fk
from flytekit.types.file import FlyteFile

from nerf import tasks
from nerf.core.structs import Hyperparameters
from nerf.tasks.params import hp_grid_search
from nerf.tasks.fit import fit_no_predictions
from nerf.tasks.test import TestResults


@fk.workflow
def train(
    image: FlyteFile,
    params: Hyperparameters = Hyperparameters(),
) -> FlyteFile:
    predictions, model = tasks.fit(params=params, image=image)
    tasks.animate(predictions=predictions)
    tasks.test(model=model, image=image, params=params)
    return model


@fk.workflow
def hp_search(image: FlyteFile, base_params: Hyperparameters = Hyperparameters()) -> typing.List[TestResults]:
    hps_to_try = hp_grid_search(base=base_params)

    models = fk.map_task(partial(fit_no_predictions, image=image))(params=hps_to_try)
    results = fk.map_task(partial(tasks.test, image=image))(model=models, params=hps_to_try)
    return results


if __name__ == "__main__":
    train()
