from functools import partial

import flytekit as fk
from flytekit.types.file import FlyteFile

from nerf import tasks
from nerf.core.structs import Metric


@fk.workflow
def train(
    image: FlyteFile=FlyteFile('images/papas.jpg'),
    overrides: dict[str, list[float | int]] = {
        "d_model": [8, 12, 16, 24, 32],
        "n_bands": [8, 10],
        "n_layers": [2, 3, 4],
    }
) -> list[Metric]:
    
    grid = tasks.gridsearch(overrides=overrides)
    results = fk.map_task(partial(tasks.fit, image=image))(params=grid)
    scores = fk.map_task(partial(tasks.test, image=image))(result=results)

    fk.map_task(tasks.animate)(result=results)
    tasks.plot(scores=scores)

    return scores

if __name__ == "__main__":
    train()
