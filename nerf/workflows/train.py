from functools import partial

import flytekit as fk
from flytekit.types.file import FlyteFile

from nerf import tasks
from nerf.core.structs import Metric, SearchSpace

@fk.workflow
def train(
    image: FlyteFile="album.jpg",
    overrides: SearchSpace = SearchSpace(
        batch_size=[256],
        d_model=[64, 128, 256],
        n_bands=[8, 12],
        n_layers=[4, 8, 16, 32],
        offset=[2],
        max_epochs=[64],
        learning_rate=[0.001, 0.0001],
        patience=[128],
    )
) -> list[Metric]:
    
    
    grid = tasks.gridsearch(searchspace=overrides)
    names = fk.map_task(tasks.label)(params=grid)
    results = fk.map_task(partial(tasks.fit, image=image))(params=grid, name=names)
    scores = fk.map_task(partial(tasks.test, image=image))(result=results, name=names)

    fk.map_task(tasks.animate)(result=results, name=names)
    tasks.plot(scores=scores)

    return scores

if __name__ == "__main__":
    train()
