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
        d_model=[16, 32, 64],
        n_bands=[8, 10],
        n_layers=[3, 4],
        offset=[2],
        max_epochs=[128],
        learning_rate=[0.001],
        patience=[16],
    )
) -> list[Metric]:
    
    grid = tasks.gridsearch(searchspace=overrides)
    results = fk.map_task(partial(tasks.fit, image=image))(params=grid)
    scores = fk.map_task(partial(tasks.test, image=image))(result=results)

    # fk.map_task(tasks.animate)(result=results)
    tasks.plot(scores=scores)

    return scores

if __name__ == "__main__":
    train()
