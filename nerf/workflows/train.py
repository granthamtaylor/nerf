from functools import partial

import flytekit
from flytekit.types.file import FlyteFile

from nerf import tasks
from nerf.core.structs import Metric, SearchSpace

@flytekit.workflow
def train(
    image: FlyteFile="assets/album.jpg",
    overrides: SearchSpace = SearchSpace(
        batch_size=[512],
        d_model=[64, 128, 256],
        n_bands=[8],
        n_layers=[4, 8, 16],
        offset=[2],
        max_epochs=[16],
        learning_rate=[0.0001],
        patience=[128],
    )
) -> list[Metric]:
    """Train a model on the image"""
    
    grid = tasks.gridsearch(searchspace=overrides)
    names = flytekit.map_task(tasks.label)(params=grid)
    results = flytekit.map_task(partial(tasks.fit, image=image))(params=grid, name=names)
    scores = flytekit.map_task(partial(tasks.test, image=image))(result=results, name=names)

    flytekit.map_task(tasks.animate)(result=results, name=names)
    tasks.plot(scores=scores)

    return scores

if __name__ == "__main__":
    train()
