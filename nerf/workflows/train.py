
import flytekit as fk
from flytekit.types.file import FlyteFile

from nerf import tasks
from nerf.core.structs import Hyperparameters

@fk.workflow
def train(
    image: FlyteFile,
    params: Hyperparameters = Hyperparameters(),
) -> None:

    predictions, model = tasks.fit(params=params, image=image)
    tasks.animate(predictions=predictions)
    tasks.test(model=model, image=image, params=params)

if __name__ == '__main__':
    train()