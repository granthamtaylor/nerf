import flytekit as fk

from nerf.orchestration import image
from nerf.core.structs import Hyperparameters


@fk.task(container_image=image)
def parameterize() -> Hyperparameters:
    return Hyperparameters()
