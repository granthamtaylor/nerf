import os

from lightning.pytorch import Trainer
import flytekit as fk
from flytekit.types.file import FlyteFile

from nerf.core.model import NeRFModule
from nerf.orchestration import image
from nerf.core.structs import Hyperparameters


@fk.task(
    container_image=image,
    requests=fk.Resources(gpu="1", cpu="16", mem="32Gi"),
    accelerator=fk.extras.accelerators.A100,
    cache=True,
    cache_version='#cache-v1',
)
def test(
    image: FlyteFile,
    model: FlyteFile,
    params: Hyperparameters
) -> tuple[float, float]:

    module = NeRFModule.load_from_checkpoint(model.path, image=image.path, params=params)
    trainer = Trainer(enable_progress_bar=False)

    losses = trainer.test(module)
    print(losses)
    loss = losses[0]['test/loss']
    compression = len(module) / os.path.getsize(image)
    
    return loss, compression
