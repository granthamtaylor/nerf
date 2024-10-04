import os

from lightning.pytorch import Trainer
import flytekit as fk
from flytekit.types.file import FlyteFile

from nerf.core.model import NeRFModule
from nerf.orchestration.images import image
from nerf.core.structs import Metric, Result

@fk.task(
    container_image=image,
    requests=fk.Resources(gpu="1", cpu="16", mem="32Gi"),
    accelerator=fk.extras.accelerators.A100,
    cache=True,
    cache_version="#cache-v1",
)
def test(image: FlyteFile, result: Result) -> Metric:

    module = NeRFModule.load_from_checkpoint(result.model.path, image=image.path, params=result.params)
    trainer = Trainer(enable_progress_bar=False)

    losses = trainer.test(module)
    loss = losses[0]["test/loss"]
    compression = len(module) / os.path.getsize(image)

    return Metric(loss=loss, compression=compression)
