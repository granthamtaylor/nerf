import os

import wandb
from lightning.pytorch import Trainer
import flytekit
from flytekit.types.file import FlyteFile

from nerf.core.model import NeRFModule
from nerf.orchestration.constants import image, wandb_secret
from nerf.core.structs import Metric, Result

@flytekit.task(
    container_image=image,
    requests=flytekit.Resources(gpu="1", cpu="16", mem="64Gi"),
    secret_requests=[wandb_secret],
    cache=True,
    cache_version="#cache-v1",
)
def test(image: FlyteFile, result: Result, name: str) -> Metric:
    """Test the model on the image"""
    
    key = flytekit.current_context().secrets.get(key="WANDB_API_KEY")
    wandb.login(key=key)
    
    with wandb.init(project='nerf', id=name, reinit=True) as run:

        image.download()
        result.animation.download()
        result.model.download()

        module = NeRFModule.load_from_checkpoint(result.model.path, image=image.path, params=result.params)
        trainer = Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

        losses = trainer.test(module)

        loss = losses[0]["test/loss"]
        compression = len(module) / os.path.getsize(image)
        
        run.summary["loss"] = loss
        run.summary["compression"] = compression
        
    return Metric(loss=loss, compression=compression)
