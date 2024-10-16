from pathlib import Path

from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
import flytekit
from flytekit.types.file import FlyteFile

from nerf.core.model import NeRFModule
from nerf.orchestration.constants import context
from nerf.core.structs import Hyperparameters, Result
from nerf.core.callbacks import ParquetBatchWriter

@context['gpu']
def fit(params: Hyperparameters, image: FlyteFile, name: str) -> Result:
    """Fit the model to the image"""
    
    key = flytekit.current_context().secrets.get(key="WANDB_API_KEY")
    wandb.login(key=key)
    
    with wandb.init(project='nerf', id=name):
        
        image.download()

        datapath = Path(flytekit.current_context().working_directory) / "results.parquet"

        module = NeRFModule(params=params, image=str(image.path))
        
        checkpointer = ModelCheckpoint(
            dirpath=flytekit.current_context().working_directory,
            monitor="validate/loss",
        )

        trainer = Trainer(
            logger=WandbLogger(project="nerf", id=name),
            enable_progress_bar=False,
            max_epochs=params.max_epochs,
            callbacks=[ParquetBatchWriter(path=datapath), checkpointer],
        )

        trainer.fit(module)

        animation = FlyteFile(str(datapath))
        model = FlyteFile(checkpointer.best_model_path)
        
    print(datapath)

    return Result(animation=animation, model=model, params=params)
