from pathlib import Path

from lightning.pytorch import Trainer
import flytekit as fk
from flytekit.types.file import FlyteFile

from nerf.core.model import NeRFModule
from nerf.orchestration.images import image
from nerf.core.structs import Hyperparameters, Result
from nerf.core.callbacks import ParquetBatchWriter

@fk.task(
    container_image=image,
    requests=fk.Resources(gpu="1", cpu="16", mem="64Gi"),
    cache=True,
    cache_version="#cache-v1",
)
def fit(params: Hyperparameters, image: FlyteFile) -> Result:
    
    image.download()

    datapath = Path(fk.current_context().working_directory) / "results.parquet"

    module = NeRFModule(params=params, image=str(image.path))

    trainer = Trainer(
        enable_progress_bar=False,
        max_epochs=params.max_epochs,
        callbacks=[ParquetBatchWriter(path=datapath)],
    )

    trainer.fit(module)

    animation = FlyteFile(str(datapath))
    model = FlyteFile(trainer.checkpoint_callback.best_model_path)

    return Result(animation=animation, model=model, params=params)
