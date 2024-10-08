import os

import pyarrow.parquet as pq
import lightning.pytorch as lit
from lightning.pytorch import callbacks
import polars as pl
import torch

from nerf.core.structs import InputTensor

class ParquetBatchWriter(callbacks.Callback):
    def __init__(self, path: str | os.PathLike):
        super().__init__()

        self.path: str = str(path)
        self.schema = None
        self.writer = None

    def on_validation_batch_end(
        self,
        trainer: lit.Trainer,
        pl_module: lit.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: InputTensor,
        batch_idx: int,
    ) -> None:
        if trainer.sanity_checking:
            return

        df = pl.DataFrame(
            {
                "epoch": [pl_module.current_epoch] * int(batch.batch_size[0]),
                "coordinates": batch.coordinates.xy.cpu().detach().numpy(),
                "color": outputs["predictions"].cpu().detach().numpy(),
            }
        )

        table = df.to_arrow()

        if self.writer is None:

            self.schema = table.schema
            self.writer = pq.ParquetWriter(self.path, self.schema)

        self.writer.write_table(table)

    def on_fit_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        if self.writer:
            self.writer.close()
            self.writer = None
