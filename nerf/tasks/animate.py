import flytekit

import wandb
import polars as pl
import numpy as np

from nerf.orchestration.constants import context
from nerf.core.structs import Result


@context['gpu']
def animate(result: Result, name: str):
    """Animate the image over training epochs"""
    
    result.animation.download()

    key = flytekit.current_context().secrets.get(key="WANDB_API_KEY")
    wandb.login(key=key)

    with wandb.init(project='nerf', id=name) as run:


        # LWHC
        animation = np.array(
            pl.read_parquet(result.animation.path)
            .select(
                x=pl.col("coordinates").arr.first(),
                y=pl.col("coordinates").arr.last(),
                epoch=pl.col("epoch"),
                color=pl.col("color"),
            )
            .sort("epoch", "x", "y")
            .group_by("epoch", "x")
            .agg("color")
            .sort("epoch", "x")
            .group_by("epoch")
            .agg("color")
            .get_column("color")
            .to_list()
        )
        
        images = [wandb.Image(frame) for frame in animation]
        
        run.log({"animation": images})
