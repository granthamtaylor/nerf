import os
import flytekit

import wandb
import polars as pl
import numpy as np

from nerf.orchestration.images import image, wandb_secret
from nerf.core.structs import Result


@flytekit.task(
    container_image=image,
    requests=flytekit.Resources(gpu="1", cpu="16", mem="64Gi"),
    secret_requests=[wandb_secret],
    cache=True,
    cache_version="#cache-v1",
)
def animate(result: Result, name: str):
    
    key = flytekit.current_context().secrets.get(key="WANDB_API_KEY")
    wandb.login(key=key)

    with wandb.init(project='nerf', id=name) as run:

        result.animation.download()    

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
