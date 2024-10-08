import flytekit as fk

import plotly
import polars as pl
import numpy as np
import plotly.express as px

from nerf.orchestration.images import image
from nerf.core.structs import Result


@fk.task(
    container_image=image,
    requests=fk.Resources(cpu="16", mem="256Gi"),
    enable_deck=True
)
def animate(result: Result):

    result.animation.download()

    animation = np.array(
        pl.read_parquet(result.animation.path)
        .select(
            t=pl.col("epoch"),
            x=pl.col("coordinates").arr.first(),
            y=pl.col("coordinates").arr.last(),
            c=pl.col("color"),
        )
        .sort("t", "x", "y")
        .group_by("t", "x")
        .agg("c")
        .sort("t", "x")
        .group_by("t")
        .agg("c")
        .get_column("c")
        .to_list()
    )

    fig = px.imshow(animation, animation_frame=0, binary_string=True, binary_compression_level=9, zmax=[255, 255, 255])

    fig.update_layout(coloraxis_showscale=False, height=900, hovermode=False)
    fig.update_xaxes(showticklabels=False, fixedrange=True)
    fig.update_yaxes(showticklabels=False, fixedrange=True)

    fig.show(
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["zoom", "pan", "toImage"],
        }
    )

    render = plotly.io.to_html(fig, include_plotlyjs=False)

    print(len(render))

    fk.Deck("my_plot", html=render)
