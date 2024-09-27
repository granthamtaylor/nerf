import flytekit
from flytekit.types.file import FlyteFile

import plotly
import polars as pl
import numpy as np
import plotly.express as px
from itertools import product
from nerf.orchestration import image


@flytekit.task(container_image=image, enable_deck=True)
def animate(predictions: FlyteFile) -> int:
    df = pl.read_parquet(predictions.path).select(
        t=pl.col("epoch"),
        x=pl.col("coordinates").arr.first(),
        y=pl.col("coordinates").arr.last(),
        c=pl.col("color"),
    )

    # fill in missing
    maxes = df.select(pl.max("x"), pl.max("y"))
    max_x = maxes["x"][0] + 1
    max_y = maxes["y"][0] + 1

    cartesian_product = list(product(range(max_x), range(max_y)))
    xys = pl.DataFrame({"x": [x for x, y in cartesian_product], "y": [y for x, y in cartesian_product]}).sort("x", "y")

    joined = xys.join(df, on=["x", "y"], how="left", coalesce=True)
    filled = joined.fill_null(strategy="forward")

    final = filled.sort("t", "x", "y").group_by("t", "x").agg("c").sort("t", "x").group_by("t").agg("c").get_column("c")

    animation = np.array(final.to_list())

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

    render = plotly.io.to_html(fig)

    print(len(render))

    # flytekit.Deck("my_plot", html=render)

    return len(render)
