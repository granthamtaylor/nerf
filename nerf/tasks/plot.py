import flytekit

import plotly
import plotly.express as px

from nerf.orchestration import image
from nerf.core.structs import Metric


@flytekit.task(container_image=image, enable_deck=True)
def plot(scores: list[Metric]):

    loss = [score.loss for score in scores]
    compression = [score.compression for score in scores]

    # Create a scatter plot using Plotly
    fig = px.scatter(x=loss, y=compression, title="Scatter Plot of Results")
    
    fig.show(
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["zoom", "pan", "toImage"],
        }
    )
    
    # render = plotly.io.to_html(fig)
    
    # print(len(render))
