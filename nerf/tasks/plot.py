import flytekit
import plotly.express as px
import plotly

from nerf.orchestration.constants import context
from nerf.core.structs import Metric

@context["basic"]
def plot(scores: list[Metric]):
    """Plot the image compression vs loss"""

    loss = [score.loss for score in scores]
    compression = [score.compression for score in scores]

    # Create a scatter plot using Plotly
    fig = px.scatter(
        x=loss,
        y=compression,
        title="Image Compression vs Loss",
        labels={"x": "Loss", "y": "Compression"}
    )

    config={
        "displaylogo": False,
        "modeBarButtonsToRemove": ["zoom", "pan", "toImage"],
    }

    render = plotly.io.to_html(fig, config=config)
    
    flytekit.Deck("my_plot", html=render)
