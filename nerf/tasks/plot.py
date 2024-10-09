import flytekit
import plotly.express as px
import plotly

from nerf.orchestration.images import image
from nerf.core.structs import Metric


@flytekit.task(container_image=image, enable_deck=True)
def plot(scores: list[Metric]):

    loss = [score.loss for score in scores]
    compression = [score.compression for score in scores]

    # Create a scatter plot using Plotly
    fig = px.scatter(x=loss, y=compression, title="Scatter Plot of Results")
    
    config={
        "displaylogo": False,
        "modeBarButtonsToRemove": ["zoom", "pan", "toImage"],
    }

    render = plotly.io.to_html(fig, config=config)
    
    flytekit.Deck("my_plot", html=render)
