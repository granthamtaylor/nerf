import flytekit

import plotly
import plotly.express as px

from nerf.orchestration import image

@flytekit.task(container_image=image, enable_deck=True)
def animate(results: list[tuple[float, float]]):
    # Extract x and y values from the list of tuples
    x_values = [result[0] for result in results]
    y_values = [result[1] for result in results]
    
    # Create a scatter plot using Plotly
    fig = px.scatter(x=x_values, y=y_values, title="Scatter Plot of Results")
    
    # Return the plotly figure
    return fig