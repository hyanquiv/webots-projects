from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import os

MAP_FILE = "map_points.txt"
ROBOT_FILE = "robot_pos.txt"
REFRESH_INTERVAL = 1000  # ms (1 seg)

app = Dash(__name__)
app.title = "LIDAR + GPS — Visualización en tiempo real"

app.layout = html.Div([
    html.H2("Mapa 2D — LIDAR + GPS", style={"textAlign": "center"}),
    dcc.Graph(id="live-map", style={"height": "80vh"}),
    dcc.Interval(id="update-interval", interval=REFRESH_INTERVAL, n_intervals=0)
])

def load_data():
    if os.path.exists(MAP_FILE) and os.path.getsize(MAP_FILE) > 0:
        data = pd.read_csv(MAP_FILE, sep=' ', header=None, names=['x', 'y'], on_bad_lines='skip')
    else:
        data = pd.DataFrame(columns=['x', 'y'])

    robot_x, robot_y = None, None
    if os.path.exists(ROBOT_FILE) and os.path.getsize(ROBOT_FILE) > 0:
        try:
            pos = pd.read_csv(ROBOT_FILE, sep=' ', header=None, names=['x', 'y'], on_bad_lines='skip')
            robot_x, robot_y = pos.iloc[-1]['x'], pos.iloc[-1]['y']
        except Exception:
            pass

    return data, robot_x, robot_y

@app.callback(Output("live-map", "figure"), Input("update-interval", "n_intervals"))
def update_graph(_):
    data, robot_x, robot_y = load_data()

    scatter_data = []
    if not data.empty:
        scatter_data.append(go.Scattergl(
            x=data["x"],
            y=data["y"],
            mode="markers",
            marker=dict(color="black", size=3),
            name="Obstáculos"
        ))

    if robot_x is not None and robot_y is not None:
        scatter_data.append(go.Scatter(
            x=[robot_x],
            y=[robot_y],
            mode="markers",
            marker=dict(color="red", size=12),
            name="Robot"
        ))

    if not data.empty:
        xmin, xmax = data["x"].min(), data["x"].max()
        ymin, ymax = data["y"].min(), data["y"].max()
    else:
        xmin, xmax, ymin, ymax = -2, 2, -2, 2

    fig = go.Figure(
        data=scatter_data,
        layout=go.Layout(
            xaxis=dict(range=[xmin - 0.5, xmax + 0.5], title="Posición X (m)", zeroline=True),
            yaxis=dict(range=[ymin - 0.5, ymax + 0.5], title="Posición Y (m)", zeroline=True, scaleanchor="x", scaleratio=1),
            title="Mapa 2D (Vista Global)",
            showlegend=True,
            template="plotly_white",
        )
    )

    return fig

if __name__ == "__main__":
    app.run(debug=True, port=8050)