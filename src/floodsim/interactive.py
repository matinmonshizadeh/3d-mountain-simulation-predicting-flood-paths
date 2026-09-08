"""Optional interactive 3D view exported as an HTML file (needs plotly).

Matplotlib renders 3D in software and re-projects every triangle on each
mouse move, so rotating the terrain window is slow. This module writes the
same scene to a self-contained HTML page rendered with WebGL, which stays
smooth in any browser.
"""

from pathlib import Path

import numpy as np

from .flow import cell_edges

INTERACTIVE_FIGURE = "terrain_flood_paths.html"


def _polyline_trace(go, segments, name, color, width, dash=None):
    """One Scatter3d trace for many polylines, separated by None gaps."""
    xs, ys, zs = [], [], []
    for seg in segments:
        xs += list(seg[:, 0]) + [None]
        ys += list(seg[:, 1]) + [None]
        zs += list(seg[:, 2]) + [None]
    line = dict(color=color, width=width)
    if dash:
        line["dash"] = dash
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=line,
                        name=name, hoverinfo="skip")


def save_interactive(result, out_dir, title="3D terrain with flood paths"):
    """Write ``terrain_flood_paths.html`` for ``result`` and return its path."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the interactive view needs plotly: "
                          "pip install plotly") from exc

    pts, z, tri = result.points, result.z, result.tri
    simplices = tri.simplices
    surface = go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=z,
        i=simplices[:, 0], j=simplices[:, 1], k=simplices[:, 2],
        intensity=z, colorscale="Earth", opacity=0.9, name="terrain",
        colorbar=dict(title="Elevation"), showscale=True, flatshading=True)

    x_edges, y_edges = cell_edges(result.n_cells, result.boundary)
    x_grid, y_grid = np.meshgrid(x_edges, y_edges)
    z_grid = result.height_fn(x_grid, y_grid)
    grid = [np.c_[x_grid[i], y_grid[i], z_grid[i]] for i in range(x_grid.shape[0])]
    grid += [np.c_[x_grid[:, j], y_grid[:, j], z_grid[:, j]] for j in range(x_grid.shape[1])]
    paths = [np.c_[p[:, 0], p[:, 1], result.height_fn(p[:, 0], p[:, 1])]
             for p in result.paths]

    fig = go.Figure([
        surface,
        _polyline_trace(go, grid, "grid", "black", 1.5, dash="dash"),
        _polyline_trace(go, paths, "flood paths", "blue", 4),
    ])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Elevation (Z)",
                   aspectmode="cube"),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=40, b=0))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / INTERACTIVE_FIGURE
    fig.write_html(html_path, include_plotlyjs="cdn")
    return html_path
