"""Figures: 3D terrain with flood paths, and the water-volume heatmap."""

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .flow import UNIT_SQUARE, cell_edges

TERRAIN_FIGURE = "terrain_flood_paths.png"
WATER_FIGURE = "water_volume.png"


def plot_terrain_with_paths(ax, points, z, tri, height_fn, paths,
                            n_cells=10, boundary=UNIT_SQUARE,
                            title="3D terrain with flood paths"):
    """Draw the triangulated surface, the grid overlay and the paths."""
    ax.plot_trisurf(points[:, 0], points[:, 1], z, triangles=tri.simplices,
                    cmap="terrain", edgecolor="none", alpha=0.8)

    x_edges, y_edges = cell_edges(n_cells, boundary)
    x_grid, y_grid = np.meshgrid(x_edges, y_edges)
    z_grid = height_fn(x_grid, y_grid)
    for i in range(x_grid.shape[0]):
        ax.plot(x_grid[i, :], y_grid[i, :], z_grid[i, :],
                color="black", linestyle="--", linewidth=0.5)
    for j in range(x_grid.shape[1]):
        ax.plot(x_grid[:, j], y_grid[:, j], z_grid[:, j],
                color="black", linestyle="--", linewidth=0.5)

    for path in paths:
        path_z = height_fn(path[:, 0], path[:, 1])
        ax.plot(path[:, 0], path[:, 1], path_z,
                color="blue", linewidth=1, alpha=0.7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation (Z)")
    ax.set_title(title)


def plot_water_volume(ax, water, boundary=UNIT_SQUARE,
                      title="Accumulated water volume (litres)"):
    """Heatmap of the water grid, in the same X/Y frame as the 3D plot.

    ``water[ix, iy]`` is indexed column-first, so it is transposed for
    ``imshow`` (rows = y) and drawn with ``origin="lower"`` so that y
    increases upwards exactly as in the 3D view.
    """
    x_min, x_max, y_min, y_max = boundary
    image = ax.imshow(water.T, cmap="Blues", interpolation="nearest",
                      origin="lower", extent=(x_min, x_max, y_min, y_max))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, label="Water volume (litres)")
    return image


def save_figures(result, out_dir, show=False, label=None):
    """Save the two figures to ``out_dir`` and return their paths.

    ``result`` is a :class:`floodsim.SimulationResult`. ``label`` is
    appended to the figure titles (for example the terrain name).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f" ({label})" if label else ""

    fig1 = plt.figure(figsize=(8, 7))
    ax1 = fig1.add_subplot(111, projection="3d")
    plot_terrain_with_paths(ax1, result.points, result.z, result.tri,
                            result.height_fn, result.paths, result.n_cells,
                            result.boundary,
                            title="3D terrain with flood paths" + suffix)
    fig1.tight_layout()
    terrain_path = out_dir / TERRAIN_FIGURE
    fig1.savefig(terrain_path, dpi=150)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    plot_water_volume(ax2, result.water, result.boundary,
                      title="Accumulated water volume (litres)" + suffix)
    fig2.tight_layout()
    water_path = out_dir / WATER_FIGURE
    fig2.savefig(water_path, dpi=150)

    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig1)
    plt.close(fig2)
    return terrain_path, water_path
