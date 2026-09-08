"""floodsim: flood-path prediction on a triangulated synthetic terrain.

Pipeline (Bachelor's thesis, Shiraz University, 2024):

1. sample random 2D points and lift them with a terrain function
2. Delaunay-triangulate the points
3. fit a plane to every triangle to get its downhill direction
4. trace a path from the centre of every grid cell by gradient descent
5. add each cell's rain to the cell where its path stops
"""

from dataclasses import dataclass

import numpy as np

from . import terrain, mesh, flow, plot  # noqa: F401
from .flow import UNIT_SQUARE

__all__ = ["SimulationResult", "simulate", "terrain", "mesh", "flow", "plot"]


@dataclass
class SimulationResult:
    height_fn: object          # callable z = f(x, y)
    points: np.ndarray         # (n, 2) sampled points
    z: np.ndarray              # (n,) elevations at the points
    tri: object                # scipy.spatial.Delaunay
    gradients: np.ndarray      # (n_triangles, 2) downhill vectors
    paths: list                # one (k, 2) array per grid cell
    water: np.ndarray          # (n_cells, n_cells) litres, indexed [ix, iy]
    n_cells: int
    boundary: tuple = UNIT_SQUARE

    @property
    def wettest_cell(self):
        """(ix, iy) of the cell holding the most water."""
        return tuple(int(v) for v in np.unravel_index(
            np.argmax(self.water), self.water.shape))


def simulate(height_fn, points, n_cells=10, rain_per_cell=10.0,
             boundary=UNIT_SQUARE, **trace_kwargs):
    """Run the full pipeline for a height function and a set of 2D points."""
    points = np.asarray(points, dtype=float)
    z = height_fn(points[:, 0], points[:, 1])
    tri = mesh.triangulate(points)
    gradients = mesh.triangle_gradients(tri, z)
    paths, water = flow.accumulate(tri, gradients, n_cells, boundary,
                                   rain_per_cell, **trace_kwargs)
    return SimulationResult(height_fn, points, z, tri, gradients,
                            paths, water, n_cells, boundary)
