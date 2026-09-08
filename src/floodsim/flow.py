"""Flood-path tracing and water accumulation on a grid."""

import numpy as np

UNIT_SQUARE = (0.0, 1.0, 0.0, 1.0)  # x_min, x_max, y_min, y_max


def inside(point, boundary):
    """True if ``point`` lies inside the rectangular ``boundary``."""
    x_min, x_max, y_min, y_max = boundary
    return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max


def trace_path(start, tri, gradients, boundary=UNIT_SQUARE,
               step=0.01, max_steps=100, flat_tol=1e-6):
    """Trace the downhill path of a drop of water released at ``start``.

    At each step the drop moves ``step`` times the downhill vector of the
    triangle it is in. Tracing stops when the drop leaves the triangulated
    hull, reaches a flat triangle, would leave ``boundary``, or after
    ``max_steps``. The returned ``(k, 2)`` array always contains at least
    the start point.
    """
    current = np.asarray(start, dtype=float)
    path = [current]
    for _ in range(max_steps):
        simplex = tri.find_simplex(current)
        if simplex == -1:
            break  # outside the triangulation
        gradient = gradients[simplex]
        if np.linalg.norm(gradient) < flat_tol:
            break  # flat area
        nxt = current + step * gradient
        if not inside(nxt, boundary):
            break  # would leave the simulation area
        path.append(nxt)
        current = nxt
    return np.array(path)


def cell_edges(n_cells, boundary=UNIT_SQUARE):
    """Grid-line coordinates ``(x_edges, y_edges)`` for ``n_cells`` per side."""
    x_min, x_max, y_min, y_max = boundary
    return (np.linspace(x_min, x_max, n_cells + 1),
            np.linspace(y_min, y_max, n_cells + 1))


def accumulate(tri, gradients, n_cells=10, boundary=UNIT_SQUARE,
               rain_per_cell=10.0, **trace_kwargs):
    """Rain on every grid cell and collect the water where the paths stop.

    A path is traced from the centre of each of the ``n_cells x n_cells``
    cells. The rain of that cell (``rain_per_cell``) is added to the cell
    containing the path's last point.

    Returns ``(paths, water)`` where ``paths`` is the list of traced paths
    (row-major over the grid) and ``water[ix, iy]`` is the volume that ended
    in the cell with column ``ix`` and row ``iy``.
    """
    x_min, x_max, y_min, y_max = boundary
    x_edges, y_edges = cell_edges(n_cells, boundary)
    cell_w = (x_max - x_min) / n_cells
    cell_h = (y_max - y_min) / n_cells
    water = np.zeros((n_cells, n_cells))
    paths = []
    for iy in range(n_cells):
        for ix in range(n_cells):
            start = ((x_edges[ix] + x_edges[ix + 1]) / 2,
                     (y_edges[iy] + y_edges[iy + 1]) / 2)
            path = trace_path(start, tri, gradients, boundary, **trace_kwargs)
            paths.append(path)
            stop = path[-1]
            cx = min(int((stop[0] - x_min) / cell_w), n_cells - 1)
            cy = min(int((stop[1] - y_min) / cell_h), n_cells - 1)
            if 0 <= cx < n_cells and 0 <= cy < n_cells:
                water[cx, cy] += rain_per_cell
    return paths, water
