"""Delaunay triangulation and per-triangle gradients."""

import numpy as np
from scipy.spatial import Delaunay


def triangulate(points):
    """Delaunay-triangulate an ``(n, 2)`` array of 2D points."""
    return Delaunay(np.asarray(points)[:, :2])


def triangle_gradients(tri, z):
    """Downhill direction of every triangle in the mesh.

    A plane ``z = a*x + b*y + c`` is fitted by least squares through the
    three vertices of each triangle. The returned array holds ``(-a, -b)``
    per triangle: the negative gradient, which points downhill.
    """
    gradients = np.zeros((len(tri.simplices), 2))
    for i, simplex in enumerate(tri.simplices):
        pts = tri.points[simplex]
        A = np.c_[pts[:, 0], pts[:, 1], np.ones(3)]
        coef, _, _, _ = np.linalg.lstsq(A, z[simplex], rcond=None)
        gradients[i] = -coef[:2]
    return gradients
