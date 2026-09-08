"""Synthetic terrain functions and 2D point sampling.

Every terrain is a function ``z = f(x, y)`` defined on the unit square.
The seven closed-form surfaces are the ones used in the thesis; ``perlin``
is the procedural terrain that the thesis proposed as future work.
"""

import numpy as np


def gaussian(x, y):
    """Single Gaussian peak in the centre."""
    return np.exp(-5 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))


def cone(x, y):
    """Simple hill: a circular cone."""
    return 1 - np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)


def peaks(x, y):
    """Two Gaussian peaks."""
    return (np.exp(-5 * ((x - 0.25) ** 2 + (y - 0.25) ** 2))
            + np.exp(-5 * ((x - 0.75) ** 2 + (y - 0.75) ** 2)))


def ridges(x, y):
    """Sinusoidal ridges."""
    return np.sin(5 * np.pi * x) * np.sin(5 * np.pi * y)


def crater(x, y):
    """Volcanic crater: a ring with a central depression."""
    r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    return np.exp(-5 * r) - np.exp(-20 * r)


def plateau(x, y):
    """Flat plateau with steep cliffs."""
    return np.tanh(10 * (0.5 - np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)))


def valley(x, y):
    """Central mountain with a valley carved into its north-west flank.

    This is the default terrain of the thesis (Part2.py).
    """
    return (np.exp(-5 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
            - 0.5 * np.exp(-5 * ((x - 0.25) ** 2 + (y - 0.75) ** 2)))


def perlin(seed=None, octaves=6):
    """Return a Perlin-noise height function (requires ``perlin-noise``)."""
    try:
        from perlin_noise import PerlinNoise
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'perlin' terrain needs the perlin-noise package: "
            "pip install perlin-noise") from exc
    noise = PerlinNoise(octaves=octaves, seed=seed)

    def height(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        flat = [noise([xi, yi]) for xi, yi in zip(x.ravel(), y.ravel())]
        return np.asarray(flat, dtype=float).reshape(x.shape)

    return height


TERRAINS = {
    "gaussian": gaussian,
    "cone": cone,
    "peaks": peaks,
    "ridges": ridges,
    "crater": crater,
    "plateau": plateau,
    "valley": valley,
    "perlin": perlin,
}


def names():
    """Names accepted by :func:`get_terrain`."""
    return list(TERRAINS)


def get_terrain(name, seed=None):
    """Return the height function ``f(x, y)`` for a terrain name."""
    if name not in TERRAINS:
        raise ValueError(f"unknown terrain {name!r}; choose from {names()}")
    if name == "perlin":
        return perlin(seed=seed)
    return TERRAINS[name]


def sample_points(n_points, seed=42):
    """Sample ``n_points`` uniformly in the unit square.

    Uses the legacy NumPy generator so that a seed reproduces exactly the
    point set of the original thesis script.
    """
    return np.random.RandomState(seed).rand(n_points, 2)
