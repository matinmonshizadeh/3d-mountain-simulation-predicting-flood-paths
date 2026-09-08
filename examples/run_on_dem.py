#!/usr/bin/env python
"""EXPERIMENTAL: run the flood simulation on a real SRTM elevation tile.

The thesis used synthetic terrain functions only. This script crops a
window out of a GeoTIFF digital elevation model (DEM), rescales the crop
to the unit square with elevations normalised to [0, 1], and feeds it
through the unchanged floodsim pipeline. It needs the optional
``rasterio`` package and an SRTM 1-arc-second tile (not included in the
repository, ~26 MB each; see the README for where to download one).

Example (Derak mountain, north-west of Shiraz, tile N29E052):
    python examples/run_on_dem.py --dem N29E052.tif \
        --bbox 52.3963 29.7007 52.4159 29.7145 --out output/derak
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import floodsim  # noqa: E402
from floodsim import terrain, plot  # noqa: E402

DERAK_BBOX = (52.3963, 29.7007, 52.4159, 29.7145)  # lon_min lat_min lon_max lat_max


def load_dem_window(path, bbox):
    """Read the DEM pixels inside ``bbox`` as a float array (rows = north to south)."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise SystemExit("This example needs rasterio: pip install rasterio") from exc
    lon_min, lat_min, lon_max, lat_max = bbox
    with rasterio.open(path) as dataset:
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, dataset.transform)
        dem = dataset.read(1, window=window).astype(float)
        nodata = dataset.nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan
    if dem.size == 0 or np.isnan(dem).all():
        raise SystemExit("the bounding box contains no valid elevation data")
    dem = np.where(np.isnan(dem), np.nanmin(dem), dem)
    return dem


def dem_height_function(dem):
    """Nearest-pixel height lookup on the unit square, elevations scaled to [0, 1]."""
    rows, cols = dem.shape
    z_min, z_max = dem.min(), dem.max()
    scaled = (dem - z_min) / (z_max - z_min if z_max > z_min else 1.0)

    def height(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        col = np.clip(np.rint(x * (cols - 1)).astype(int), 0, cols - 1)
        row = np.clip(np.rint((1 - y) * (rows - 1)).astype(int), 0, rows - 1)
        return scaled[row, col]

    return height


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dem", required=True, help="GeoTIFF DEM tile, e.g. an SRTM 1-arcsec tile")
    p.add_argument("--bbox", type=float, nargs=4, default=DERAK_BBOX,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                   help="crop window in degrees (default: Derak, Shiraz)")
    p.add_argument("--points", type=int, default=400)
    p.add_argument("--grid", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rain", type=float, default=10.0)
    p.add_argument("--out", default="output/dem")
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dem = load_dem_window(args.dem, args.bbox)
    print(f"DEM window: {dem.shape[1]}x{dem.shape[0]} pixels, "
          f"elevation {dem.min():.0f}-{dem.max():.0f} m")
    height_fn = dem_height_function(dem)
    points = terrain.sample_points(args.points, args.seed)
    result = floodsim.simulate(height_fn, points, args.grid, args.rain)
    files = plot.save_figures(result, args.out, show=args.show, label="real DEM, experimental")
    ix, iy = result.wettest_cell
    print(f"traced {len(result.paths)} paths, {result.water.sum():.0f} L of rain; "
          f"wettest cell column {ix}, row {iy} with {result.water.max():.0f} L")
    for f in files:
        print(f"saved {f}")


if __name__ == "__main__":
    main()
