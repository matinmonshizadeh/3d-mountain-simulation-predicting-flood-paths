#!/usr/bin/env python
"""Simulate rain on a synthetic terrain and save the two result figures.

Example:
    python run.py --terrain valley --points 400 --grid 10 --seed 42 --out docs/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import floodsim  # noqa: E402
from floodsim import terrain, plot  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--terrain", default="valley", choices=terrain.names(),
                   help="terrain function (default: valley)")
    p.add_argument("--points", type=int, default=400,
                   help="number of random 2D sample points (default: 400)")
    p.add_argument("--grid", type=int, default=10,
                   help="grid cells per side (default: 10)")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for point sampling (default: 42)")
    p.add_argument("--rain", type=float, default=10.0,
                   help="litres of rain per grid cell (default: 10)")
    p.add_argument("--out", default="docs/",
                   help="directory for the PNG figures (default: docs/)")
    p.add_argument("--show", action="store_true",
                   help="also open the figures in a window")
    p.add_argument("--html", action="store_true",
                   help="also write an interactive 3D view as HTML (needs plotly)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    height_fn = terrain.get_terrain(args.terrain, seed=args.seed)
    points = terrain.sample_points(args.points, args.seed)
    result = floodsim.simulate(height_fn, points, args.grid, args.rain)

    files = plot.save_figures(result, args.out, show=args.show,
                              label=args.terrain)
    ix, iy = result.wettest_cell
    print(f"terrain={args.terrain} points={args.points} grid={args.grid}x{args.grid} seed={args.seed}")
    print(f"traced {len(result.paths)} paths, {result.water.sum():.0f} L of rain")
    print(f"wettest cell: column {ix}, row {iy} with {result.water.max():.0f} L")
    for f in files:
        print(f"saved {f}")
    if args.html:
        from floodsim.interactive import save_interactive
        html_path = save_interactive(
            result, args.out, title=f"3D terrain with flood paths ({args.terrain})")
        print(f"saved {html_path}")
        if args.show:
            import webbrowser
            webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
