"""Overlay phonon band structures from saved SCPH .fcp checkpoints on one high-symmetry path."""

from __future__ import annotations

import argparse
import os

import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from phonopy.phonon.band_structure import BandPlot, get_band_qpoints_by_seekpath

from generate_scph_fc2_fc3_agent import (
    phonopysupercell,
    parse_primitive_matrix,
    parse_sdim,
)
from plot_scph_free_energy import find_scph_checkpoints, load_fc2_into_phonon


def plot_bands_vs_iteration(
    phonon,
    supercell,
    checkpoints: list[tuple[float, str]],
    npoints:     int,
    out_png:     str,
    T:           float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands, labels, path_connections = get_band_qpoints_by_seekpath(
        phonon.primitive, npoints, is_const_interval=True
    )
    n_panels = len([c for c in path_connections if not c])

    fig = plt.figure(figsize=(2.5 * n_panels + 2, 4))
    axs = ImageGrid(fig, 111, nrows_ncols=(1, n_panels), axes_pad=0.11, label_mode="L")
    bp = BandPlot(axs)

    finite_iters = [it for it, _ in checkpoints if np.isfinite(it)]
    norm = plt.Normalize(vmin=min(finite_iters), vmax=max(max(finite_iters), 1))
    cmap = plt.get_cmap("viridis")

    decorated = False
    for it, path in checkpoints:
        load_fc2_into_phonon(path, supercell, phonon)
        phonon.run_band_structure(
            bands, path_connections=path_connections, labels=labels, is_legacy_plot=False
        )
        bs = phonon.band_structure

        if not decorated:
            bp.decorate(labels, path_connections, bs.frequencies, bs.distances)
            decorated = True

        is_final = not np.isfinite(it)
        color = "black" if is_final else cmap(norm(it))
        lw    = 1.8 if is_final else 1.0
        zorder = 3 if is_final else 2

        distances_scaled = [d * bp.xscale for d in bs.distances]
        count = 0
        for d, f, c in zip(distances_scaled, bs.frequencies, path_connections):
            axs[count].plot(d, f, color=color, linewidth=lw, zorder=zorder)
            if not c:
                count += 1

    axs[0].set_ylabel("Frequency [THz]")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=list(axs), fraction=0.05, pad=0.02, label="SCPH iteration")
    fig.suptitle(f"T = {T:.0f} K  (black = final)")

    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Overlay phonon band structures from saved SCPH .fcp checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-prim", "--prim_file", required=True,
                   help="Primitive structure file (same one used for the SCPH run)")
    p.add_argument("-sdim", "--sdim", default="2 2 2",
                   help="Supercell dimensions (must match the SCPH run)")
    p.add_argument("-pa", "--primitive_matrix", default="auto",
                   help="Primitive matrix (must match the SCPH run)")
    p.add_argument("-tolerance", "--symprec", type=float, default=1e-3,
                   help="Symmetry precision -- must match the -tolerance/"
                        "--symprec used for the SCPH run being inspected")
    p.add_argument("-o", "--outdir", required=True,
                   help="Output directory passed to generate_scph_fc2_fc3_agent.py (-o/--outdir)")
    p.add_argument("-temps", "--temperatures", required=True,
                   help="Space-separated temperatures (K) to plot, e.g. \"100 200 300\"")
    p.add_argument("--npoints", type=int, default=50,
                   help="q-points per high-symmetry path segment (seekpath)")
    p.add_argument("--stride", type=int, default=1,
                   help="Plot every Nth checkpointed iteration (final is always included)")

    args = p.parse_args()

    sdim             = parse_sdim(args.sdim)
    primitive_matrix = parse_primitive_matrix(args.primitive_matrix)
    temperatures     = [float(x) for x in args.temperatures.split()]

    _, supercell, _, phonon = phonopysupercell(
        args.prim_file, sdim, primitive_matrix, args.symprec
    )

    for T in temperatures:
        print(f"\nT = {T:.0f} K")
        checkpoints = find_scph_checkpoints(args.outdir, T)
        checkpoints = [c for i, c in enumerate(checkpoints)
                       if i % args.stride == 0 or not np.isfinite(c[0])]

        out_png = os.path.join(args.outdir, f"T{T:.0f}", "bands_vs_scph_iteration.png")
        plot_bands_vs_iteration(phonon, supercell, checkpoints, args.npoints, out_png, T)


if __name__ == "__main__":
    main()
