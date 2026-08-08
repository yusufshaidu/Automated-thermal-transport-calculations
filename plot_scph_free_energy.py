"""Plot harmonic free energy F(T) vs. SCPH iteration from saved .fcp checkpoints.

python plot_scph_free_energy.py -prim POSCAR-unitcell -sdim "2 2 2" \\
    -o output/ -temps "100 200 300" --mesh "20 20 20"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from generate_scph_fc2_fc3_agent import (
    phonopysupercell,
    parse_primitive_matrix,
    parse_sdim,
)
from hiphive import ForceConstantPotential


_ITER_RE = re.compile(r"_iter(\d+)\.fcp$")


def find_scph_checkpoints(outdir: str, T: float) -> list[tuple[float, str]]:
    """(iteration, path) for each checkpoint; final.fcp gets iteration=inf, sorts last."""
    fcp_dir = os.path.join(outdir, f"T{T:.0f}", "fcp_scph")
    found: list[tuple[float, str]] = []

    for path in glob.glob(os.path.join(fcp_dir, f"scph_T{T:.0f}_iter*.fcp")):
        m = _ITER_RE.search(path)
        if m:
            found.append((float(m.group(1)), path))

    final_path = os.path.join(fcp_dir, f"scph_T{T:.0f}_final.fcp")
    if os.path.exists(final_path):
        found.append((float("inf"), final_path))

    if not found:
        raise FileNotFoundError(
            f"No scph_T{T:.0f}_iter*.fcp or scph_T{T:.0f}_final.fcp files "
            f"found in {fcp_dir!r}."
        )

    found.sort(key=lambda x: x[0])
    return found


def free_energy_from_fcp(
    fcp_path:  str,
    supercell,
    phonon,
    mesh:      list[int],
    T:         float,
    classical: bool,
) -> float:
    """Return the harmonic free energy (kJ/mol) at T from a saved FCP's FC2."""
    N = len(supercell)

    fcp = ForceConstantPotential.read(fcp_path)

    # fcp.primitive_structure is hiphive's symmetry-reduced primitive (e.g. 1
    # atom for fcc), not the ClusterSpace supercell — not comparable to N.
    # The real shape check is on the FC2 array itself, below.
    fc2 = fcp.get_force_constants(supercell).get_fc_array(order=2, format="phonopy")
    if fc2.shape != (N, N, 3, 3):
        raise ValueError(
            f"{fcp_path}: FC2 array has shape {fc2.shape}, expected "
            f"({N}, {N}, 3, 3) to match the supercell built from "
            "-prim/-sdim/-pa. Make sure these flags match the "
            "generate_scph_fc2_fc3_agent.py run that produced this checkpoint."
        )

    phonon.force_constants = fc2
    phonon.symmetrize_force_constants()
    phonon.run_mesh(mesh, is_gamma_center=True)
    phonon.run_thermal_properties(temperatures=[T], classical=classical)

    return float(phonon.thermal_properties.free_energy[0])   # kJ/mol


def plot_convergence(results: dict, outdir: str, classical: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for T, points in results.items():
        iters = [p[0] for p in points if np.isfinite(p[0])]
        fe    = [p[1] for p in points if np.isfinite(p[0])]
        final = next((p[1] for p in points if not np.isfinite(p[0])), None)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(iters, fe, "o-", label="SCPH iteration")
        if final is not None:
            ax.axhline(final, color="k", ls="--", lw=1, label="final (collected fit)")
        ax.set_xlabel("SCPH iteration")
        ax.set_ylabel("Free energy F(T) [kJ/mol]")
        stat = "classical" if classical else "quantum"
        ax.set_title(f"T = {T:.0f} K  ({stat} statistics)")
        ax.legend()
        fig.tight_layout()

        out_png = os.path.join(outdir, f"T{T:.0f}", "free_energy_vs_scph_iteration.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot harmonic free energy vs. SCPH iteration from saved .fcp checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-prim", "--prim_file", required=True,
                   help="Primitive structure file (same one used for the SCPH run)")
    p.add_argument("-sdim", "--sdim", default="2 2 2",
                   help="Supercell dimensions (must match the SCPH run)")
    p.add_argument("-pa", "--primitive_matrix", default="auto",
                   help="Primitive matrix (must match the SCPH run)")
    p.add_argument("-o", "--outdir", required=True,
                   help="Output directory passed to generate_scph_fc2_fc3_agent.py (-o/--outdir)")
    p.add_argument("-temps", "--temperatures", required=True,
                   help="Space-separated temperatures (K) to plot, e.g. \"100 200 300\"")
    p.add_argument("--mesh", default="20 20 20",
                   help="q-point mesh for the thermal-properties calculation")
    p.add_argument("--classical", action="store_true",
                   help="Use classical (Boltzmann) statistics instead of quantum (Bose-Einstein)")

    args = p.parse_args()

    sdim             = parse_sdim(args.sdim)
    primitive_matrix = parse_primitive_matrix(args.primitive_matrix)
    mesh             = [int(x) for x in args.mesh.split()]
    temperatures     = [float(x) for x in args.temperatures.split()]

    _, supercell, _, phonon = phonopysupercell(
        args.prim_file, sdim, primitive_matrix
    )

    results: dict = {}
    for T in temperatures:
        print(f"\nT = {T:.0f} K")
        checkpoints = find_scph_checkpoints(args.outdir, T)
        points = []
        for it, path in checkpoints:
            fe = free_energy_from_fcp(
                path, supercell, phonon, mesh, T, args.classical
            )
            label = "final" if not np.isfinite(it) else f"iter {int(it)}"
            print(f"  {label:>10s}  F = {fe:.6f} kJ/mol   ({os.path.basename(path)})")
            points.append((it, fe))
        results[T] = points

        summary_path = os.path.join(args.outdir, f"T{T:.0f}", "free_energy_vs_scph_iteration.json")
        with open(summary_path, "w") as f:
            json.dump(
                {"temperature": T, "mesh": mesh, "classical": args.classical,
                 "points": [{"iteration": (None if not np.isfinite(it) else int(it)),
                             "free_energy_kJ_per_mol": fe} for it, fe in points]},
                f, indent=2,
            )
        print(f"  wrote {summary_path}")

    plot_convergence(results, args.outdir, args.classical)


if __name__ == "__main__":
    main()
