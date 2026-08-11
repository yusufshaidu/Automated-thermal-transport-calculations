#!/usr/bin/env python
"""Extract the lowest-F(T), stable FC2 from a saved SCPH run (fcp_scph/scph_T{T}_iter*.fcp
checkpoints, no new MLIP calls) and, optionally, pair it with an FC3 computed by a
separate finite-displacement thermal_transport_agent.py run for BTE. See README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

from generate_scph_fc2_fc3_agent import (
    phonopysupercell,
    parse_primitive_matrix,
    parse_sdim,
)
from plot_scph_free_energy import (
    find_scph_checkpoints,
    analyze_fcp,
    load_fc2_into_phonon,
)
from phono3py.file_IO import write_fc2_to_hdf5, read_fc3_from_hdf5


def select_best_checkpoint(
    outdir:        str,
    T:             float,
    supercell,
    phonon,
    fe_mesh:       list[int],
    stability_tol: float,
    classical:     bool,
) -> tuple[dict, list[dict]]:
    """Rank every saved checkpoint for *T* by F(T); return (selected, all_candidates)."""
    checkpoints = find_scph_checkpoints(outdir, T)
    candidates = []
    for it, path in checkpoints:
        fe, min_freq = analyze_fcp(path, supercell, phonon, fe_mesh, T, classical)
        label = "final" if not np.isfinite(it) else f"iter {int(it)}"
        stable = min_freq > stability_tol
        print(f"  {label:>10s}  F={fe:.6f} kJ/mol  min_freq={min_freq:.4f} THz  "
              f"stable={stable}  ({os.path.basename(path)})")
        candidates.append({
            "iteration": None if not np.isfinite(it) else int(it),
            "path": path,
            "free_energy_kJ_per_mol": fe,
            "min_frequency_THz": min_freq,
            "stable": stable,
        })

    stable = [c for c in candidates if c["stable"]]
    if not stable:
        worst = min(candidates, key=lambda c: c["min_frequency_THz"])
        worst_label = "final" if worst["iteration"] is None else f"iteration {worst['iteration']}"
        fcp_dir = os.path.join(outdir, f"T{T:.0f}", "fcp_scph")
        raise RuntimeError(
            f"No stable SCPH checkpoint found for T={T:.0f}K among "
            f"{len(candidates)} candidate(s) in {fcp_dir} "
            f"(stability_tol={stability_tol} THz); least-imaginary was "
            f"{worst_label} with min_freq={worst['min_frequency_THz']:.4f} THz. "
            f"Rerun the SCPH loop with more iterations, a larger --fe_mesh, a "
            f"looser --stability_tol, or inspect the run with plot_scph_free_energy.py."
        )

    best = min(stable, key=lambda c: c["free_energy_kJ_per_mol"])
    return best, candidates


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract the lowest-F(T), stable FC2 from saved SCPH checkpoints "
                     "(without disturbing the joint FC2..FC{n} fit's fc2.hdf5), and "
                     "optionally assemble it with an FC3 from a separate "
                     "finite-displacement thermal_transport_agent.py run into a "
                     "directory ready for `thermal_transport_agent.py bte`.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-prim", "--prim_file", required=True,
                   help="Primitive structure file (same one used for the SCPH run)")
    p.add_argument("-sdim", "--sdim", default="2 2 2",
                   help="Supercell dimensions (must match the SCPH run)")
    p.add_argument("-pa", "--primitive_matrix", default="auto",
                   help="Primitive matrix (must match the SCPH run)")
    p.add_argument("-tolerance", "--symprec", type=float, default=1e-3,
                   help="Symmetry precision -- must match the -tolerance/--symprec "
                        "used for the SCPH run being inspected")
    p.add_argument("-o", "--outdir", required=True,
                   help="Output directory passed to generate_scph_fc2_fc3_agent.py (-o/--outdir)")
    p.add_argument("-temps", "--temperatures", required=True,
                   help="Space-separated temperatures (K) to process, e.g. \"100 200 300\"")
    p.add_argument("--fe_mesh", default="10 10 10",
                   help="q-point mesh for the per-checkpoint free energy calc "
                        "(match the --fe_mesh used with --select_best_iteration, "
                        "if any, for a consistent ranking)")
    p.add_argument("--stability_tol", type=float, default=-0.01,
                   help="A mesh frequency (THz) below this (negative) tolerance "
                        "marks a checkpoint as imaginary/unstable")
    p.add_argument("--qm_statistics", action="store_true", default=False,
                   help="Quantum Bose-Einstein statistics for F(T) (default: "
                        "classical). Match whatever the SCPH run used.")
    p.add_argument("--out_name", default="fc2_scph_best.hdf5",
                   help="Filename (within outdir/T{T}/) for the extracted FC2 -- "
                        "deliberately distinct from fc2.hdf5, which holds the "
                        "joint FC2..FC{n} refit and is never touched by this script")
    p.add_argument("--fc3_dir", default=None,
                   help="A finite-displacement thermal_transport_agent.py --out_dir "
                        "containing fc3.hdf5. If given, that FC3 is paired with the "
                        "extracted FC2 in --combined_dir, ready for `bte`.")
    p.add_argument("--combined_dir", default=None,
                   help="Where to assemble fc2.hdf5+fc3.hdf5 for BTE. Default: "
                        "outdir/T{T}/combined_scph_fc2_finite_fc3. Ignored without --fc3_dir.")
    p.add_argument("--copy", action="store_true",
                   help="Copy FC2/FC3 into --combined_dir instead of symlinking "
                        "(symlink is the default -- fc3.hdf5 can be large, and "
                        "symlinks keep provenance to the source run obvious)")

    args = p.parse_args()

    sdim             = parse_sdim(args.sdim)
    primitive_matrix = parse_primitive_matrix(args.primitive_matrix)
    fe_mesh          = [int(x) for x in args.fe_mesh.split()]
    temperatures     = [float(x) for x in args.temperatures.split()]
    classical        = not args.qm_statistics

    _, supercell, _, phonon = phonopysupercell(
        args.prim_file, sdim, primitive_matrix, args.symprec
    )
    N = len(supercell)

    for T in temperatures:
        print(f"\n=== T = {T:.0f} K ===")
        best, candidates = select_best_checkpoint(
            args.outdir, T, supercell, phonon, fe_mesh, args.stability_tol, classical
        )
        best_label = "final" if best["iteration"] is None else f"iteration {best['iteration']}"
        print(f"  -> selected {best_label}  F={best['free_energy_kJ_per_mol']:.6f} kJ/mol  "
              f"min_freq={best['min_frequency_THz']:.4f} THz  "
              f"({os.path.basename(best['path'])})")

        load_fc2_into_phonon(best["path"], supercell, phonon)
        fc2 = phonon.force_constants   # symmetrized by load_fc2_into_phonon

        T_dir = os.path.join(args.outdir, f"T{T:.0f}")
        os.makedirs(T_dir, exist_ok=True)
        fc2_out_path = os.path.join(T_dir, args.out_name)
        write_fc2_to_hdf5(fc2, filename=fc2_out_path)
        print(f"  Wrote best-iteration FC2 {fc2.shape} -> {fc2_out_path}  "
              f"(fc2.hdf5 in this dir, if present, is the joint FC2..FC{{n}} refit "
              f"and is untouched)")

        summary_path = os.path.join(
            T_dir, os.path.splitext(args.out_name)[0] + ".json"
        )
        with open(summary_path, "w") as f:
            json.dump({
                "temperature":     T,
                "fe_mesh":         fe_mesh,
                "classical":       classical,
                "stability_tol":   args.stability_tol,
                "selected":        best,
                "candidates":      candidates,
            }, f, indent=2)
        print(f"  Wrote selection summary -> {summary_path}")

        if not args.fc3_dir:
            continue

        fc3_src = os.path.join(args.fc3_dir, "fc3.hdf5")
        if not os.path.exists(fc3_src):
            raise FileNotFoundError(f"--fc3_dir {args.fc3_dir!r} has no fc3.hdf5")

        fc3 = read_fc3_from_hdf5(filename=fc3_src)
        if fc3.shape[0] != N:
            raise ValueError(
                f"Supercell size mismatch: the SCPH FC2 has N={N} atoms (from "
                f"-prim/-sdim/-pa/-tolerance), but the FC3 at {fc3_src} has "
                f"N={fc3.shape[0]} atoms. FC2 and FC3 must come from the "
                f"identical supercell (same size, primitive orientation, and "
                f"symprec) -- check that the finite-displacement run's "
                f"--supercell/--primitive_matrix/--symprec matched -sdim/-pa/"
                f"-tolerance used here."
            )

        combined_dir = args.combined_dir or os.path.join(
            T_dir, "combined_scph_fc2_finite_fc3"
        )
        os.makedirs(combined_dir, exist_ok=True)
        fc2_dst = os.path.join(combined_dir, "fc2.hdf5")
        fc3_dst = os.path.join(combined_dir, "fc3.hdf5")

        for src, dst in [(fc2_out_path, fc2_dst), (fc3_src, fc3_dst)]:
            if os.path.lexists(dst):
                os.remove(dst)
            if args.copy:
                shutil.copy2(src, dst)
            else:
                os.symlink(os.path.abspath(src), dst)

        verb = "Copied" if args.copy else "Symlinked"
        print(f"  {verb} FC2 ({fc2_out_path}) and FC3 ({fc3_src}) -> {combined_dir}/")
        print(f"""
  Run BTE on the combined FC2/FC3:
    python thermal_transport_agent.py bte \\
        --out_dir {combined_dir}/ \\
        --structure {args.prim_file} --supercell "{args.sdim}" \\
        --primitive_matrix "{args.primitive_matrix}" --symprec {args.symprec} \\
        --mesh "11 11 11" --temperatures "{T:.0f}" \\
        --solver rta --transport_type SMM19

  NOTE: --structure/--supercell/--primitive_matrix/--symprec above must match
  BOTH the SCPH run that produced the FC2 AND the finite-displacement run
  ({args.fc3_dir}) that produced the FC3 -- an N-atom match (checked above)
  does not guarantee the same primitive orientation/atom ordering.
""")


if __name__ == "__main__":
    main()
