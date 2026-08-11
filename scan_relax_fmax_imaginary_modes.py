#!/usr/bin/env python
"""Relax a structure at several --fmax thresholds (and, optionally, several random
position-noise magnitudes) and check the impact on harmonic stability: for each
combination, relax independently from the (possibly rattled) input structure, build a
finite-displacement FC2 (phonopy), and look for imaginary (negative) frequencies on a
dense q-point mesh. See README.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.filters import UnitCellFilter
from ase.io import read as ase_read, write as ase_write
from ase.optimize import BFGS
from phonopy import Phonopy

from phono3py.file_IO import write_fc2_to_hdf5

from thermal_transport_agent import Config, make_calc, ase_to_phonopy, phonopy_to_ase
from generate_scph_fc2_fc3_agent import parse_primitive_matrix, parse_sdim


def relax_at_fmax(atoms, calc, fmax: float, steps: int, pressure: float, logfile):
    """Variable-cell relax (cell lengths only, angles fixed -- matches thermal_transport_agent.py's stage_relax) to *fmax*. Returns the relaxed Atoms (still carrying *calc* and its cached results)."""
    atoms = atoms.copy()
    atoms.calc = calc
    p_eV = pressure * 0.00624
    filt = UnitCellFilter(
        atoms, mask=[True, True, True, False, False, False], scalar_pressure=p_eV
    )
    BFGS(filt, logfile=logfile).run(fmax=fmax, steps=steps)
    fmax_final = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
    return atoms, fmax_final, fmax_final < fmax


def min_nonacoustic_freq(phonon) -> float:
    """Min mesh frequency, excluding the 3 trivially ~0 acoustic bands at Gamma. Requires phonon.run_mesh() first."""
    freqs   = phonon.mesh.frequencies.copy()
    qpoints = phonon.mesh.qpoints
    gamma_rows = np.where(np.all(np.abs(qpoints) < 1e-8, axis=1))[0]
    for row in gamma_rows:
        acoustic = np.argsort(freqs[row])[:3]
        freqs[row, acoustic] = np.inf
    return float(np.min(freqs))


def build_fc2(atoms, calc, sdim, primitive_matrix, symprec: float, amplitude: float,
              log_prefix: str = "") -> tuple[Phonopy, float]:
    """Standard finite-displacement FC2 (phonopy), residual-force corrected against the perfect supercell."""
    unitcell = ase_to_phonopy(atoms)
    phonon = Phonopy(
        unitcell,
        supercell_matrix = np.diag(sdim),
        primitive_matrix = primitive_matrix,
        symprec          = symprec,
    )
    phonon.generate_displacements(distance=amplitude, is_plusminus=True, is_diagonal=True)

    supercells = phonon.supercells_with_displacements
    print(f"{log_prefix}  {len(supercells)} displaced supercells, amplitude={amplitude} Å")

    sc_perfect = phonopy_to_ase(phonon.supercell)
    sc_perfect.calc = calc
    f_perfect = sc_perfect.get_forces()
    resid = float(np.max(np.abs(f_perfect)))
    print(f"{log_prefix}  max residual force on perfect supercell: {resid:.3e} eV/Å"
          + ("  -- check relaxation!" if resid > 0.01 else ""))

    force_sets = []
    for i, sc in enumerate(supercells):
        a = phonopy_to_ase(sc)
        a.calc = calc
        force_sets.append(a.get_forces() - f_perfect)
        if (i + 1) % max(1, len(supercells) // 10) == 0:
            print(f"{log_prefix}    forces {i + 1}/{len(supercells)}")

    phonon.forces = force_sets
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    return phonon, resid


def plot_summary(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    noise_levels = sorted({r["noise_stdev"] for r in rows})
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k, noise in enumerate(noise_levels):
        group = sorted((r for r in rows if r["noise_stdev"] == noise),
                        key=lambda r: r["target_fmax"])
        fmax  = [r["target_fmax"] for r in group]
        minf  = [r["min_frequency_THz"] for r in group]
        color = cmap(k / max(1, len(noise_levels) - 1))
        ax.plot(fmax, minf, "-", color=color, lw=1, zorder=2,
                label=f"noise={noise:g} Å")
        edge_colors = ["tab:red" if r["has_imaginary_modes"] else color for r in group]
        ax.scatter(fmax, minf, c=[color] * len(group), edgecolors=edge_colors,
                   linewidths=1.5, zorder=3)
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Relaxation fmax threshold (eV/Å)")
    ax.set_ylabel("Min non-acoustic frequency (THz)")
    ax.set_title("Imaginary modes vs. relaxation tightness (red outline = imaginary)")
    if len(noise_levels) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = out_dir / "relax_scan_summary.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Relax a structure at several --fmax thresholds, build a "
                     "finite-displacement FC2 for each, and check for imaginary "
                     "(negative) frequencies on a dense mesh -- to see how tightly "
                     "a relaxation needs to converge before the harmonic spectrum "
                     "is trustworthy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--structure", required=True, help="Input structure (e.g. POSCAR)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--fmax", required=True,
                   help="Space-separated list of relaxation force thresholds (eV/Å) "
                        "to test independently from --structure, e.g. "
                        "\"0.05 0.01 0.005 0.001 0.0001\"")
    p.add_argument("--relax_steps", type=int, default=500)
    p.add_argument("--relax_pressure", type=float, default=0.0, help="Target pressure (GPa)")
    p.add_argument("--noise", default="0.0",
                   help="Space-separated list of Gaussian position-noise magnitudes "
                        "(Å stdev) to rattle --structure with before each relaxation "
                        "(ase.Atoms.rattle). A symmetric structure can sit exactly at "
                        "a saddle point that symmetry-preserving relaxation can never "
                        "leave, however tight --fmax gets; rattling it first tests "
                        "whether an imaginary mode is a genuine instability or an "
                        "artifact of that symmetry. Every (fmax, noise) combination "
                        "is run. Default: no noise.")
    p.add_argument("--noise_seed", type=int, default=0,
                   help="Seed for --noise (same seed reused across --fmax so every "
                        "threshold at a given noise level starts from the identical "
                        "rattled structure)")

    p.add_argument("--supercell", default="2 2 2")
    p.add_argument("--primitive_matrix", default="auto")
    p.add_argument("--symprec", type=float, default=1e-5)
    p.add_argument("--amplitude", type=float, default=0.03, help="FC2 displacement distance (Å)")
    p.add_argument("--mesh", default="20 20 20", help="Dense q-point mesh for the stability check")

    grp = p.add_argument_group("Calculator")
    grp.add_argument("--calc_type", default="mace", choices=["mace", "uma"])
    grp.add_argument("--mace_model", default="")
    grp.add_argument("--mace_head", default="")
    grp.add_argument("--mace_device", default="cuda")
    grp.add_argument("--mace_dtype", default="float64")
    grp.add_argument("--uma_model", default="uma-s-1p2")
    grp.add_argument("--uma_task", default="omc")
    grp.add_argument("--uma_device", default="cuda")
    grp.add_argument("--hf_token", default="")
    grp.add_argument("--dftd3", action="store_true")

    p.add_argument("--overwrite", action="store_true",
                    help="Recompute even if a threshold's result.json already exists")

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        calc_type   = args.calc_type,
        mace_model  = args.mace_model,  mace_head   = args.mace_head,
        mace_device = args.mace_device, mace_dtype  = args.mace_dtype,
        uma_model   = args.uma_model,   uma_task    = args.uma_task,
        uma_device  = args.uma_device,  hf_token    = args.hf_token,
        dftd3       = args.dftd3,
    )
    calc = make_calc(cfg)

    sdim             = parse_sdim(args.supercell)
    primitive_matrix = parse_primitive_matrix(args.primitive_matrix)
    mesh             = [int(x) for x in args.mesh.split()]
    fmax_list        = sorted({float(x) for x in args.fmax.split()}, reverse=True)
    noise_list       = sorted({float(x) for x in args.noise.split()})

    atoms0 = ase_read(args.structure)
    ase_write(str(out_dir / "POSCAR-input"), atoms0, format="vasp", direct=True)

    rows = []
    for noise in noise_list:
        start = atoms0.copy()
        if noise > 0:
            start.rattle(stdev=noise, seed=args.noise_seed)

        for fmax in fmax_list:
            tag  = f"fmax_{fmax:g}_noise_{noise:g}"
            fdir = out_dir / tag
            fdir.mkdir(exist_ok=True)
            result_path = fdir / "result.json"

            if result_path.exists() and not args.overwrite:
                print(f"\n=== {tag}: [cached] {result_path} ===")
                rows.append(json.loads(result_path.read_text()))
                continue

            print(f"\n=== {tag} ===")
            t0 = time.time()

            relaxed, fmax_final, converged = relax_at_fmax(
                start, calc, fmax, args.relax_steps, args.relax_pressure,
                logfile=str(fdir / "relax_BFGS.log"),
            )
            print(f"  relax: noise={noise:g} Å  target_fmax={fmax:.5f}  "
                  f"achieved_fmax={fmax_final:.5f}  converged={converged}")
            energy = float(relaxed.get_potential_energy())
            volume = float(relaxed.get_volume())
            ase_write(str(fdir / "POSCAR-relaxed"), relaxed, format="vasp", direct=True)

            phonon, resid = build_fc2(
                relaxed, calc, sdim, primitive_matrix, args.symprec, args.amplitude,
                log_prefix="  [fc2]",
            )
            write_fc2_to_hdf5(phonon.force_constants, filename=str(fdir / "fc2.hdf5"))

            phonon.run_mesh(mesh, is_gamma_center=True)
            min_freq  = min_nonacoustic_freq(phonon)
            imaginary = min_freq < 0

            row = {
                "noise_stdev":         noise,
                "target_fmax":         fmax,
                "achieved_fmax":       fmax_final,
                "relax_converged":     converged,
                "energy_eV":           energy,
                "volume_A3":           volume,
                "residual_force":      resid,
                "min_frequency_THz":   min_freq,
                "has_imaginary_modes": imaginary,
                "elapsed_s":           time.time() - t0,
            }
            result_path.write_text(json.dumps(row, indent=2))
            rows.append(row)

            print(f"  min_freq={min_freq:.4f} THz  "
                  f"{'IMAGINARY MODES' if imaginary else 'stable (all real)'}  "
                  f"({row['elapsed_s']:.0f}s)")

    print("\n" + "=" * 80)
    print(f"  {'noise':>8s}  {'fmax':>10s}  {'achieved':>10s}  {'conv':>5s}  "
          f"{'min_freq(THz)':>14s}  {'stable':>7s}")
    for r in sorted(rows, key=lambda r: (r["noise_stdev"], -r["target_fmax"])):
        print(f"  {r['noise_stdev']:>8g}  {r['target_fmax']:>10.5f}  "
              f"{r['achieved_fmax']:>10.5f}  "
              f"{'yes' if r['relax_converged'] else 'no':>5s}  "
              f"{r['min_frequency_THz']:>14.4f}  "
              f"{'no' if r['has_imaginary_modes'] else 'yes':>7s}")
    print("=" * 80)

    summary_path = out_dir / "relax_scan_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"\n  Wrote {summary_path}")

    try:
        plot_summary(rows, out_dir)
    except Exception as e:
        print(f"  (skipping plot: {e})")


if __name__ == "__main__":
    main()
