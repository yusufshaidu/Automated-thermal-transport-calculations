#!/usr/bin/env python
"""Load a saved FC2 (or FORCE_SETS) and report band structure, DOS, thermal properties,
and mean-square displacements. Flags imaginary modes at the exact high-symmetry
q-points of --band_path, not just a mesh minimum -- a mesh whose size shares no
common factor with a path's fractional q-points (e.g. 1/3 on a hex path) can miss
them entirely. See README.md.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_SETS
from phonopy.interface.vasp import read_vasp
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

from phono3py.file_IO import read_fc2_from_hdf5

from generate_scph_fc2_fc3_agent import parse_primitive_matrix, parse_sdim


# =============================================================================
# High-symmetry band paths (fractional reciprocal-lattice coordinates)
# =============================================================================

BAND_PATHS = {
    "hex": {
        "points": {
            "G": (0., 0., 0.), "A": (0., 0., 0.5),
            "H": (1 / 3, 1 / 3, 0.5), "K": (1 / 3, 1 / 3, 0.),
            "L": (0.5, 0., 0.5), "M": (0.5, 0., 0.),
        },
        "path": ["G", "M", "K", "G", "A", "L", "H", "A", "L", "M", "K", "H"],
    },
    "ortho": {
        "points": {
            "G": (0., 0., 0.), "F": (0.5, -0.5, 0.), "L": (0.5, 0., 0.),
            "P": (0.33674351, -0.66325649, 0.33674351),
            "P1": (0.66325649, -0.33674351, -0.33674351),
            "Q": (0.17348702, 0.17348702, 0.17348702),
            "Q1": (0.82651298, -0.17348702, -0.17348702),
            "Z": (0.5, -0.5, 0.5),
        },
        "path": ["G", "P", "Z", "Q", "G", "F", "P1", "Q1", "L", "Z"],
    },
}


def band_qpoints(path_name: str, npoints: int):
    """Return (qpoints, connections, labels) for a single connected chain through BAND_PATHS[path_name]."""
    spec = BAND_PATHS[path_name]
    labels = spec["path"]
    chain = [spec["points"][label] for label in labels]
    qpoints, connections = get_band_qpoints_and_path_connections([chain], npoints=npoints)
    return qpoints, connections, labels


# =============================================================================
# Load
# =============================================================================

def load_phonon(indir, unitcell_file, sdim, primitive_matrix, symprec, source):
    unitcell = read_vasp(os.path.join(indir, unitcell_file))
    phonon = Phonopy(
        unitcell, supercell_matrix=np.diag(sdim),
        primitive_matrix=primitive_matrix, symprec=symprec,
    )
    if source == "fc2":
        fc2 = read_fc2_from_hdf5(filename=os.path.join(indir, "fc2.hdf5"))
        phonon.force_constants = fc2
    else:
        phonon.dataset = parse_FORCE_SETS(filename=os.path.join(indir, "FORCE_SETS"))
        phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    return phonon


# =============================================================================
# Band structure + exact-q stability check
# =============================================================================

def run_bands(phonon, path_name: str, npoints: int, out_dir: str, stability_tol: float):
    qpoints, connections, labels = band_qpoints(path_name, npoints)
    phonon.run_band_structure(
        qpoints, path_connections=connections, labels=labels,
        with_eigenvectors=False, is_band_connection=False,
    )
    phonon.write_yaml_band_structure(filename=os.path.join(out_dir, "bands.yaml"))
    phonon.plot_band_structure().savefig(os.path.join(out_dir, "bands.png"))

    bs = phonon.band_structure
    imaginary = []
    for seg_q, seg_freq in zip(bs.qpoints, bs.frequencies):
        for q, freqs in zip(seg_q, seg_freq):
            bad = freqs[freqs < stability_tol]
            if bad.size:
                imaginary.append((tuple(q), float(bad.min())))

    if imaginary:
        worst_q, worst_f = min(imaginary, key=lambda x: x[1])
        print(f"  UNSTABLE: {len(imaginary)} q-point(s) on the '{path_name}' path "
              f"below {stability_tol} THz; worst is {worst_f:.4f} THz at q={worst_q}")
    else:
        print(f"  Stable: no frequency below {stability_tol} THz on the "
              f"'{path_name}' path ({len(labels)} points, {npoints} pts/segment)")
    return imaginary


# =============================================================================
# DOS / thermal properties / MSD
# =============================================================================

def run_dos(phonon, mesh, mesh_symmetry: bool, out_dir: str):
    phonon.run_mesh(mesh, is_mesh_symmetry=mesh_symmetry)
    phonon.run_total_dos()
    dos = phonon.total_dos
    out = np.column_stack([dos.frequency_points, dos.dos])
    np.savetxt(os.path.join(out_dir, "total_dos.dat"), out, header="frequency(THz) dos")
    phonon.plot_total_dos().savefig(os.path.join(out_dir, "total_dos.png"))
    print(f"  DOS -> {out_dir}/total_dos.{{dat,png}}")


def run_thermal(phonon, mesh, mesh_symmetry: bool, t_min: float, t_max: float,
                 t_step: float, debye_tmax: float, out_dir: str):
    phonon.run_mesh(mesh, is_mesh_symmetry=mesh_symmetry)
    phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step, cutoff_frequency=0.0)
    tp = phonon.thermal_properties
    out = np.column_stack([tp.temperatures, tp.free_energy, tp.entropy, tp.heat_capacity])
    np.savetxt(
        os.path.join(out_dir, "thermal_properties.dat"), out,
        header="T(K) free_energy(kJ/mol) entropy(J/K/mol) heat_capacity(J/K/mol)",
    )
    phonon.write_yaml_thermal_properties(filename=os.path.join(out_dir, "thermal_properties.yaml"))
    phonon.plot_thermal_properties().savefig(os.path.join(out_dir, "thermal_properties.png"))
    print(f"  Thermal properties -> {out_dir}/thermal_properties.{{dat,yaml,png}}")

    if debye_tmax > 0:
        _debye_fit(tp.temperatures, tp.heat_capacity, debye_tmax, out_dir)


def _debye_fit(temperatures, heat_capacity, debye_tmax: float, out_dir: str):
    from scipy.optimize import curve_fit

    R = 8.314  # J/K/mol
    T   = np.asarray(temperatures)
    cv_r = np.asarray(heat_capacity) / R
    mask = T < debye_tmax
    if mask.sum() < 2:
        print(f"  Debye T^3 fit skipped: fewer than 2 points below {debye_tmax} K")
        return

    def cv_model(x, a):
        return a * x ** 3

    popt, _ = curve_fit(cv_model, T[mask], cv_r[mask])
    theta_d = (12.0 * np.pi ** 4 / 5 / popt[0]) ** (1 / 3.0)
    print(f"  Debye T^3 fit (T<{debye_tmax:.0f} K): a={popt[0]:.6e}  Theta_D={theta_d:.2f} K")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(T, cv_r * R, "--", label="C_v")
    ax.plot(T, cv_model(T, popt[0]) * R, "k-.", label="Debye T^3 fit")
    ax.set_xlabel("T (K)")
    ax.set_ylabel("C_v (J/K/mol)")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "debye_fit.png"))
    plt.close(fig)


def run_msd(phonon, mesh, t_min: float, t_max: float, t_step: float,
            freq_min: float, out_dir: str):
    phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
    phonon.run_thermal_displacements(t_min=t_min, t_max=t_max, t_step=t_step, freq_min=freq_min)
    td = phonon.thermal_displacements
    temperatures = td.temperatures
    u2 = np.reshape(td.thermal_displacements, [len(temperatures), -1, 3])
    msd_per_atom = np.sum(u2, axis=-1)
    msd_total    = np.mean(msd_per_atom, axis=1)

    species     = sorted(set(phonon.unitcell.symbols))
    species_arr = np.array(phonon.unitcell.symbols)

    columns = [temperatures, msd_total]
    header  = ["T(K)", "MSD_total(A^2)"]
    for sp in species:
        columns.append(np.mean(msd_per_atom[:, species_arr == sp], axis=1))
        header.append(f"MSD_{sp}(A^2)")

    out = np.column_stack(columns)
    out_path = os.path.join(out_dir, "msd_vs_T.dat")
    np.savetxt(out_path, out, header="  ".join(header))
    print(f"  MSD -> {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Report phonon band structure (with an exact-q stability "
                     "check), DOS, thermal properties, and MSD from a saved FC2 "
                     "or FORCE_SETS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--indir", required=True,
                   help="Directory containing fc2.hdf5 (or FORCE_SETS with --source force_sets)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--unitcell", default="POSCAR-unitcell",
                   help="Unit cell filename within --indir")
    p.add_argument("--source", choices=["fc2", "force_sets"], default="fc2")

    p.add_argument("--supercell", default="2 2 2")
    p.add_argument("--primitive_matrix", default="auto")
    p.add_argument("--symprec", type=float, default=1e-3)

    p.add_argument("--band_path", choices=sorted(BAND_PATHS), default="hex")
    p.add_argument("--npoints", type=int, default=51)
    p.add_argument("--stability_tol", type=float, default=-0.01,
                   help="Frequency (THz) below which a band-path q-point is flagged imaginary")

    p.add_argument("--mesh", default="20 20 20", help="Mesh for --dos/--thermal")
    p.add_argument("--mesh_symmetry", dest="mesh_symmetry", action="store_true", default=True)
    p.add_argument("--no_mesh_symmetry", dest="mesh_symmetry", action="store_false")

    p.add_argument("--dos", action="store_true")

    p.add_argument("--thermal", action="store_true")
    p.add_argument("--debye_tmax", type=float, default=500.0,
                   help="Upper T (K) for the Debye T^3 heat-capacity fit; 0 to disable")

    p.add_argument("--msd", action="store_true")
    p.add_argument("--msd_mesh", default="15 15 15")
    p.add_argument("--freq_min", type=float, default=0.01,
                   help="Frequency cutoff (THz) below which modes are excluded from MSD")

    p.add_argument("--t_min", type=float, default=0.0)
    p.add_argument("--t_max", type=float, default=1000.0)
    p.add_argument("--t_step", type=float, default=10.0)

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sdim             = parse_sdim(args.supercell)
    primitive_matrix = parse_primitive_matrix(args.primitive_matrix)
    mesh             = [int(x) for x in args.mesh.split()]

    if args.band_path == "hex" and any(m % 3 for m in mesh):
        print(f"  WARNING: --mesh {mesh} is not divisible by 3 -- it cannot land "
              f"on the 'hex' path's K/H points (1/3, 1/3, *), so --dos/--thermal's "
              f"mesh sum can miss instabilities there. Trust --band_path for the "
              f"stability verdict, not the mesh.")

    phonon = load_phonon(
        args.indir, args.unitcell, sdim, primitive_matrix, args.symprec, args.source
    )
    print(f"  Space group: {phonon.symmetry.get_international_table()}")

    imaginary = run_bands(
        phonon, args.band_path, args.npoints, args.out_dir, args.stability_tol
    )

    if args.dos:
        run_dos(phonon, mesh, args.mesh_symmetry, args.out_dir)
    if args.thermal:
        run_thermal(
            phonon, mesh, args.mesh_symmetry, args.t_min, args.t_max, args.t_step,
            args.debye_tmax, args.out_dir,
        )
    if args.msd:
        msd_mesh = [int(x) for x in args.msd_mesh.split()]
        run_msd(phonon, msd_mesh, args.t_min, args.t_max, args.t_step, args.freq_min, args.out_dir)

    if imaginary:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
