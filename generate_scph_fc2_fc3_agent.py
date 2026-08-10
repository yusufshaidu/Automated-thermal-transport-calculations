#!/usr/bin/env python
"""generate_scph_fc2_fc3_agent.py

Self-consistent phonon (SCPH) loop (harmonic FC2 only) followed by a joint
many-body fit (FC2..FC{N+1}, from --cutoffs) to the accumulated thermal
displacements, exported to phono3py HDF5. See README.md for the full
restart/seed/skip-SCPH/select-best-iteration options.

    python generate_scph_fc2_fc3_agent.py \\
        -prim POSCAR-unitcell -sdim "2 2 2" \\
        --model mace.model --head omat_pbe \\
        -cutoffs "6.0 4.5" -temps "100 200 300" \\
        -N 100 -niter 50 -o output/
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import time
from collections import deque

import numpy as np
from ase import Atoms, units
from ase.io import read, write
from ase.geometry import get_distances
from phonopy import Phonopy
from phonopy.interface.calculator import read_crystal_structure

from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential, ForceConstants
from hiphive.structure_generation import (
    generate_rattled_structures,
    generate_phonon_rattled_structures,
)
from hiphive.utilities import extract_parameters
from trainstation import Optimizer

from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5, read_fc2_from_hdf5
from phono3py_compat import print_version_banner, recommend_bte_cli

from enum import Enum


# =============================================================================
# Logging
# =============================================================================

def setup_logging(out_dir: str, name: str) -> logging.Logger:
    """File-only logger at out_dir/pipeline_scph.log, appended to (not
    overwritten) across restarts/reruns. *name* must be unique per T (e.g.
    f"scph_T{T:.0f}") since logging.getLogger() caches by name -- reusing a
    name across different out_dir values would silently keep writing into
    the first out_dir's file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.FileHandler(os.path.join(out_dir, "pipeline_scph.log"))
    h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger.addHandler(h)
    logger.propagate = False
    return logger


# =============================================================================
# ClusterSpace construction with automatic cutoff-shell avoidance
# =============================================================================

def build_cluster_space_safe(
    supercell,
    cutoffs:      list,
    symprec:      float,
    acoustic_sum_rules: bool = True,
    cutoff_step:  float = 0.01,
    max_attempts: int   = 50,
):
    """
    Construct a hiphive ClusterSpace, automatically nudging the cutoff(s)
    if hiphive complains that the maximum cutoff lands exactly on a
    neighbor shell ("Maximum cutoff close to neighbor shell, change
    cutoff"). Whichever cutoff(s) currently equal the max are bumped by
    +cutoff_step and construction is retried, mirroring the manual
    "increase by 0.01 until it works" workaround.

    Returns (ClusterSpace, cutoffs_used).
    """
    cutoffs = list(cutoffs)
    for attempt in range(1, max_attempts + 1):
        try:
            cs = ClusterSpace(
                supercell,
                cutoffs,
                symprec            = symprec,
                acoustic_sum_rules = acoustic_sum_rules,
            )
            if attempt > 1:
                print(f"  [cutoff-retry] succeeded on attempt {attempt} "
                      f"with cutoffs={cutoffs}")
            return cs, cutoffs
        except Exception as e:
            if 'Maximum cutoff close to neighbor shell' not in str(e):
                raise
            max_val = max(cutoffs)
            cutoffs = [c + cutoff_step if c == max_val else c for c in cutoffs]
            print(f"  [cutoff-retry] max cutoff hit a neighbor shell "
                  f"(attempt {attempt}/{max_attempts}); bumping to "
                  f"cutoffs={cutoffs} and retrying")
    raise RuntimeError(
        f"build_cluster_space_safe: could not clear neighbor-shell cutoff "
        f"after {max_attempts} attempts (step={cutoff_step}); "
        f"last cutoffs tried={cutoffs}"
    )


# =============================================================================
# Displacement method
# =============================================================================

class DisplacementMethod(Enum):
    HIPHIVE = "hiphive"   # classical, hiphive phonon rattler
    PHONOPY = "phonopy"   # quantum or classical via phonopy


# =============================================================================
# Structure / phonopy helpers
# =============================================================================

def phonopysupercell(prim_file: str, dim: np.ndarray, primitive_matrix, symprec: float = 1e-3):
    """
    Build a Phonopy object and matching ASE supercell from a structure file.

    Tries phonopy's own VASP reader first; falls back to ASE (which handles
    POSCAR, CIF, XYZ, etc.) if phonopy returns None.

    symprec must match whatever tolerance is used elsewhere in the same run
    (hiphive's ClusterSpace, and the downstream thermal_transport_agent.py
    bte's --symprec) -- a mismatch means the FC2/FC3 fit's symmetry
    assumptions can disagree with what gets used to symmetrize them or
    reduce q-points over the Brillouin zone later.
    """
    from phonopy.structure.atoms import PhonopyAtoms as _PhonopyAtoms

    # ── Try phonopy's native reader ───────────────────────────────────────
    unitcell, _ = read_crystal_structure(prim_file, interface_mode="vasp")

    # ── Fall back to ASE for any other format ─────────────────────────────
    if unitcell is None:
        print(f"  phonopy VASP reader returned None for {prim_file!r} — "
              f"falling back to ASE reader.")
        ase_atoms = read(prim_file)
        unitcell  = _PhonopyAtoms(
            cell             = ase_atoms.get_cell()[:],
            scaled_positions = ase_atoms.get_scaled_positions(),
            symbols          = ase_atoms.get_chemical_symbols(),
        )

    if unitcell is None:
        raise FileNotFoundError(
            f"Could not read structure from {prim_file!r} with either "
            f"phonopy (VASP mode) or ASE. Check the file path and format."
        )

    phonon = Phonopy(
        unitcell,
        supercell_matrix = np.diag(dim),
        primitive_matrix = primitive_matrix,
        symprec          = symprec,
    )
    phonopy_sup = phonon.supercell
    supercell = Atoms(
        symbols          = phonopy_sup.symbols,
        cell             = phonopy_sup.cell,
        scaled_positions = phonopy_sup.scaled_positions,
        pbc              = True,
    )
    return unitcell, supercell, phonopy_sup, phonon


def parse_primitive_matrix(s: str):
    parts = s.split()
    return parts[0] if len(parts) == 1 else \
           np.array(parts, dtype=float).reshape(3, 3)


def parse_sdim(s: str) -> np.ndarray:
    for sep in (" ", ","):
        try:
            return np.array(s.split(sep), dtype=int)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse supercell dimension: {s!r}")


def parse_cutoffs(s: str) -> list:
    """
    Parse a space- or comma-separated list of cluster cutoffs (Å).

    The number of cutoffs determines the highest body order fitted:
    one cutoff -> 2nd order only, two -> 2nd+3rd, three -> 2nd..4th, etc.
    (N cutoffs fit orders 2..N+1, in the order given.)
    """
    for sep in (" ", ","):
        tokens = [t for t in s.split(sep) if t]
        if not tokens:
            continue
        try:
            cutoffs = [float(t) for t in tokens]
        except ValueError:
            continue
        if cutoffs:
            return cutoffs
    raise ValueError(f"Cannot parse cutoffs: {s!r}")


# =============================================================================
# Generate displaced structures
# =============================================================================

def generate_displaced_structures(
    atoms_ideal: Atoms,
    fc2:         np.ndarray,      # hiphive ASE format (3N, 3N)
    phonon:      Phonopy,
    n_structures: int,
    T:           float,
    method:      DisplacementMethod,
    qm_statistics: bool  = False,
    imag_freq_factor: float = 1.0,
    random_seed: int = None,
) -> list:
    """
    Generate n_structures thermally displaced ASE Atoms from FC2.

    DisplacementMethod.HIPHIVE -> hiphive phonon rattler (classical)
    DisplacementMethod.PHONOPY -> phonopy quantum or classical
    """
    structures = []

    if method == DisplacementMethod.HIPHIVE:
        dist_label = "classical (hiphive rattler)"
        print(f"  [hiphive]  {n_structures} structures at {T} K  ({dist_label})")
        structures = generate_phonon_rattled_structures(
            atoms_ideal,
            fc2,
            n_structures,
            T,
            QM_statistics    = False,      # hiphive rattler is always classical
            imag_freq_factor = imag_freq_factor,
        )

    elif method == DisplacementMethod.PHONOPY:
        dist_func  = "quantum" if qm_statistics else "classical"
        dist_label = f"{'quantum (Bose-Einstein)' if qm_statistics else 'classical (Maxwell-Boltzmann)'}"
        print(f"  [phonopy]  {n_structures} structures at {T} K  ({dist_label})")

        phonon.init_random_displacements(
            dist_func        = dist_func,
            cutoff_frequency = 0.01,
            max_distance     = None,
        )
        displacements = phonon.get_random_displacements_at_temperature(
            temperature         = T,
            number_of_snapshots = n_structures,
            is_plusminus        = False,
            random_seed         = random_seed,
        )   # (n_structures, N_atoms, 3)

        pos0 = atoms_ideal.get_positions()
        for k in range(displacements.shape[0]):
            atoms = atoms_ideal.copy()
            atoms.set_positions(pos0 + displacements[k])
            structures.append(atoms)

    return structures


# =============================================================================
# Evaluate forces + build hiphive-ready structures
# =============================================================================

def evaluate_forces(
    structures:  list,
    supercell:   Atoms,
    calc,
) -> list:
    """
    Evaluate forces with *calc* for each displaced structure and attach
    'displacements' and 'forces' arrays in hiphive convention.
    """
    pos0     = supercell.get_positions()
    result   = []
    t0       = time.time()

    for k, disp_atoms in enumerate(structures):
        disp_atoms.calc = calc
        forces = disp_atoms.get_forces()

        atoms_out = supercell.copy()
        disps = []
        for i, pos in enumerate(disp_atoms.positions):
            d, _ = get_distances(pos0[i], pos, cell=disp_atoms.cell, pbc=True)
            disps.append(d.flatten())

        atoms_out.set_array("displacements", np.array(disps, dtype=float))
        atoms_out.set_array("forces",        forces.astype(float))
        result.append(atoms_out)

        if (k + 1) % 20 == 0:
            print(f"    forces: {k+1}/{len(structures)}  ({time.time()-t0:.0f}s)")

    return result


# =============================================================================
# FC2 shape conversion: hiphive ASE -> phono3py
# =============================================================================

def fc2_to_phonopy(fc2_ase: np.ndarray, N: int) -> np.ndarray:
    """(3N, 3N) -> (N, N, 3, 3)"""
    return fc2_ase.reshape(N, 3, N, 3).transpose(0, 2, 1, 3)


def fc2_to_ase(fc2_phonopy: np.ndarray) -> np.ndarray:
    """(N, N, 3, 3) -> (3N, 3N)"""
    N = fc2_phonopy.shape[0]
    return fc2_phonopy.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)


# =============================================================================
# Seed the SCPH loop from an existing fc2.hdf5
# =============================================================================

def load_fc2_hdf5(fc2_path: str, N: int) -> np.ndarray:
    """
    Read an existing fc2.hdf5 (phono3py/phonopy format, full supercell FC2).

    Returns the array in phonopy convention, shape (N, N, 3, 3).
    """
    fc2_phonopy = read_fc2_from_hdf5(filename=fc2_path)
    if fc2_phonopy.shape[:2] != (N, N):
        raise ValueError(
            f"fc2 in {fc2_path!r} has shape {fc2_phonopy.shape[:2]}, "
            f"expected ({N}, {N}) to match the supercell. Make sure it was "
            f"generated for the same -prim/-sdim as this run."
        )
    return fc2_phonopy


def parameters_from_fc2(
    fc2_phonopy: np.ndarray,
    supercell:   Atoms,
    cs2:         ClusterSpace,
) -> np.ndarray:
    """
    Project an existing FC2 array onto the cs2 ClusterSpace basis via linear
    least-squares (hiphive.utilities.extract_parameters). No structures,
    forces, or MLIP evaluation needed.
    """
    fcs = ForceConstants.from_arrays(supercell, fc2_array=fc2_phonopy)
    return extract_parameters(fcs, cs2)


def parameters_from_fcp_checkpoint(
    fcp_path:  str,
    supercell: Atoms,
    cs2:       ClusterSpace,
) -> np.ndarray:
    """
    Extract 2nd-order parameters from a saved SCPH checkpoint
    (fcp_scph/scph_T{T}_iter{i}.fcp or ..._final.fcp) to restart an
    interrupted run, by projecting its FC2 onto *cs2* -- the same
    technique --init_fc2 uses for fc2.hdf5.
    """
    N = len(supercell)
    fcp = ForceConstantPotential.read(fcp_path)
    fc2 = fcp.get_force_constants(supercell).get_fc_array(order=2, format="phonopy")
    if fc2.shape != (N, N, 3, 3):
        raise ValueError(
            f"{fcp_path}: FC2 array has shape {fc2.shape}, expected "
            f"({N}, {N}, 3, 3) to match the supercell built from "
            "-prim/-sdim/-pa. Make sure these flags match the run that "
            "produced this checkpoint."
        )
    return parameters_from_fc2(fc2, supercell, cs2)


# =============================================================================
# Locate + load previously saved configs (for --skip_scph)
# =============================================================================

def find_config_files(configs_dir: str, T: float) -> list:
    """
    Find config_T{T}_N*_iter*.extxyz files written by a previous
    run_scph_and_collect call, sorted by iteration number (ascending).
    """
    pattern = os.path.join(configs_dir, f"config_T{T:.0f}_N*_iter*.extxyz")
    files = glob.glob(pattern)

    def iter_num(f):
        m = re.search(r"_iter(\d+)\.extxyz$", f)
        return int(m.group(1)) if m else -1

    files.sort(key=iter_num)
    return files


def find_latest_scph_checkpoint(fcp_dir: str, T: float) -> tuple[int, str] | None:
    """
    Find the highest-iteration scph_T{T}_iter*.fcp checkpoint in *fcp_dir*
    (used to auto-resume/extend a run). Returns (iteration, path), or None
    if no checkpoint exists. Deliberately ignores ..._final.fcp, since its
    filename doesn't encode which iteration it was written at.
    """
    pattern = os.path.join(fcp_dir, f"scph_T{T:.0f}_iter*.fcp")
    iter_re = re.compile(r"_iter(\d+)\.fcp$")

    best = None
    for f in glob.glob(pattern):
        m = iter_re.search(f)
        if m:
            it = int(m.group(1))
            if best is None or it > best[0]:
                best = (it, f)
    return best


def load_collected_configs(configs_dir: str, T: float, n_collect: int) -> list:
    """
    Load configs (with forces) from the last *n_collect* SCPH-iteration
    files found in *configs_dir* for temperature *T*. Mirrors the
    "collect last n_collect iterations" logic in run_scph_and_collect,
    but reads from disk instead of running SCPH.
    """
    files = find_config_files(configs_dir, T)
    if not files:
        raise FileNotFoundError(
            f"--skip_scph: no config files found in {configs_dir!r} "
            f"matching 'config_T{T:.0f}_N*_iter*.extxyz'. Point "
            f"--configs_dir at the 'configs/' folder from a previous "
            f"run, or pass explicit files via --configs."
        )

    collect_files = files[-n_collect:]
    collected = []
    for f in collect_files:
        collected.extend(read(f"{f}@:"))
    print(f"  [skip_scph] Loaded {len(collected)} configs from "
          f"{len(collect_files)}/{len(files)} existing iteration files "
          f"in {configs_dir}")
    return collected


def load_explicit_configs(configs_arg: str) -> list:
    """
    Load configs from an explicit, comma-separated list of file paths
    and/or glob patterns given via --configs.
    """
    collected = []
    n_files = 0
    for token in configs_arg.split(","):
        token = token.strip()
        if not token:
            continue
        matched = sorted(glob.glob(token))
        if not matched and os.path.exists(token):
            matched = [token]
        if not matched:
            print(f"  WARNING: --configs entry {token!r} matched no files")
            continue
        for f in matched:
            collected.extend(read(f"{f}@:"))
            n_files += 1
    if not collected:
        raise FileNotFoundError(
            f"--configs={configs_arg!r} matched no readable config files."
        )
    print(f"  [skip_scph] Loaded {len(collected)} configs from "
          f"{n_files} file(s) given via --configs")
    return collected


# =============================================================================
# SCPH loop + accumulate configs
# =============================================================================

def _min_nonacoustic_freq(phonon) -> float:
    """Min phonon frequency over the current mesh, excluding the 3 trivially
    ~0 acoustic bands at Gamma (translational invariance) -- so this is a
    real instability check, not just ~0 regardless of what happens
    elsewhere in the BZ. Requires phonon.run_mesh() to have been called."""
    freqs   = phonon.mesh.frequencies.copy()
    qpoints = phonon.mesh.qpoints
    gamma_rows = np.where(np.all(np.abs(qpoints) < 1e-8, axis=1))[0]
    for row in gamma_rows:
        acoustic = np.argsort(freqs[row])[:3]
        freqs[row, acoustic] = np.inf
    return float(np.min(freqs))


# =============================================================================
# Anderson-accelerated parameter mixing
# =============================================================================

class AndersonMixer:
    """
    Anderson-accelerated mixing for the SCPH fixed-point update
    x_{n+1} = f(x_n), where f(x_n) is a fresh (noisy) least-squares fit at
    each iteration.

    Plain linear mixing (x_{n+1} = beta*f(x_n) + (1-beta)*x_n) only ever
    uses the previous iterate. Anderson mixing keeps a rolling window of
    the last `depth` (input, output) pairs and, at each step, solves a
    small least-squares problem for the linear combination of past
    residuals (r_k = g_k - x_k) that best cancels the current residual,
    then damps the result by `beta` -- typically converges in far fewer
    iterations than plain linear mixing for this kind of fixed-point loop.

    Falls back to plain linear mixing once history is empty (first call,
    or after a safeguard drop empties it). The safeguard compares the
    Anderson-extrapolated residual norm against the plain residual norm
    and shrinks the history window if extrapolation made things worse --
    this can happen when stochastic fit-data sampling noise makes recent
    residuals nearly collinear, ill-conditioning the small least-squares
    solve.
    """

    def __init__(self, depth: int = 5):
        self.depth  = depth
        self._x_hist = deque(maxlen=depth)
        self._g_hist = deque(maxlen=depth)

    def mix(self, x_n: np.ndarray, g_n: np.ndarray, beta: float) -> np.ndarray:
        r_n = g_n - x_n
        x_new, r_bar = self._propose(x_n, g_n, r_n, beta)
        while self._x_hist and np.linalg.norm(r_bar) > np.linalg.norm(r_n):
            self._x_hist.popleft()
            self._g_hist.popleft()
            x_new, r_bar = self._propose(x_n, g_n, r_n, beta)
        self._x_hist.append(x_n.copy())
        self._g_hist.append(g_n.copy())
        return x_new

    def _propose(self, x_n, g_n, r_n, beta):
        if not self._x_hist:
            return beta * g_n + (1 - beta) * x_n, r_n

        r_hist = [g_k - x_k for x_k, g_k in zip(self._x_hist, self._g_hist)]
        dR = np.column_stack([r_n - r_k for r_k in r_hist])
        gamma, *_ = np.linalg.lstsq(dR, r_n, rcond=None)

        dX = np.column_stack([x_n - x_k for x_k in self._x_hist])
        dG = np.column_stack([g_n - g_k for g_k in self._g_hist])
        x_bar = x_n - dX @ gamma
        g_bar = g_n - dG @ gamma
        return beta * g_bar + (1 - beta) * x_bar, g_bar - x_bar


def run_scph_and_collect(
    supercell:       Atoms,
    cs2:             ClusterSpace,    # 2nd order only (SCPH)
    T:               float,
    alpha:           float,
    n_iterations:    int,
    n_structures:    int,
    prim_file:       str,
    sdim:            np.ndarray,
    primitive_matrix,
    calc,
    n_collect:       int,             # collect configs from last n_collect iters
    symprec:         float,
    qm_statistics:   bool,
    imag_freq_factor: float,
    ckpt_interval:   int,
    out_dir:         str,
    nstart:          int              = 0,
    parameters_start: np.ndarray | None = None,
    random_seed:     int              = 42,
    init_fc2:        np.ndarray | None = None,
    fc2_only:        bool             = False,
    select_best_iteration: bool       = False,
    fe_mesh:         list[int] | None = None,
    stability_tol:   float            = -0.01,
    mixing:          str              = "linear",
    mixing_depth:    int              = 5,
    log:             logging.Logger | None = None,
) -> tuple[np.ndarray, list, str]:
    """
    Run the SCPH loop on *cs2* (2nd order ClusterSpace) and accumulate
    displaced configs + forces from the last *n_collect* iterations.

    init_fc2 : optional (N, N, 3, 3) phonopy-format FC2 (e.g. loaded from a
        previously computed fc2.hdf5). When given, it seeds the initial 2nd
        order parameters via an algebraic ClusterSpace projection (no
        forces/MLIP needed), replacing the small-amplitude rattled-structure
        initialisation. Ignored if parameters_start is already given (e.g.
        checkpoint restart).
    fc2_only : if True (requires init_fc2), skip the iterative SCPH
        refinement entirely: generate a single batch of thermal
        displacements directly from init_fc2 at temperature T, evaluate
        forces with the MLIP, and return that batch as collected_configs
        for higher-order fitting.
    select_best_iteration : if True, track the harmonic free energy F(T)
        at every iteration (phonopy mesh sum on fe_mesh, no new MLIP
        evaluations) and anchor the n_collect window of collected configs
        on the iteration with the lowest F(T) instead of the last
        iteration. F(T) is not guaranteed to decrease monotonically
        (stochastic sampling noise each iteration), so this avoids
        collecting from a worse-than-best snapshot just because it's last.
        The lowest F(T) is picked only among iterations whose FC2 has no
        imaginary (non-acoustic-Gamma) mode on fe_mesh -- otherwise the
        selection can lock onto an unstable model that merely has a low
        harmonic free energy. If no iteration is stable, raises
        RuntimeError rather than silently returning an unstable model.
    stability_tol : a mesh frequency (THz) is treated as imaginary only if
        it falls below this (negative) tolerance, to avoid rejecting an
        otherwise-stable iteration over near-zero numerical/symmetrization
        noise in a soft optical mode.
    mixing : "linear" (default) mixes only the previous iteration's
        parameters (parameters_new = alpha*fit + (1-alpha)*parameters_old).
        "anderson" additionally uses the last `mixing_depth` iterations'
        (input, output) pairs to extrapolate a better fixed point before
        applying the same alpha damping -- see AndersonMixer.
    mixing_depth : history window for "anderson" mixing. Ignored for
        "linear".

    Returns
    -------
    (parameters_converged, collected_configs, fcp_path)
        parameters_converged : final SCPH parameter vector
        collected_configs    : list of hiphive-ready ASE Atoms with
                               'displacements' and 'forces' arrays,
                               drawn from the last n_collect iterations
                               (or the single fc2_only batch)
    """
    from hiphive.force_constant_model import ForceConstantModel

    if log:
        log.info(f"SCPH start: T={T:.0f}K  n_iterations={n_iterations}  "
                 f"n_structures={n_structures}  alpha={alpha}  "
                 f"mixing={mixing}"
                 + (f"  mixing_depth={mixing_depth}" if mixing == "anderson" else "")
                 + f"  qm_statistics={qm_statistics}  fc2_only={fc2_only}  "
                 f"select_best_iteration={select_best_iteration}"
                 + (f"  fe_mesh={fe_mesh}  stability_tol={stability_tol}"
                    if select_best_iteration else ""))

    sc  = StructureContainer(cs2)
    fcm = ForceConstantModel(supercell, cs2)
    mixer = AndersonMixer(depth=mixing_depth) if mixing == "anderson" else None

    # Choose displacement method
    disp_method = DisplacementMethod.PHONOPY if qm_statistics \
                  else DisplacementMethod.HIPHIVE

    if select_best_iteration and fe_mesh is None:
        fe_mesh = [10, 10, 10]

    # ── Initial model ─────────────────────────────────────────────────────
    if parameters_start is None:
        if init_fc2 is not None:
            print("  Initialising 2nd-order parameters from --init_fc2 "
                  "(algebraic ClusterSpace projection, no forces needed) …")
            parameters_start = parameters_from_fc2(init_fc2, supercell, cs2)
        else:
            print("  Initialising with rattled structures (amplitude=0.001 Å) …")
            init_structs = generate_rattled_structures(supercell, n_structures, 0.001)
            init_tagged  = evaluate_forces(init_structs, supercell, calc)
            for s in init_tagged:
                sc.add_structure(s)
            opt = Optimizer(sc.get_fit_data(), train_size=1.0, check_condition=False)
            opt.train()
            parameters_start = opt.parameters
            sc.delete_all_structures()
            print(f"  Initial model: rmse = {opt.rmse_train:.5f}")

    parameters_old = parameters_start.copy()

    # ── Directories ──────────────────────────────────────────────────────
    fcp_dir  = os.path.join(out_dir, "fcp_scph")
    cfg_dir  = os.path.join(out_dir, "configs")
    os.makedirs(fcp_dir, exist_ok=True)
    os.makedirs(cfg_dir, exist_ok=True)

    # ── --fc2_only: single displacement batch, no SCPH refinement ──────────
    if fc2_only:
        print(f"\n  --fc2_only: generating a single batch of {n_structures} "
              f"thermal displacements from --init_fc2 at T={T:.0f} K "
              f"(SCPH iteration skipped) …")
        fcm.parameters = parameters_old
        fc2_ase = fcm.get_force_constants().get_fc_array(order=2, format="ase")
        _, _, _, phonon = phonopysupercell(prim_file, sdim, primitive_matrix, symprec)
        phonon.force_constants = fc2_to_phonopy(fc2_ase, len(supercell))
        phonon.symmetrize_force_constants()

        displaced = generate_displaced_structures(
            atoms_ideal      = supercell,
            fc2              = fc2_ase,
            phonon           = phonon,
            n_structures     = n_structures,
            T                = T,
            method           = disp_method,
            qm_statistics    = qm_statistics,
            imag_freq_factor = imag_freq_factor,
            random_seed      = random_seed,
        )
        tagged = evaluate_forces(displaced, supercell, calc)

        cfg_file = os.path.join(
            cfg_dir, f"config_T{T:.0f}_N{n_structures}_iter0.extxyz"
        )
        write(cfg_file, tagged, format="extxyz")

        fcp_path = os.path.join(fcp_dir, f"scph_T{T:.0f}_final.fcp")
        ForceConstantPotential(cs2, parameters_old).write(fcp_path)
        print(f"  fc2-only FCP (from --init_fc2, unrefined) -> {fcp_path}")
        if log:
            log.info(f"Model chosen: fc2_only, unrefined --init_fc2 -> {fcp_path} "
                     f"(F(T)/min_freq not evaluated)")

        return parameters_old, tagged, fcp_path

    # ── SCPH iterations ───────────────────────────────────────────────────
    # Track which iteration config files are saved so we can collect the
    # last n_collect of them for FC3 fitting. On restart (nstart > 0), seed
    # this with the config files already on disk from before the
    # interruption, so --n_collect spans the full run, not just the
    # iterations run after this restart.
    saved_config_files = []
    if nstart > 0:
        iter_re = re.compile(r"_iter(\d+)\.extxyz$")
        for f in find_config_files(cfg_dir, T):
            m = iter_re.search(f)
            if m and int(m.group(1)) < nstart:
                saved_config_files.append(f)
        if saved_config_files:
            print(f"  Restart: found {len(saved_config_files)} pre-existing "
                  f"config file(s) from iterations < {nstart} in {cfg_dir}")

    # [(iteration, F(T) kJ/mol), ...], only tracked if select_best_iteration.
    # On restart, recompute F(T) for whichever pre-existing checkpoints are
    # on disk (only every ckpt_interval iterations were saved), so the
    # best-iteration search still spans the full run, not just the
    # iterations run after this restart.
    #
    # Checkpoint scph_T{T}_iter{k}.fcp is written *after* iteration k
    # updates parameters_old, i.e. it holds the model entering iteration
    # k+1 -- the same model whose free energy the live loop above labels
    # "iteration k+1". So recovered points must be relabeled it = k + 1.
    # Iteration 0's own free energy (from the initial, pre-loop model) has
    # no corresponding checkpoint and cannot be recovered after a restart.
    free_energies = []
    if select_best_iteration and nstart > 0:
        iter_re_fcp = re.compile(r"_iter(\d+)\.fcp$")
        for f in glob.glob(os.path.join(fcp_dir, f"scph_T{T:.0f}_iter*.fcp")):
            m = iter_re_fcp.search(f)
            if not m:
                continue
            it = int(m.group(1)) + 1
            if it >= nstart:
                continue
            _, _, _, phonon_prev = phonopysupercell(prim_file, sdim, primitive_matrix, symprec)
            fc2_prev = ForceConstantPotential.read(f).get_force_constants(
                supercell
            ).get_fc_array(order=2, format="phonopy")
            phonon_prev.force_constants = fc2_prev
            phonon_prev.symmetrize_force_constants()
            phonon_prev.run_mesh(fe_mesh, is_gamma_center=True)
            min_freq_prev = _min_nonacoustic_freq(phonon_prev)
            phonon_prev.run_thermal_properties(temperatures=[T], classical=not qm_statistics)
            free_energies.append(
                (it, float(phonon_prev.thermal_properties.free_energy[0]), min_freq_prev)
            )
        if free_energies:
            print(f"  Restart: recomputed F(T) for {len(free_energies)} "
                  f"pre-existing checkpoint(s) from iterations < {nstart} "
                  f"(iteration 0's F(T) cannot be recovered after restart)")

    for i in range(nstart, n_iterations):
        t_iter = time.time()
        print(f"\n  ── SCPH iteration {i} ──────────────────────")

        # FC2 from current parameters -> set on phonon
        fcm.parameters = parameters_old
        fc2_ase = fcm.get_force_constants().get_fc_array(order=2, format="ase")
        _, _, _, phonon = phonopysupercell(prim_file, sdim, primitive_matrix, symprec)
        N = len(supercell)
        phonon.force_constants = fc2_to_phonopy(fc2_ase, N)
        phonon.symmetrize_force_constants()

        # This FC2 is exactly what will generate this iteration's config
        # batch below, so its free energy represents that batch's model.
        fe_i, min_freq_i = None, None
        if select_best_iteration:
            phonon.run_mesh(fe_mesh, is_gamma_center=True)
            min_freq_i = _min_nonacoustic_freq(phonon)
            phonon.run_thermal_properties(temperatures=[T], classical=not qm_statistics)
            fe_i = float(phonon.thermal_properties.free_energy[0])
            free_energies.append((i, fe_i, min_freq_i))
            print(f"    F({T:.0f}K) = {fe_i:.6f} kJ/mol   min_freq = {min_freq_i:.4f} THz")

        # Generate displaced structures
        displaced = generate_displaced_structures(
            atoms_ideal      = supercell,
            fc2              = fc2_ase,
            phonon           = phonon,
            n_structures     = n_structures,
            T                = T,
            method           = disp_method,
            qm_statistics    = qm_statistics,
            imag_freq_factor = imag_freq_factor,
            random_seed      = random_seed + i,
        )

        # Evaluate forces — this is the expensive step
        tagged = evaluate_forces(displaced, supercell, calc)

        # Save configs to disk (iteration-stamped for later accumulation)
        cfg_file = os.path.join(cfg_dir, f"config_T{T:.0f}_N{n_structures}_iter{i}.extxyz")
        write(cfg_file, tagged, format="extxyz")
        saved_config_files.append(cfg_file)

        # Fit 2nd order model
        sc.delete_all_structures()
        for s in tagged:
            sc.add_structure(s)
        opt = Optimizer(sc.get_fit_data(), train_size=1.0, check_condition=False)
        opt.train()

        # Update parameters: plain linear/damped mixing, or Anderson-
        # accelerated mixing over the last `mixing_depth` iterations.
        if mixer is not None:
            parameters_new = mixer.mix(parameters_old, opt.parameters, beta=alpha)
        else:
            parameters_new = alpha * opt.parameters + (1 - alpha) * parameters_old

        disps         = np.concatenate([s.get_array("displacements") for s in tagged])
        delta_x_norm  = np.linalg.norm(parameters_old - parameters_new) / \
                        np.linalg.norm(parameters_old)
        print(f"    rmse={opt.rmse_train:.5f}  |delta_x|/|x|={delta_x_norm:.6f}  "
              f"disp_max={np.max(np.abs(disps)):.4f} Å  "
              f"({time.time()-t_iter:.0f}s)")

        if log:
            log.info(
                f"iter {i}: rmse={opt.rmse_train:.5f}  "
                f"|delta_x|/|x|={delta_x_norm:.6f}"
                + (f"  F({T:.0f}K)={fe_i:.6f} kJ/mol  min_freq={min_freq_i:.4f} THz"
                   if select_best_iteration else "")
            )

        parameters_old = parameters_new

        # Checkpoint
        if i % ckpt_interval == 0:
            fcp_ckpt = ForceConstantPotential(cs2, parameters_old)
            ckpt_path = os.path.join(fcp_dir, f"scph_T{T:.0f}_iter{i}.fcp")
            fcp_ckpt.write(ckpt_path)
            if log:
                log.info(f"checkpoint saved: iter {i} -> {ckpt_path}")

    # Save final FCP
    fcp_final = ForceConstantPotential(cs2, parameters_old)
    fcp_path  = os.path.join(fcp_dir, f"scph_T{T:.0f}_final.fcp")
    fcp_final.write(fcp_path)
    print(f"\n  Final SCPH FCP -> {fcp_path}")
    if log:
        log.info(f"final SCPH FCP -> {fcp_path}")

    # Collect configs for the higher-order fit: either the last n_collect
    # iterations (default), or -- if select_best_iteration -- the n_collect
    # iterations ending at whichever iteration had the lowest F(T) among
    # iterations with no imaginary (non-acoustic-Gamma) mode on fe_mesh, since
    # F(T) is not guaranteed to decrease monotonically iteration to
    # iteration (stochastic sampling noise), and a low F(T) from an unstable
    # model is not a meaningful comparison to a stable one.
    if select_best_iteration and free_energies:
        stable = [x for x in free_energies if x[2] > stability_tol]
        if not stable:
            worst = min(free_energies, key=lambda x: x[2])
            msg = (f"--select_best_iteration: no SCPH iteration for T={T:.0f}K "
                   f"had an all-real spectrum on fe_mesh={fe_mesh} "
                   f"(stability_tol={stability_tol} THz); least-imaginary was "
                   f"iteration {worst[0]} with min_freq={worst[2]:.4f} THz. "
                   f"Refusing to select an unstable model -- rerun with more "
                   f"iterations, a larger --fe_mesh, looser --stability_tol, "
                   f"or inspect the run with plot_scph_free_energy.py.")
            if log:
                log.error(msg)
            raise RuntimeError(msg)

        best_iter, best_fe, best_min_freq = min(stable, key=lambda x: x[1])
        print(f"\n  --select_best_iteration: lowest F({T:.0f}K) among stable "
              f"iterations at iteration {best_iter}  (F={best_fe:.6f} kJ/mol, "
              f"min_freq={best_min_freq:.4f} THz)")
        if log:
            log.info(f"Model chosen: iteration {best_iter}  "
                     f"F({T:.0f}K)={best_fe:.6f} kJ/mol  "
                     f"min_freq={best_min_freq:.4f} THz  (stable; lowest F(T) "
                     f"among {len(stable)}/{len(free_energies)} stable iterations)")
        by_iter = {}
        iter_re = re.compile(r"_iter(\d+)\.extxyz$")
        for f in find_config_files(cfg_dir, T):
            m = iter_re.search(f)
            if m:
                by_iter[int(m.group(1))] = f
        window_start = max(0, best_iter - n_collect + 1)
        collect_files = [by_iter[j] for j in range(window_start, best_iter + 1)
                          if j in by_iter]
        print(f"  Collecting configs from iterations {window_start}..{best_iter} "
              f"(anchored on best iteration)")
    else:
        collect_files = saved_config_files[-n_collect:]
        if log:
            log.info(f"Model chosen: final iteration {n_iterations - 1} "
                     f"(--select_best_iteration not used; F(T)/min_freq not "
                     f"evaluated per-iteration)")

    collected = []
    for f in collect_files:
        collected.extend(read(f"{f}@:"))
    print(f"  Collected {len(collected)} configs from "
          f"{len(collect_files)} iterations for FC3 fitting")
    if log:
        log.info(f"Collected {len(collected)} configs from "
                 f"{len(collect_files)} iterations for FC3 fitting")

    return parameters_old, collected, fcp_path


# =============================================================================
# Fit force constants jointly, order 2..len(cutoffs)+1
# =============================================================================

def fit_force_constants(
    supercell:   Atoms,
    configs:     list,
    cutoffs:     list,
    symprec:     float,
    train_size:  float = 1.0,
) -> ForceConstantPotential:
    """
    Fit a joint ForceConstantPotential to *configs*, spanning body orders
    2..max_order where max_order = len(cutoffs) + 1 (e.g. cutoffs=[c2, c3]
    fits 2nd+3rd order, cutoffs=[c2, c3, c4] fits 2nd..4th order).
    """
    max_order = len(cutoffs) + 1
    print(f"\n  Fitting FC2..FC{max_order}: cutoffs={cutoffs} Å  "
          f"n_configs={len(configs)}")
    cs, cutoffs = build_cluster_space_safe(
        supercell,
        cutoffs,
        symprec            = symprec,
        acoustic_sum_rules = True,
    )
    print(cs)
    cs.print_orbits()

    # By hiphive convention (see evaluate_forces) every saved config's cell
    # and positions are supposed to be bit-identical to *supercell* -- only
    # 'displacements'/'forces' vary per structure. Configs reloaded from
    # extxyz go through ASE's %16.8f-precision writer, which is far below
    # symprec but, for cells sitting near a symmetry degeneracy, can still
    # be enough to make spglib's primitive-cell standardization (called
    # independently per structure in hiphive's align_supercell) pick a
    # different-but-equivalent orientation than the one baked into
    # cs.primitive_structure -- causing a spurious "Found no translation!"
    # for every structure. Snap cell/positions back onto the authoritative
    # in-memory supercell to remove that precision mismatch entirely.
    for s in configs:
        if len(s) == len(supercell):
            # Forces read back from extxyz live on a SinglePointCalculator
            # (forces is a canonical ASE property), not in s.arrays.
            # set_positions() below would invalidate that calculator (its
            # cached check_state no longer matches), so pull forces out
            # into a plain array first -- that survives the reposition
            # and is what hiphive's add_structure looks for directly.
            if "forces" not in s.arrays:
                s.set_array("forces", s.get_forces())
            s.cell = supercell.cell
            s.set_positions(supercell.positions)

    sc = StructureContainer(cs)
    n_ok = 0
    for s in configs:
        try:
            sc.add_structure(s)
            n_ok += 1
        except Exception as e:
            print(f"    Skipping structure: {e}")
    print(f"  Loaded {n_ok}/{len(configs)} structures into StructureContainer")
    print(f"  {sc}")

    opt = Optimizer(
        sc.get_fit_data(),
        fit_method      = "least-squares",
        train_size      = train_size,
        check_condition = False,
    )
    opt.train()
    print(opt)

    return ForceConstantPotential(cs, opt.parameters)


# =============================================================================
# Export force constants -> phono3py HDF5 (+ generic HDF5 for order > 3)
# =============================================================================

def export_to_phono3py(
    fcp:       ForceConstantPotential,
    supercell: Atoms,
    phonon:    Phonopy,
    out_dir:   str,
    max_order: int,
) -> dict:
    """
    Export every fitted order (2..max_order) to disk.

    FC2: hiphive ASE (3N,3N) -> reshape+transpose -> (N,N,3,3) [phono3py]
    FC3: hiphive default (N,N,N,3,3,3) = phono3py full FC3, no transposition
    FC4+ : phono3py has no native reader for these, so they're written to
           generic fc{n}.hdf5 files (dataset "fc{n}", hiphive's raw shape)
           for use with custom four(+)-phonon tooling.

    Returns a dict {order: fc_array} for every order actually exported.
    """
    fcs = fcp.get_force_constants(supercell)
    N   = len(supercell)
    exported = {}

    fc2_ase = fcs.get_fc_array(order=2, format="ase")
    fc2     = fc2_to_phonopy(fc2_ase, N)
    fc2_path = os.path.join(out_dir, "fc2.hdf5")
    write_fc2_to_hdf5(fc2, filename=fc2_path)
    print(f"  FC2 {fc2.shape} -> {fc2_path}")
    exported[2] = fc2

    if max_order >= 3:
        fc3      = fcs.get_fc_array(order=3)          # (N, N, N, 3, 3, 3)
        fc3_path = os.path.join(out_dir, "fc3.hdf5")
        write_fc3_to_hdf5(fc3, filename=fc3_path)
        print(f"  FC3 {fc3.shape} -> {fc3_path}")
        exported[3] = fc3

    for order in range(4, max_order + 1):
        import h5py
        fc_n     = fcs.get_fc_array(order=order)
        fc_path  = os.path.join(out_dir, f"fc{order}.hdf5")
        with h5py.File(fc_path, "w") as f:
            f.create_dataset(f"fc{order}", data=fc_n, compression="gzip")
        print(f"  FC{order} {fc_n.shape} -> {fc_path}  "
              f"(generic HDF5 — phono3py has no native reader for order>3)")
        exported[order] = fc_n

    return exported


# =============================================================================
# Calculator factory
# =============================================================================

def make_calc(args):
    from ase.calculators.mixing import SumCalculator
    ct = args.calc_type.lower()

    if ct == "mace":
        from mace.calculators import MACECalculator
        print(f"  [MACE]  model={args.model}  head={args.head}  "
              f"device={args.device}  dtype=float64")
        base = MACECalculator(
            model_paths   = args.model,
            device        = args.device,
            head          = args.head,
            default_dtype = "float64",
        )
    elif ct == "uma":
        if not args.hf_token:
            raise ValueError("--hf_token required for --calc_type uma")
        os.environ["HF_TOKEN"] = args.hf_token
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        print(f"  [UMA]  task={args.head}  device={args.device}")
        predictor = pretrained_mlip.get_predict_unit(
            "uma-s-1p2", device=args.device
        )
        base = FAIRChemCalculator(predictor, task_name=args.head)
    else:
        raise NotImplementedError(f"calc_type '{args.calc_type}' not supported")

    if args.include_d3:
        from dftd3.ase import DFTD3
        d3 = DFTD3(
            method  = "pbe", damping = "d3bj",
            realspace_cutoff = {"disp2": 60.0 * units.Bohr,
                                "disp3": 40.0 * units.Bohr},
            params_tweaks    = {"s6": 1.0, "s8": 0.7875, "s9": 0.0,
                                "a1": 0.4289, "a2": 4.4407, "alp": 14},
        )
        return SumCalculator([base, d3])
    return base


# =============================================================================
# Main
# =============================================================================

def main(args):
    primitive_matrix = parse_primitive_matrix(args.primitive_matrix)
    sdim             = parse_sdim(args.sdim)
    cutoffs          = parse_cutoffs(args.cutoffs)
    fe_mesh          = [int(x) for x in args.fe_mesh.split()]
    max_order        = len(cutoffs) + 1
    temperatures     = np.array(args.temperatures.split(), dtype=float)
    out_dir          = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"  SCPH -> FC2..FC{max_order} -> phono3py")
    print("=" * 60)
    print_version_banner()
    print(f"  Temperatures     : {temperatures} K")
    print(f"  Supercell        : {args.sdim}")
    if args.skip_scph:
        print(f"  Mode             : --skip_scph (reusing existing configs)")
        print(f"  Configs source   : {args.configs or args.configs_dir or '<outdir>/T<T>/configs (auto)'}")
    else:
        print(f"  n_structures/iter: {args.n_structures}")
        print(f"  n_iterations     : {args.n_iterations}")
    print(f"  n_collect        : {args.n_collect} last iterations for fitting")
    cutoff_str = "  ".join(
        f"order{o}={c}" for o, c in zip(range(2, max_order + 1), cutoffs)
    )
    print(f"  Cutoffs          : {cutoff_str} Å  (max order = {max_order})")
    print(f"  Statistics       : {'quantum (Bose-Einstein)' if args.qm_statistics else 'classical (Maxwell-Boltzmann)'}")

    # ── Build supercell + 2nd order ClusterSpace ─────────────────────────
    _, supercell, _, phonon = phonopysupercell(
        args.prim_file, sdim, primitive_matrix, args.symprec
    )
    N = len(supercell)
    print(f"\n  Space group: {phonon.symmetry.get_international_table()}")
    print(f"  Supercell: {N} atoms")

    # SCPH always uses only the first (2nd order) cutoff — it only needs a
    # harmonic model to generate thermal displacements.
    cs2, cs2_cutoffs = build_cluster_space_safe(
        supercell,
        [cutoffs[0]],
        symprec            = args.symprec,
        acoustic_sum_rules = True,
    )
    cutoffs[0] = cs2_cutoffs[0]
    print(f"\n  2nd order ClusterSpace (SCPH):\n  {cs2}")

    # ── Optional: seed the 2nd order model from an existing fc2.hdf5 ──────
    init_fc2 = None
    if args.init_fc2:
        if args.skip_scph:
            print(f"  WARNING: --init_fc2 is ignored with --skip_scph "
                  f"(existing configs are reused as-is).")
        else:
            init_fc2 = load_fc2_hdf5(args.init_fc2, N)
            print(f"  Init FC2         : {args.init_fc2}")
            if args.fc2_only:
                print(f"  Mode             : --fc2_only (single displacement "
                      f"batch from --init_fc2, no SCPH iteration)")

    if args.select_best_iteration:
        if args.skip_scph or args.fc2_only:
            print(f"  WARNING: --select_best_iteration is ignored with "
                  f"--skip_scph/--fc2_only (no SCPH loop runs).")
        else:
            print(f"  Select best      : lowest F(T) among stable iterations "
                  f"anchors --n_collect (fe_mesh={fe_mesh}, "
                  f"stability_tol={args.stability_tol})")

    # ── Calculator (not needed if reusing existing forces) ────────────────
    calc = None if args.skip_scph else make_calc(args)

    # ── Per-temperature SCPH + FC3 fitting ───────────────────────────────
    for T in temperatures:
        print(f"\n{'='*60}")
        print(f"  T = {T} K")
        print(f"{'='*60}")

        T_dir = os.path.join(out_dir, f"T{T:.0f}")
        os.makedirs(T_dir, exist_ok=True)

        # Unique logger name per T -- logging.getLogger() caches by name, so
        # reusing one name across temperatures would keep writing into
        # whichever T_dir/pipeline.log was opened first.
        log = setup_logging(T_dir, name=f"scph_T{T:.0f}")
        log.info(f"=== T={T:.0f}K  cutoffs={cutoffs}  sdim={args.sdim}  "
                 f"calc={args.calc_type}  mode="
                 + ("skip_scph" if args.skip_scph
                    else "fc2_only" if args.fc2_only else "scph"))

        if args.skip_scph:
            # ── Reuse configs from a previous SCPH run ────────────────────
            if args.configs:
                collected_configs = load_explicit_configs(args.configs)
            else:
                configs_dir = args.configs_dir or os.path.join(T_dir, "configs")
                collected_configs = load_collected_configs(
                    configs_dir, T, args.n_collect
                )
            fcp_path = None
        else:
            # Handle restart from checkpoint
            parameters_start = None
            nstart           = 0
            if args.nstart > 0 and args.initial_parameter_file:
                # Explicit restart: takes precedence over --resume.
                parameters_start = parameters_from_fcp_checkpoint(
                    args.initial_parameter_file, supercell, cs2
                )
                nstart = args.nstart
                print(f"  Restarting from {args.initial_parameter_file}  "
                      f"(nstart={nstart})")
            elif args.resume and not args.fc2_only:
                # Auto-resume: reuse the latest checkpoint for this T, if
                # any, so --n_iterations can simply be raised to extend a
                # previous run instead of starting over.
                latest = find_latest_scph_checkpoint(
                    os.path.join(T_dir, "fcp_scph"), T
                )
                if latest is not None:
                    latest_iter, latest_path = latest
                    parameters_start = parameters_from_fcp_checkpoint(
                        latest_path, supercell, cs2
                    )
                    nstart = latest_iter + 1
                    print(f"  --resume: found checkpoint at iteration "
                          f"{latest_iter} -> {latest_path}  (nstart={nstart})")
                    if nstart >= args.n_iterations:
                        print(f"  WARNING: --n_iterations={args.n_iterations} "
                              f"<= next iteration ({nstart}); no new SCPH "
                              f"iterations will run. Raise --n_iterations to "
                              f"extend this run further.")
                else:
                    print(f"  --resume: no existing checkpoint found for "
                          f"T={T:.0f} K, starting from scratch")

            # ── SCPH loop — accumulates configs internally ────────────────
            parameters_converged, collected_configs, fcp_path = run_scph_and_collect(
                supercell        = supercell,
                cs2              = cs2,
                T                = T,
                alpha            = args.alpha,
                n_iterations     = args.n_iterations,
                n_structures     = args.n_structures,
                prim_file        = args.prim_file,
                sdim             = sdim,
                primitive_matrix = primitive_matrix,
                calc             = calc,
                n_collect        = args.n_collect,
                symprec          = args.symprec,
                qm_statistics    = args.qm_statistics,
                imag_freq_factor = args.imag_freq_factor,
                ckpt_interval    = args.ckpt,
                out_dir          = T_dir,
                nstart           = nstart,
                parameters_start = parameters_start,
                init_fc2         = init_fc2,
                fc2_only         = args.fc2_only,
                select_best_iteration = args.select_best_iteration,
                fe_mesh          = fe_mesh,
                stability_tol    = args.stability_tol,
                mixing           = args.mixing,
                mixing_depth     = args.mixing_depth,
                log              = log,
            )

        # ── Fit FC2..FC{max_order} jointly to accumulated configs ─────────
        log.info(f"Fitting joint FC2..FC{max_order} to "
                 f"{len(collected_configs)} collected configs")
        fcp_all = fit_force_constants(
            supercell  = supercell,
            configs    = collected_configs,
            cutoffs    = cutoffs,
            symprec    = args.symprec,
            train_size = args.train_size,
        )
        fcp_path_out = os.path.join(T_dir, f"fcp_order2to{max_order}.fcp")
        fcp_all.write(fcp_path_out)
        print(f"  Joint FCP written -> {fcp_path_out}")
        log.info(f"Joint FCP written -> {fcp_path_out}")

        # ── Export force constants to phono3py HDF5 (+ generic for order>3)
        print("\n  Exporting force constants …")
        _, _, _, phonon_T = phonopysupercell(args.prim_file, sdim, primitive_matrix, args.symprec)
        fc_all = export_to_phono3py(
            fcp_all, supercell, phonon_T, T_dir, max_order=max_order
        )
        log.info(f"Exported fc2..fc{max_order} to {T_dir}  -- done")

        dim_str    = " ".join(map(str, sdim))
        fc_summary = "\n".join(
            f"    fc{order}.hdf5   {fc_all[order].shape}"
            for order in sorted(fc_all)
        )
        print(f"""
  ─────────────────────────────────────────────────────
  T = {T:.0f} K  ->  {T_dir}/
{fc_summary}

  Thermal conductivity — Python pipeline (recommended, version-agnostic):
    python thermal_transport_agent.py bte \\
        --out_dir {T_dir}/ \\
        --mesh "11 11 11" --temperatures "{T:.0f}" \\
        --solver rta --transport_type SMM19 --parallel_mode serial_gp --resume
""")
        print(recommend_bte_cli(
            mesh      = "11 11 11",
            prim_file = args.prim_file,
            dim       = dim_str,
            pa        = args.primitive_matrix,
            tmin      = T, tmax = T, tstep = 1,
            full_fc   = True,   # export_to_phono3py always writes full arrays
        ))
        if max_order >= 4:
            print(f"  NOTE: fc4.hdf5{'..fc' + str(max_order) + '.hdf5' if max_order > 4 else ''} "
                  f"are generic HDF5 files — phono3py's BTE CLI only "
                  f"consumes fc2.hdf5/fc3.hdf5 above.")
        print("  ─────────────────────────────────────────────────────")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="SCPH loop -> joint multi-order force constant fitting "
                     "-> phono3py HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Structure
    p.add_argument("-prim",       "--prim_file",        required=True)
    p.add_argument("-sdim",       "--sdim",              default="2 2 2")
    p.add_argument("-pa",         "--primitive_matrix",  default="auto")
    p.add_argument("-tolerance",  "--symprec",  type=float, default=1e-3)
    p.add_argument("-o",          "--outdir",            required=True)

    # Calculator
    p.add_argument("--calc_type", default="mace", choices=["mace", "uma"])
    p.add_argument("--model",     default="",
                   help="MACE model path (ignored with --skip_scph)")
    p.add_argument("--head",      default="omat_pbe",
                   help="MACE head or UMA task name (ignored with --skip_scph)")
    p.add_argument("--device",    default="cuda", choices=["cpu", "cuda"],
                   help="Device for the calculator (ignored with --skip_scph)")
    p.add_argument("--hf_token",  default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--include_d3", action="store_true")

    # SCPH
    p.add_argument("-temps",  "--temperatures",  required=True,
                   help="Space-separated temperatures in K")
    p.add_argument("-N",      "--n_structures",  type=int, default=100,
                   help="Displaced structures per SCPH iteration "
                        "(ignored with --skip_scph)")
    p.add_argument("-niter",  "--n_iterations",  type=int, default=50,
                   help="Ignored with --skip_scph")
    p.add_argument("-alpha",  "--alpha",         type=float, default=0.2,
                   help="SCPH momentum (learning rate); ignored with --skip_scph")
    p.add_argument("--mixing", default="linear", choices=["linear", "anderson"],
                   help="Parameter-update scheme between SCPH iterations. "
                        "'linear' (default) mixes only the previous "
                        "iteration's fit, weighted by --alpha. 'anderson' "
                        "additionally extrapolates from the last "
                        "--mixing_depth iterations' inputs/outputs before "
                        "applying the same --alpha damping -- typically "
                        "converges in fewer iterations. Ignored with "
                        "--skip_scph.")
    p.add_argument("--mixing_depth", type=int, default=5,
                   help="History window (# past iterations) for "
                        "--mixing anderson. Ignored for --mixing linear.")
    p.add_argument("-cutoffs", "--cutoffs",      required=True,
                   help="Space-separated cluster cutoffs (Å), one per body "
                        "order starting at 2nd order. N cutoffs fit orders "
                        "2..N+1: '5.0' -> 2nd order only; '5.0 3.0' -> "
                        "2nd+3rd; '5.0 3.0 3.0' -> 2nd..4th order. The SCPH "
                        "loop itself always uses only the first (2nd order) "
                        "cutoff.")
    p.add_argument("--qm_statistics", action="store_true", default=False,
                   help="Quantum Bose-Einstein displacement statistics "
                        "(default: classical Maxwell-Boltzmann). "
                        "Recommended at low T. Ignored with --skip_scph.")
    p.add_argument("--imag_freq_factor", type=float, default=1.0)
    p.add_argument("-ckpt",   "--ckpt",          type=int, default=2,
                   help="Save FCP checkpoint every N iterations "
                        "(ignored with --skip_scph)")
    p.add_argument("-nstart", "--nstart",         type=int, default=0,
                   help="Restart: iteration to resume at (loop runs "
                        "range(nstart, n_iterations)). Set to 1 + the "
                        "iteration number in the --initial_parameter_file "
                        "checkpoint's filename, so that iteration is not "
                        "redone. Requires --initial_parameter_file.")
    p.add_argument("-init",   "--initial_parameter_file", default=None,
                   help="Restart: path to a saved SCPH checkpoint "
                        "(fcp_scph/scph_T{T}_iter{i}.fcp) to resume from. "
                        "Its FC2 is projected onto the ClusterSpace to seed "
                        "the parameters, the same way --init_fc2 does. "
                        "Requires -nstart > 0.")
    p.add_argument("--resume", action="store_true",
                   help="Auto-resume: for each temperature, reuse the "
                        "highest-iteration fcp_scph/scph_T{T}_iter*.fcp "
                        "checkpoint under --outdir, if any, and continue "
                        "from there -- raise --n_iterations beyond the "
                        "previous run's to extend it instead of starting "
                        "over. Ignored if -nstart/--initial_parameter_file "
                        "are given explicitly (those take precedence), and "
                        "with --skip_scph/--fc2_only.")
    p.add_argument("--init_fc2", default=None,
                   help="Path to an existing fc2.hdf5 (phono3py format, "
                        "full supercell) used to seed the 2nd-order model "
                        "instead of small-amplitude rattled structures. "
                        "The FC2 is projected onto the ClusterSpace "
                        "algebraically (no forces/MLIP evaluation for this "
                        "step). By default the SCPH loop still iterates "
                        "from this starting point; add --fc2_only to skip "
                        "iteration entirely. Ignored with --skip_scph.")
    p.add_argument("--fc2_only", action="store_true",
                   help="Requires --init_fc2. Skip the iterative SCPH "
                        "refinement: generate one batch of thermal "
                        "displacements directly from --init_fc2 at each "
                        "temperature, evaluate forces with the MLIP, and "
                        "fit FC2..FC{max_order} to that batch. Useful when "
                        "you already trust an existing fc2.hdf5 (e.g. from "
                        "a converged SCPH run) and only want higher-order "
                        "terms without re-running the full SCPH loop.")

    # Higher-order fitting
    p.add_argument("--n_collect", type=int, default=10,
                   help="Collect configs from the last N SCPH iterations "
                        "for the higher-order fit. More iterations = more "
                        "data = better-conditioned fit. Also used to select "
                        "the last N iteration files with --skip_scph.")
    p.add_argument("--train_size", type=float, default=1.0,
                   help="Fraction of collected data used for training")
    p.add_argument("--select_best_iteration", action="store_true",
                   help="Track the harmonic free energy F(T) at every SCPH "
                        "iteration (cheap phonopy mesh sum on --fe_mesh, no "
                        "extra MLIP evaluations) and anchor the --n_collect "
                        "window on the iteration with the LOWEST F(T) among "
                        "iterations that have no imaginary (non-acoustic-"
                        "Gamma) mode on --fe_mesh, instead of the last "
                        "iteration. F(T) is not guaranteed to decrease "
                        "monotonically between iterations (stochastic "
                        "sampling noise), so this avoids training the "
                        "higher-order fit on a worse-than-best snapshot "
                        "just because it ran last -- and the stability "
                        "filter avoids locking onto an unstable model that "
                        "merely has a low harmonic free energy. Raises if "
                        "no iteration is stable. Ignored with "
                        "--skip_scph/--fc2_only.")
    p.add_argument("--stability_tol", type=float, default=-0.01,
                   help="With --select_best_iteration: a mesh frequency "
                        "(THz) below this (negative) tolerance marks an "
                        "iteration as imaginary/unstable. Slightly negative "
                        "by default to tolerate numerical/symmetrization "
                        "noise in near-zero soft optical modes.")
    p.add_argument("--fe_mesh", default="10 10 10",
                   help="q-point mesh for the per-iteration free energy "
                        "calc used by --select_best_iteration (lighter than "
                        "a production kappa mesh -- only ranks iterations, "
                        "not a final result)")

    # Skip SCPH — reuse existing configs (e.g. to re-fit with new cutoffs)
    p.add_argument("--skip_scph", action="store_true",
                   help="Skip the SCPH loop entirely and go straight to "
                        "force constant fitting using previously saved "
                        "configs with forces (no MLIP/calc needed). Useful "
                        "for re-fitting with a different --cutoffs (e.g. "
                        "adding a 4th order term) without re-running "
                        "expensive force evaluations. Combine with "
                        "--configs_dir or --configs to point at the "
                        "existing data.")
    p.add_argument("--configs_dir", default=None,
                   help="Directory containing config_T{T}_N*_iter*.extxyz "
                        "files from a previous run. Used with --skip_scph. "
                        "Default: <outdir>/T<T>/configs (matches the "
                        "layout run_scph_and_collect writes).")
    p.add_argument("--configs", default=None,
                   help="Comma-separated explicit file paths and/or glob "
                        "patterns of config files (with forces) to use "
                        "directly for force constant fitting. Used with "
                        "--skip_scph; overrides --configs_dir "
                        "auto-discovery. Applied identically at every "
                        "temperature in --temperatures, so normally used "
                        "with a single temperature.")

    args = p.parse_args()

    # Validate
    if args.n_collect > args.n_iterations and not args.skip_scph:
        print(f"  WARNING: --n_collect ({args.n_collect}) > --n_iterations "
              f"({args.n_iterations}), clamping to n_iterations.")
        args.n_collect = args.n_iterations

    if args.skip_scph and args.calc_type == "uma" and not args.hf_token:
        # make_calc() is never called with --skip_scph, so this is fine;
        # nothing to validate here.
        pass

    if args.fc2_only and not args.init_fc2:
        p.error("--fc2_only requires --init_fc2")
    if args.fc2_only and args.skip_scph:
        p.error("--fc2_only and --skip_scph are mutually exclusive")

    main(args)
