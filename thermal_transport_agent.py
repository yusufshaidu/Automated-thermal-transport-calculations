#!/usr/bin/env python
"""
thermal_transport_mace.py
=========================
Automated lattice thermal conductivity pipeline using MACE / UMA + phono3py.

Workflow
--------
1. Read POSCAR / CIF
2. Variable-cell relaxation  (ASE + calculator + FrechetCellFilter)
3. Generate displaced supercells  (phono3py Python API)
4. Evaluate forces  (MACE or UMA, optional DFT-D3 dispersion)
5. Produce FC2 + FC3  (phono3py Python API)
6. Solve phonon BTE:
      solver         : "rta"  -> single-mode RTA  (fast)
                       "lbte" -> iterative BTE    (accurate)
      transport_type : None    -> kappa_P only (standard particle-like RTA/LBTE)
                       "SMM19" -> kappa_P + kappa_C, Simoncelli-Marzari-Mauri (2019)
                                 Wigner transport equation
                       "NJC23" -> kappa_P + kappa_C, alternative inter-band formulation
                       "IBDB19"-> kappa_P + kappa_C, Isaeva-Barbalinardo-Donadio-Baroni
                                 (2019) formulation
   Parallelism modes:
      serial      : all q at once, one process, no disk writes mid-run
      serial_gp   : one q at a time, kappa-m*-g*.hdf5 written after each q,
                    per-q resume via checkpoint + disk scan on restart
      omp         : same as serial; OMP_NUM_THREADS controls threads
      grid_points : split irreducible q across Python workers / SLURM arrays
7. Collect per-q kappa-m*-g*.hdf5 -> final kappa-m*.hdf5
8. Write results + plots

Resume behaviour (serial_gp)
-----------------------------
On every restart _sync_gp_progress() reconciles:
  - kappa-m{mesh}-g{N}.hdf5 files present on disk
  - done_gps list stored in checkpoint.json
The union is trusted; checkpoint wins only when the file also exists.
Progress is written to checkpoint.json after EVERY q-point so a killed
job loses at most one q-point of work.

Monitor progress without logging in:
  python3 -c "
  import json; from pathlib import Path
  d = json.loads(Path('results/checkpoint.json').read_text())
  p = d.get('gp_progress', {})
  print(f\"{len(d.get('done_gps',[]))} q-pts done | {p.get('pct','?')}% | last={p.get('last_gp','?')}\")
  "
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from ase import Atoms, units
from ase.calculators.mixing import SumCalculator
from ase.filters import FrechetCellFilter
from ase.io import read as ase_read, write as ase_write
from ase.optimize import FIRE, BFGS

from phono3py import Phono3py
from phono3py.file_IO import (
    write_fc2_to_hdf5,
    write_fc3_to_hdf5,
    write_FORCES_FC3,
)
from phono3py.interface.phono3py_yaml import Phono3pyYaml
from phonopy.structure.atoms import PhonopyAtoms

from phono3py_compat import (
    print_version_banner,
    is_v4_or_later,
    get_thermal_conductivity_RTA_compat,
    get_ir_grid_points_compat,
    recommend_bte_cli,
)


# =============================================================================
# Logging
# =============================================================================

def setup_logging(out_dir: Path, name: str = "thermal") -> logging.Logger:
    fmt    = "%(asctime)s  %(levelname)-8s  %(message)s"
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    for h in [logging.FileHandler(out_dir / "pipeline.log"),
              logging.StreamHandler()]:
        h.setFormatter(logging.Formatter(fmt))
        logger.addHandler(h)
    return logger


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # I/O
    structure:         str   = "POSCAR"
    out_dir:           str   = "results"

    # Calculator — "mace" | "uma"
    calc_type:         str   = "mace"

    # MACE
    mace_model:        str   = ""
    mace_head:         str   = ""
    mace_device:       str   = "cuda"
    mace_dtype:        str   = "float64"

    # UMA
    uma_model:         str   = "uma-s-1p2"
    uma_task:          str   = "omc"
    uma_device:        str   = "cuda"
    hf_token:          str   = ""

    # Dispersion (both calc types)
    dftd3:             bool  = False

    # Relaxation
    no_relax:          bool  = False
    relax_fmax:        float = 0.001
    relax_steps:       int   = 500
    relax_pressure:    float = 0.0

    # Phono3py supercell
    supercell:         str   = "2 2 2"
    primitive_matrix:  str   = "auto"
    symprec:           float = 1e-5
    amplitude:         float = 0.03
    cutoff_pair:       float = 5.0

    # BTE
    solver:            str   = "rta"
    # transport_type : None | "SMM19" | "NJC23" | "IBDB19"
    #   None    -> standard particle-like transport (kappa_P only)
    #   SMM19   -> Simoncelli-Marzari-Mauri (2019) Wigner transport equation
    #   NJC23   -> alternative inter-band transport formulation
    #   IBDB19  -> Isaeva-Barbalinardo-Donadio-Baroni (2019) formulation
    transport_type:    str | None = None
    mesh:              str   = "11 11 11"
    temperatures:      str   = "300"

    # Isotope scattering
    isotope:           bool  = False
    # mass_variances: space-separated per-atom-species g-factors, in the
    # element order of the primitive cell. Empty -> phono3py's built-in
    # natural-abundance values.
    mass_variances:    str   = ""

    # Parallelism
    parallel_mode:     str   = "serial"
    n_workers:         int   = 1
    gp:                str   = ""
    gp_batch_size:     int   = 1

    # Checkpointing
    resume:            bool  = True

    @property
    def supercell_matrix(self) -> np.ndarray:
        v = np.array(self.supercell.split(), dtype=int)
        if v.size == 3:
            return np.diag(v)
        if v.size == 9:
            return v.reshape(3, 3)
        raise ValueError("supercell: 3 diagonal or 9 full integers")

    @property
    def mesh_list(self) -> list[int]:
        return [int(x) for x in self.mesh.split()]

    @property
    def temperature_list(self) -> list[float]:
        return [float(x) for x in self.temperatures.split()]

    @property
    def primitive_matrix_parsed(self):
        p = self.primitive_matrix.split()
        return p[0] if len(p) == 1 else np.array(p, dtype=float).reshape(3, 3)

    @property
    def mass_variances_parsed(self) -> list[float] | None:
        s = self.mass_variances.strip()
        return None if not s else [float(x) for x in s.split()]

    @property
    def gp_list(self) -> list[int] | None:
        s = self.gp.strip()
        return None if not s else [int(x) for x in s.split()]

    @property
    def mesh_tag(self) -> str:
        return "".join(map(str, self.mesh_list))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Config":
        return cls(**json.loads(path.read_text()))


# =============================================================================
# ph-ph interaction initialiser  (version-aware symmetrize_fc3q)
# =============================================================================

def _ph3_lang() -> dict:
    """
    Return the `lang` keyword for the Phono3py constructor.

    phono3py v4 defaults to the Rust backend (phonors), which rebuilds
    internal three-phonon interaction data on every run_thermal_conductivity
    call and is significantly slower per q-point than the v3 C backend when
    called in a loop from the Python API.

    Using lang="C" restores the v3 C-extension behaviour, which preserves
    interaction data across repeated calls and matches CLI-like per-q
    performance.  For v3 installs, lang is not passed (the parameter did
    not exist).
    """
    if is_v4_or_later():
        return {"lang": "C"}
    return {}


def _init_phph(ph3: Phono3py, log: logging.Logger | None = None) -> None:
    """
    Call ph3.init_phph_interaction() with the correct symmetrize_fc3q value
    for the installed phono3py version.

    phono3py v3.x: symmetrize_fc3q=True is cheap — apply it.
    phono3py v4.x: symmetrize_fc3q=True triggers a full reciprocal-space
        symmetrization via the Rust backend, which is very expensive for
        large supercells (MOF-5: several minutes per call).  In v4 this
        is redundant because:
          (a) We already call produce_fc3(symmetrize_fc3r=True) which
              applies real-space symmetry before writing the FC3.
          (b) The v4 Rust backend applies its own internal symmetrization.
        So we use symmetrize_fc3q=False for v4, matching the behaviour of
        the v3 code path that ran without the extra q-space step.
    """
    use_sym = not is_v4_or_later()
    msg = (
        f"  init_phph_interaction(symmetrize_fc3q={use_sym}) "
        f"[phono3py {('v4+: False — Rust backend handles symmetry' if is_v4_or_later() else 'v3: True')}] …"
    )
    if log is not None:
        log.info(msg)
    else:
        logging.info(msg)

    t0 = time.time()
    ph3.init_phph_interaction(symmetrize_fc3q=use_sym)

    elapsed = time.time() - t0
    done_msg = f"  ph-ph interaction ready  ({elapsed:.1f}s)"
    if log is not None:
        log.info(done_msg)
    else:
        logging.info(done_msg)


# =============================================================================
# BTE settings resolver
# =============================================================================

def _resolve_bte(cfg: Config) -> tuple[bool, str | None]:
    """
    Map (solver, transport_type) to (is_LBTE, transport_type) for the
    current run_thermal_conductivity API.

    solver  transport_type        ->  is_LBTE   transport_type passed through
    ------  --------------------     -------   ------------------------------
    rta     None                  ->  False     None       (kappa_P only)
    rta     SMM19/NJC23/IBDB19    ->  False     same       (kappa_P + kappa_C, RTA)
    lbte    None                  ->  True      None       (kappa_P, full BTE)
    lbte    SMM19/NJC23/IBDB19    ->  True      same       (kappa_P + kappa_C, full BTE)
    """
    is_lbte = cfg.solver.lower() == "lbte"
    return is_lbte, cfg.transport_type


def _solver_label(cfg: Config) -> str:
    s = "LBTE" if cfg.solver.lower() == "lbte" else "RTA"
    if cfg.transport_type:
        s += f" + {cfg.transport_type} (kappa_P + kappa_C)"
    else:
        s += " (kappa_P only)"
    if cfg.isotope:
        s += " + isotope"
    return s


# =============================================================================
# Checkpoint
# =============================================================================

class Checkpoint:
    def __init__(self, path: Path):
        self.path  = Path(path).resolve()   # absolute — immune to os.chdir
        self._data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def done(self, stage: str) -> bool:
        return bool(self._data.get(stage, {}).get("done", False))

    def mark(self, stage: str, meta: dict = None) -> None:
        self._data[stage] = {"done": True, "t": time.time(), **(meta or {})}
        self.path.write_text(json.dumps(self._data, indent=2))

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, val) -> None:
        self._data[key] = val
        self.path.write_text(json.dumps(self._data, indent=2))


# =============================================================================
# ASE <-> PhonopyAtoms
# =============================================================================

def ase_to_phonopy(atoms: Atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols          = atoms.get_chemical_symbols(),
        cell             = atoms.get_cell()[:],
        scaled_positions = atoms.get_scaled_positions(),
    )


def phonopy_to_ase(ph: PhonopyAtoms) -> Atoms:
    return Atoms(
        numbers          = ph.numbers,
        cell             = ph.cell,
        scaled_positions = ph.scaled_positions,
        pbc              = True,
    )


# =============================================================================
# Calculator factory
# =============================================================================

def _grimme_d3():
    try:
        from dftd3.ase import DFTD3
    except ImportError:
        raise ImportError(
            "dftd3 not installed. Install with: pip install torch-dftd"
        )
    print("  [D3]  Adding Grimme D3-BJ dispersion (PBE)")
    return DFTD3(
        method  = "pbe",
        damping = "d3bj",
        realspace_cutoff = {
            "disp2": 60.0 * units.Bohr,
            "disp3": 40.0 * units.Bohr,
        },
        params_tweaks = {
            "s6": 1.0, "s8": 0.7875, "s9": 0.0,
            "a1": 0.4289, "a2": 4.4407, "alp": 14,
        },
    )


def _make_mace(cfg: Config):
    try:
        from mace.calculators import MACECalculator
    except ImportError:
        raise ImportError("mace-torch not installed: pip install mace-torch")
    if not cfg.mace_model:
        raise ValueError("--mace_model required when --calc_type mace")
    kw = dict(
        model_paths   = cfg.mace_model,
        device        = cfg.mace_device,
        default_dtype = cfg.mace_dtype,
    )
    if cfg.mace_head:
        kw["head"] = cfg.mace_head
    print(f"  [MACE]  model={cfg.mace_model}  device={cfg.mace_device}"
          + (f"  head={cfg.mace_head}" if cfg.mace_head else ""))
    calc = MACECalculator(**kw)
    return SumCalculator([calc, _grimme_d3()]) if cfg.dftd3 else calc


def _make_uma(cfg: Config):
    if not cfg.hf_token:
        raise ValueError(
            "--hf_token required when --calc_type uma. "
            "Pass --hf_token or export HF_TOKEN."
        )
    os.environ["HF_TOKEN"] = cfg.hf_token
    try:
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
    except ImportError:
        raise ImportError("fairchem-core not installed: pip install fairchem-core")
    print(f"  [UMA]  model={cfg.uma_model}  task={cfg.uma_task}"
          f"  device={cfg.uma_device}")
    predictor = pretrained_mlip.get_predict_unit(
        cfg.uma_model, device=cfg.uma_device
    )
    calc = FAIRChemCalculator(predictor, task_name=cfg.uma_task)
    return SumCalculator([calc, _grimme_d3()]) if cfg.dftd3 else calc


def make_calc(cfg: Config):
    """Dispatch calculator based on cfg.calc_type."""
    ct = cfg.calc_type.lower()
    if ct == "mace":
        return _make_mace(cfg)
    elif ct == "uma":
        return _make_uma(cfg)
    else:
        raise NotImplementedError(
            f"Unknown calc_type '{cfg.calc_type}'. Choose: mace | uma"
        )


# =============================================================================
# Force constant IO helpers
# =============================================================================

def _read_fc2(fc2_path: Path, ph3: Phono3py) -> np.ndarray:
    """
    Read FC2 via phono3py's own reader.
    p2s_map is intentionally omitted — phono3py detects compact vs full
    automatically from the HDF5 content, consistent with how we write
    (no p2s_map in write_fc2_to_hdf5).
    """
    from phono3py.file_IO import read_fc2_from_hdf5
    return read_fc2_from_hdf5(filename=str(fc2_path))


def _read_fc3(fc3_path: Path) -> np.ndarray:
    """Read FC3 via phono3py's own reader (version-safe key handling)."""
    from phono3py.file_IO import read_fc3_from_hdf5
    return read_fc3_from_hdf5(filename=str(fc3_path))


# =============================================================================
# Stage 1: Read structure
# =============================================================================

def stage_read(cfg: Config, log: logging.Logger, out_dir: Path) -> Atoms:
    log.info("─" * 60)
    log.info("STAGE 1  Read structure")
    log.info("─" * 60)

    p = Path(cfg.structure)
    if not p.exists():
        raise FileNotFoundError(p)

    atoms = ase_read(str(p))
    canonical = out_dir / "POSCAR-unitcell"
    ase_write(str(canonical), atoms, format="vasp", direct=True)
    atoms = ase_read(str(canonical), format="vasp")

    log.info(f"  File    : {p}")
    log.info(f"  Atoms   : {len(atoms)}")
    log.info(f"  Species : {sorted(set(atoms.get_chemical_symbols()))}")
    a, b, c = [np.linalg.norm(atoms.cell[i]) for i in range(3)]
    log.info(f"  Cell    : a={a:.3f}  b={b:.3f}  c={c:.3f} Å")
    return atoms


# =============================================================================
# Stage 2: Relaxation
# =============================================================================

def stage_relax(
    atoms:   Atoms,
    cfg:     Config,
    ckpt:    Checkpoint,
    log:     logging.Logger,
    out_dir: Path,
) -> Atoms:
    log.info("─" * 60)
    log.info("STAGE 2  Variable-cell relaxation")
    log.info("─" * 60)

    relax_path = out_dir / "POSCAR-relaxed"

    if cfg.no_relax:
        log.info("  --no_relax — skipping.")
        ase_write(str(relax_path), atoms, format="vasp", direct=True)
        return atoms

    if ckpt.done("relax") and relax_path.exists() and cfg.resume:
        log.info("  [RESUME] Loading relaxed structure.")
        return ase_read(str(relax_path), format="vasp")

    t0    = time.time()
    atoms = atoms.copy()
    atoms.calc = make_calc(cfg)

    e0, v0 = atoms.get_potential_energy(), atoms.get_volume()
    log.info(f"  E0 = {e0:.6f} eV   V0 = {v0:.3f} Å^3")

    p_eV = cfg.relax_pressure * 0.00624
    filt = FrechetCellFilter(atoms, scalar_pressure=p_eV)

    #log.info(f"  FIRE  fmax = {cfg.relax_fmax*10:.4f} eV/Å …")
    #FIRE(filt, logfile=str(out_dir / "relax_FIRE.log")).run(
    #    fmax=cfg.relax_fmax * 10, steps=cfg.relax_steps // 2
    #)
    log.info(f"  BFGS  fmax = {cfg.relax_fmax:.5f} eV/Å …")
    BFGS(filt, logfile=str(out_dir / "relax_BFGS.log")).run(
        fmax=cfg.relax_fmax, steps=cfg.relax_steps
    )

    fmax_f    = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
    converged = fmax_f < cfg.relax_fmax
    log.info(f"  fmax_final = {fmax_f:.5f}  "
             f"({'OK' if converged else 'NOT CONVERGED'})")
    log.info(f"  E_f = {atoms.get_potential_energy():.6f} eV   "
             f"V_f = {atoms.get_volume():.3f} Å^3")
    if not converged:
        log.warning("  Relaxation did not converge — proceeding anyway.")

    relaxed = atoms.copy()
    ase_write(str(relax_path), relaxed, format="vasp", direct=True)
    ckpt.mark("relax", {
        "fmax": fmax_f, "converged": converged,
        "energy": atoms.get_potential_energy(),
        "volume": atoms.get_volume(),
    })
    log.info(f"  Done in {time.time()-t0:.1f}s")
    return relaxed


# =============================================================================
# Stage 3: Generate supercells
# =============================================================================

def stage_generate(
    atoms:   Atoms,
    cfg:     Config,
    ckpt:    Checkpoint,
    log:     logging.Logger,
    out_dir: Path,
) -> Phono3py:
    log.info("─" * 60)
    log.info("STAGE 3  Generate displaced supercells")
    log.info("─" * 60)

    yaml_path = out_dir / "phono3py_disp.yaml"
    unitcell  = ase_to_phonopy(atoms)

    ph3 = Phono3py(
        unitcell,
        supercell_matrix = cfg.supercell_matrix,
        primitive_matrix = cfg.primitive_matrix_parsed,
        symprec          = cfg.symprec,
        log_level        = 1,
        **_ph3_lang(),   # v4: lang="C" restores v3 per-q performance
    )
    log.info(f"  Space group : {ph3.symmetry.get_international_table()}")

    if ckpt.done("generate") and yaml_path.exists() and cfg.resume:
        log.info("  [RESUME] Loading from phono3py_disp.yaml")
        ph3yml = Phono3pyYaml()
        ph3yml.read(str(yaml_path))
        ph3.dataset = ph3yml.dataset
    else:
        kw = dict(distance=cfg.amplitude, is_plusminus=True, is_diagonal=True)
        if cfg.cutoff_pair > 0:
            kw["cutoff_pair_distance"] = cfg.cutoff_pair
        ph3.generate_displacements(**kw)
        ph3.save(str(yaml_path))
        ckpt.mark("generate")

    supercells = ph3.supercells_with_displacements
    n_active   = sum(1 for s in supercells if s is not None)
    log.info(f"  SC matrix    : {cfg.supercell_matrix.tolist()}")
    log.info(f"  SC atoms     : {len(ph3.supercell)}")
    log.info(f"  Cutoff pair  : {cfg.cutoff_pair} Å")
    log.info(f"  Displacement : {cfg.amplitude} Å  (+/-)")
    log.info(f"  Displacements: {n_active} active / {len(supercells)} total")

    disp_dir = out_dir / "supercells"
    disp_dir.mkdir(exist_ok=True)
    for i, sc in enumerate(supercells):
        if sc is not None:
            ase_write(
                str(disp_dir / f"disp-{i+1:05d}.vasp"),
                phonopy_to_ase(sc), format="vasp", direct=True,
            )
    return ph3


# =============================================================================
# Stage 4: Forces
# =============================================================================

def _force_worker(args: tuple) -> tuple[int, np.ndarray | None]:
    idx, sc, cfg_dict = args
    if sc is None:
        return idx, None
    cfg   = Config(**cfg_dict)
    calc  = make_calc(cfg)
    atoms = phonopy_to_ase(sc)
    atoms.calc = calc
    try:
        return idx, atoms.get_forces()
    except Exception as e:
        logging.error(f"Force eval failed slot {idx}: {e}")
        return idx, np.zeros_like(atoms.get_positions())


def stage_forces(
    ph3:     Phono3py,
    cfg:     Config,
    ckpt:    Checkpoint,
    log:     logging.Logger,
    out_dir: Path,
) -> Phono3py:
    log.info("─" * 60)
    log.info("STAGE 4  Evaluate forces")
    log.info("─" * 60)

    cache_dir   = out_dir / "forces_cache"
    cache_dir.mkdir(exist_ok=True)
    supercells  = ph3.supercells_with_displacements
    n_slots     = len(supercells)
    force_shape = ph3.supercell.positions.shape

    perf_cache = cache_dir / "forces_perfect.npy"
    if perf_cache.exists() and cfg.resume:
        f_perf = np.load(str(perf_cache))
    else:
        log.info("  Perfect supercell residual forces …")
        sc_perf = phonopy_to_ase(ph3.supercell)
        sc_perf.calc = make_calc(cfg)
        f_perf = sc_perf.get_forces()
        np.save(str(perf_cache), f_perf)

    resid = float(np.max(np.abs(f_perf)))
    log.info(f"  Max residual force: {resid:.3e} eV/Å"
             + (" check relaxation!" if resid > 0.01 else ""))

    to_compute, cached = [], {}
    for i, sc in enumerate(supercells):
        if sc is None:
            continue
        p = cache_dir / f"forces_{i:05d}.npy"
        if p.exists() and cfg.resume:
            cached[i] = np.load(str(p))
        else:
            to_compute.append((i, sc, asdict(cfg)))

    log.info(f"  Cached: {len(cached)}  To compute: {len(to_compute)}"
             f"  Workers: {cfg.n_workers}")

    if to_compute:
        t0 = time.time()
        if cfg.n_workers > 1:
            from multiprocessing import Pool
            with Pool(cfg.n_workers) as pool:
                results = pool.map(_force_worker, to_compute)
        else:
            results = []
            for k, arg in enumerate(to_compute):
                results.append(_force_worker(arg))
                if (k + 1) % max(1, len(to_compute) // 10) == 0:
                    log.info(f"    {k+1}/{len(to_compute)}")
        for idx, f in results:
            if f is not None:
                np.save(str(cache_dir / f"forces_{idx:05d}.npy"), f)
                cached[idx] = f
        log.info(f"  Force eval done in {time.time()-t0:.1f}s")

    forces_all = np.zeros((n_slots, force_shape[0], force_shape[1]))
    for i, f in cached.items():
        forces_all[i] = f - f_perf

    ph3.forces = forces_all

    ph3yml = Phono3pyYaml()
    ph3yml.read(str(out_dir / "phono3py_disp.yaml"))
    write_FORCES_FC3(
        ph3yml.dataset,
        forces_fc3 = [forces_all[i] for i, sc in enumerate(supercells)
                      if sc is not None],
        filename   = str(out_dir / "FORCES_FC3"),
    )
    ckpt.mark("forces", {"n_computed": len(to_compute), "max_residual": resid})
    return ph3


# =============================================================================
# Stage 5: Force constants
# =============================================================================

def stage_fc(
    ph3:     Phono3py,
    cfg:     Config,
    ckpt:    Checkpoint,
    log:     logging.Logger,
    out_dir: Path,
) -> Phono3py:
    log.info("─" * 60)
    log.info("STAGE 5  Force constants")
    log.info("─" * 60)

    fc2_path = out_dir / "fc2.hdf5"
    fc3_path = out_dir / "fc3.hdf5"

    # produce_fc3 produces both FC2 and FC3 internally — no separate
    # produce_fc2 call needed.  Both are checkpointed together since FC2
    # alone is not useful without FC3 for thermal conductivity.
    if (ckpt.done("fc3") and fc3_path.exists() and
            fc2_path.exists() and cfg.resume):
        log.info("  [RESUME] FC2 from fc2.hdf5")
        ph3.fc2 = _read_fc2(fc2_path, ph3)
        log.info(f"  FC2 shape: {ph3.fc2.shape}")
        log.info("  [RESUME] FC3 from fc3.hdf5")
        ph3.fc3 = _read_fc3(fc3_path)
        log.info(f"  FC3 shape: {ph3.fc3.shape}")
    else:
        t0 = time.time()
        ph3.produce_fc3(symmetrize_fc3r=True, is_compact_fc=False)
        write_fc2_to_hdf5(ph3.fc2, filename=str(fc2_path))
        write_fc3_to_hdf5(ph3.fc3, filename=str(fc3_path))
        log.info(f"  FC2 {ph3.fc2.shape}  FC3 {ph3.fc3.shape}"
                 f"  ({time.time()-t0:.1f}s)")
        ckpt.mark("fc3", {"fc2_shape": list(ph3.fc2.shape),
                           "fc3_shape": list(ph3.fc3.shape)})

    return ph3


# =============================================================================
# Load ph3 from disk
# =============================================================================

def load_ph3_from_disk(out_dir: Path, cfg: Config) -> Phono3py:
    """
    Rebuild a Phono3py object from FC2/FC3 on disk.

    Two code paths:

    A) Standard phono3py pipeline  (phono3py_disp.yaml present)
       Unit cell, supercell matrix, and primitive matrix are read from the
       YAML.  Dataset is also restored (needed for some phono3py internals).

    B) SCPH / hiphive pipeline  (no phono3py_disp.yaml)
       The YAML is never written in this workflow.  Build Phono3py directly
       from cfg.structure (POSCAR) + cfg.supercell_matrix + cfg.primitive_matrix.
       Requires --structure, --supercell, and --primitive_matrix to be set
       correctly on the CLI.
    """
    fc2_path  = out_dir / "fc2.hdf5"
    fc3_path  = out_dir / "fc3.hdf5"
    yaml_path = out_dir / "phono3py_disp.yaml"

    for p in (fc2_path, fc3_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required file missing: {p}\n"
                "Run the full pipeline or SCPH workflow first."
            )

    if yaml_path.exists():
        # ── Path A: standard phono3py pipeline ───────────────────────────
        ph3yml = Phono3pyYaml()
        ph3yml.read(str(yaml_path))
        ph3 = Phono3py(
            ph3yml.unitcell,
            supercell_matrix = ph3yml.supercell_matrix,
            primitive_matrix = ph3yml.primitive_matrix,
            symprec          = cfg.symprec,
            log_level        = 1,
            **_ph3_lang(),   # v4: lang="C" restores v3 per-q performance
        )
        ph3.dataset = ph3yml.dataset

    else:
        # ── Path B: SCPH / hiphive workflow — no phono3py_disp.yaml ──────
        # Build Phono3py from the original POSCAR + CLI supercell/pmat.
        struct_path = Path(cfg.structure)
        if not struct_path.exists():
            raise FileNotFoundError(
                f"phono3py_disp.yaml not found in {out_dir}.\n"
                f"This is expected for the SCPH/hiphive workflow, but "
                f"--structure {cfg.structure!r} does not exist.\n"
                "Pass the original POSCAR via --structure and the matching "
                "--supercell / --primitive_matrix flags."
            )
        logging.info(
            "  phono3py_disp.yaml not found — building Phono3py from "
            f"--structure {cfg.structure} (SCPH/hiphive path)"
        )
        print('primitive axis is: ',cfg.primitive_matrix_parsed)
        atoms    = ase_read(str(struct_path))
        unitcell = ase_to_phonopy(atoms)
        ph3 = Phono3py(
            unitcell,
            supercell_matrix = cfg.supercell_matrix,
            primitive_matrix = cfg.primitive_matrix_parsed,
            symprec          = cfg.symprec,
            log_level        = 1,
            **_ph3_lang(),   # v4: lang="C" restores v3 per-q performance
        )
        logging.info(f"  supercell built: {len(ph3.supercell)} atoms")

    ph3.fc2 = _read_fc2(fc2_path, ph3)
    ph3.fc3 = _read_fc3(fc3_path)
    return ph3


# =============================================================================
# Irreducible q-points  (phono3py-native, no spglib index-space mismatch)
# =============================================================================

def _get_ir_grid_points(ph3: Phono3py, mesh: list[int],
                        log: logging.Logger | None = None) -> np.ndarray:
    """
    Return irreducible q-point indices in phono3py's BZGrid numbering.

    IMPORTANT: the caller MUST call ph3.mesh_numbers and
    ph3.init_phph_interaction() before calling this function.
    This function deliberately does NOT call init_phph_interaction —
    doing so would run it twice, which is extremely expensive for large
    systems (symmetrize_fc3q=True on a MOF-5 848-atom supercell takes
    many minutes; doubling it causes the apparent hang).

    Strategy (in order):
      1. ph3.grid (BZGrid) public API via get_ir_grid_points_compat() —
         pure grid/symmetry construction, no phonon-phonon interaction or
         conductivity work involved. Cheap.
         NOTE: BZGrid itself has no `ir_grid_points` attribute on phono3py
         v4 (its grid module moved to phonopy.phonon.grid — see
         phono3py_compat.py); the irreducible points must be computed via
         get_ir_grid_points(bz_grid), which returns GR-grid indices, then
         mapped to BZ-grid indices via bz_grid.grg2bzg (see
         phono3py/cui/kaccum_script.py for the reference usage).
         get_ir_grid_points_compat() picks the correct import path
         (phonopy vs. phono3py) based on the installed version.
      2. phono3py RTA conductivity object .grid_points — LAST RESORT ONLY.
         Despite appearances, constructing this object (or touching its
         attributes) has been observed to actually run the single-mode
         RTA solve rather than just exposing grid indices — confirmed by
         setting log_level>0 and watching it compute kappa at the dummy
         300 K "irrelevant" temperature. It is NOT a cheap accessor on
         this phono3py version and must not be tried first. Kept only as
         a fallback for phono3py versions where Path 1 fails, with a loud
         warning so the cost is visible rather than silently eaten every
         call.
      3. Raise — spglib fallback deliberately removed to prevent mismatches
         (its regular-mesh indices differ from phono3py's BZGrid indices).
    """
    def _log(msg):
        if log is not None:
            log.info(msg)
        else:
            logging.info(msg)

    # ── Path 1 (preferred): ph3.grid (BZGrid) public API ──────────────────
    # Pure grid/symmetry construction — no init_phph_interaction, no
    # triplet enumeration, no scattering rates, no kappa. Cheap.
    bz_grid = ph3.grid
    if bz_grid is not None:
        try:
            ir_gr_points, _weights, _ir_grid_map = get_ir_grid_points_compat(bz_grid)
            pts = np.asarray(bz_grid.grg2bzg[ir_gr_points], dtype=int)
            if pts.ndim == 1 and len(pts) > 0:
                _log(f"  ir_grid_points via get_ir_grid_points_compat: n={len(pts)}")
                return pts
        except Exception as e:
            _log(f"  ph3.grid (BZGrid) path failed: {e}")

    # ── Path 2 (last resort — EXPENSIVE): RTA conductivity object ─────────
    # Only reached if Path 1 fails. Confirmed to actually run RTA-style
    # computation on this install rather than being a lightweight
    # constructor, so this is a genuine fallback, not a preferred route.
    _log("  WARNING: falling back to constructing an RTA conductivity "
         "object just to read .grid_points — this has been observed to "
         "actually run part of the thermal conductivity solve (confirmed "
         "via log_level>0 showing a real 300 K RTA calculation), not just "
         "return grid indices. This path is slow; only used because "
         "the BZGrid path above was unavailable.")
    try:
        tc = get_thermal_conductivity_RTA_compat(
            ph3._interaction,
            temperatures = [300.0],   # value irrelevant — just need grid_points
            log_level    = 1,
        )
        pts = np.array(tc.grid_points, dtype=int)
        if len(pts) > 0:
            logging.info(
                f"  ir_grid_points via RTA object (expensive fallback): "
                f"n={len(pts)}"
            )
            return pts
    except Exception as e:
        logging.warning(f"  RTA grid_points fallback also failed: {e}")

    raise RuntimeError(
        "Could not determine irreducible grid points in phono3py BZGrid "
        "numbering. spglib is intentionally not used as a fallback because "
        "its regular-mesh indices differ from phono3py's BZGrid indices, "
        "causing filename mismatches (e.g. g38 written but g40 expected).\n"
        "Please report your phono3py version:\n"
        "  python -c 'import phono3py; print(phono3py.__version__)'"
    )


# =============================================================================
# Q-point progress tracking
# =============================================================================

def _sync_gp_progress(
    out_dir: Path,
    tag:     str,
    ckpt:    Checkpoint,
    log:     logging.Logger,
) -> set[int]:
    """
    Reconcile completed q-points from disk files and checkpoint.json.

    Strategy
    --------
    - Parse gp indices from kappa-m{tag}-g{N}.hdf5 filenames on disk.
    - Load done_gps list from checkpoint.json.
    - If a gp is in checkpoint but has no file -> file is source of truth,
      remove from set (will be recomputed).
    - If a gp has a file but is not in checkpoint -> add to set (crash recovery).
    - Persist merged set back to checkpoint.json.

    Returns
    -------
    set of integer grid-point indices that are genuinely complete.
    """
    # phono3py names per-q files as kappa-m{tag}-g{N}.hdf5
    # For large systems it may split by band: kappa-m{tag}-g{N}-b{B}.hdf5
    # The regex extracts the grid-point index N from either form.
    pattern  = re.compile(rf"kappa-m{re.escape(tag)}-g(\d+)(?:-b\d+)?\.hdf5")
    disk_gps = set()
    for f in out_dir.glob(f"kappa-m{tag}-g*.hdf5"):
        m = pattern.match(f.name)
        if m:
            disk_gps.add(int(m.group(1)))

    ckpt_gps = set(ckpt.get("done_gps", []))

    on_disk_only = disk_gps - ckpt_gps
    in_ckpt_only = ckpt_gps - disk_gps

    if on_disk_only:
        log.info(
            f"  Sync: {len(on_disk_only)} q-point file(s) on disk "
            f"not yet in checkpoint — adding"
        )
    if in_ckpt_only:
        log.warning(
            f"  Sync: {len(in_ckpt_only)} q-point(s) in checkpoint "
            f"but no file on disk — will recompute: {sorted(in_ckpt_only)}"
        )

    # File on disk is the source of truth
    merged = disk_gps
    ckpt.set("done_gps", sorted(merged))

    log.info(
        f"  Progress sync: disk={len(disk_gps)}  "
        f"checkpoint={len(ckpt_gps)}  merged={len(merged)}"
    )
    return merged


def _kappa_config_fingerprint(cfg: Config) -> dict:
    """Physics-relevant settings baked into kappa-m*-g*.hdf5 / kappa-m*.hdf5.

    Anything not listed here (parallel_mode, n_workers, resume, ...) does
    not change the numbers written to these files, so it is deliberately
    excluded.
    """
    return {
        "mesh":            cfg.mesh_list,
        "solver":          cfg.solver,
        "transport_type":  cfg.transport_type,
        "isotope":         cfg.isotope,
        "mass_variances":  cfg.mass_variances_parsed,
        "temperatures":    cfg.temperature_list,
    }


def _invalidate_stale_kappa_cache(
    cfg:     Config,
    ckpt:    Checkpoint,
    log:     logging.Logger,
    out_dir: Path,
) -> None:
    """
    Move aside kappa result files if they were computed under different
    BTE settings than the current run.

    --resume (default True) otherwise trusts any kappa-m{mesh}-g*.hdf5 /
    kappa-m{mesh}.hdf5 / kappa_summary.json file found on disk purely by
    filename, regardless of which solver/isotope/mass_variances/
    temperatures settings produced it. Toggling --isotope (or solver,
    transport_type, mass_variances, temperatures) while pointing at an
    out_dir that already has cached results for the same mesh therefore
    silently returns the *old* result instead of recomputing — this is
    the cause of "the --isotope flag doesn't seem to do anything".
    """
    tag     = cfg.mesh_tag
    current = _kappa_config_fingerprint(cfg)
    previous = ckpt.get("kappa_config")

    if previous is not None and previous != current:
        stale_dir = out_dir / f"_stale_kappa_{int(time.time())}"
        stale_dir.mkdir(exist_ok=True)
        moved = []
        for pat in (f"kappa-m{tag}-g*.hdf5", f"kappa-m{tag}.hdf5",
                    "kappa_summary.json", "kappa_vs_T.png"):
            for f in out_dir.glob(pat):
                f.rename(stale_dir / f.name)
                moved.append(f.name)

        ckpt.set("done_gps", [])
        ckpt.set("collect", {"done": False})

        log.warning(
            "  [CACHE INVALIDATED] BTE settings changed since the cached "
            "results in this out_dir were computed:\n"
            f"    previous : {previous}\n"
            f"    current  : {current}\n"
            f"  Moved {len(moved)} stale file(s) to {stale_dir.name}/ "
            "— recomputing from scratch."
        )

    ckpt.set("kappa_config", current)


# =============================================================================
# Result extraction
# =============================================================================

def _resolve_kappa_attrs(
    tc,
    has_coherence: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Extract (kappa_total, kappa_intra, kappa_inter) from the thermal
    conductivity object, tolerating different attribute names across
    phono3py versions/transport_type formulations.

    Current inter-band transport naming (SMM19/NJC23/IBDB19):
        kappa        -> total
        kappa_intra  -> intra-band (particle-like) contribution
        kappa_inter  -> inter-band (coherence) contribution

    Older Wigner-only naming (kept as fallback for older installs):
        kappa_TOT_RTA -> total
        kappa_P_RTA   -> particle-like (~ intra)
        kappa_C_RTA   -> coherence (~ inter)

    Returns
    -------
    (kappa_tot, kappa_intra, kappa_inter, attr_label)
    attr_label : str describing which attribute set was used, for logging.
    """
    def first_attr(names):
        return next((n for n in names if hasattr(tc, n)), None)

    if has_coherence:
        total_attr = first_attr(["kappa", "kappa_TOT_RTA", "kappa_total"])
        intra_attr = first_attr(["kappa_intra", "kappa_intra_RTA",
                                 "kappa_P_RTA", "kappa_P"])
        inter_attr = first_attr(["kappa_inter", "kappa_inter_RTA",
                                 "kappa_C_RTA", "kappa_C"])

        if intra_attr is not None and inter_attr is not None:
            # Best case: both intra and inter are explicit attributes
            # (current inter-band transport API).
            kappa_intra = np.array(getattr(tc, intra_attr))
            kappa_inter = np.array(getattr(tc, inter_attr))
            if total_attr is not None:
                kappa_tot = np.array(getattr(tc, total_attr))
            else:
                kappa_tot = kappa_intra + kappa_inter
            return (kappa_tot, kappa_intra, kappa_inter,
                    f"total={total_attr or 'intra+inter'} "
                    f"intra={intra_attr} inter={inter_attr}")

        if intra_attr is not None and total_attr is not None:
            # Older Wigner-style API: only total + particle-like (intra)
            # attributes exist (e.g. kappa_TOT_RTA + kappa_P_RTA, no
            # separate kappa_C_RTA). Derive inter by subtraction, exactly
            # as the original kappa_C = kappa_tot - kappa_P logic did.
            kappa_tot   = np.array(getattr(tc, total_attr))
            kappa_intra = np.array(getattr(tc, intra_attr))
            kappa_inter = kappa_tot - kappa_intra
            return (kappa_tot, kappa_intra, kappa_inter,
                    f"total={total_attr} intra={intra_attr} "
                    f"inter=derived(total-intra)")

        if total_attr is not None:
            # Only a total is available — no split possible at all.
            kappa_tot = np.array(getattr(tc, total_attr))
            return (kappa_tot, kappa_tot.copy(), np.zeros_like(kappa_tot),
                    f"total={total_attr} (no intra/inter split found)")

    # ── No coherence requested, or nothing found above ─────────────────────
    for n in ("kappa_P_RTA", "kappa", "kappa_TOT_RTA"):
        if hasattr(tc, n):
            kappa_tot = np.array(getattr(tc, n))
            return kappa_tot, kappa_tot.copy(), np.zeros_like(kappa_tot), n

    raise AttributeError(
        "Cannot find kappa in thermal conductivity object. Available: "
        + str([a for a in dir(tc) if "kappa" in a.lower()])
    )


def _extract_results(
    tc,
    temps:          list[float],
    mesh:           list[int],
    log:            logging.Logger,
    solver:         str        = "rta",
    transport_type: str | None = None,
) -> dict:
    has_coherence = transport_type is not None

    kappa_tot, kappa_intra, kappa_inter, attr_label = _resolve_kappa_attrs(
        tc, has_coherence
    )
    log.info(f"  kappa attrs  : {attr_label}")

    # ── Normalise to shape (n_temps, 6) ──────────────────────────────────
    # Phono3py may return (n_temps, 6), (n_temps, 1, 6), or (n_temps, 3, 3)
    # depending on version and transport_type.  Squeeze out length-1 dims
    # then convert a full 3x3 tensor to 6-component Voigt form if needed.
    def _to_voigt(arr: np.ndarray) -> np.ndarray:
        a = np.squeeze(arr)               # remove length-1 axes
        if a.ndim == 1:                   # single temperature already squeezed
            a = a[np.newaxis, :]
        if a.ndim == 3 and a.shape[1:] == (3, 3):
            # Full tensor -> Voigt  [xx, yy, zz, yz, xz, xy]
            return np.stack([
                a[:, 0, 0], a[:, 1, 1], a[:, 2, 2],
                a[:, 1, 2], a[:, 0, 2], a[:, 0, 1],
            ], axis=1)
        if a.ndim == 2:
            return a
        raise ValueError(f"Unexpected kappa shape after squeeze: {arr.shape}")

    kappa_tot   = _to_voigt(kappa_tot)
    kappa_intra = _to_voigt(kappa_intra)
    kappa_inter = _to_voigt(kappa_inter)

    label   = ("LBTE" if solver.lower() == "lbte" else "RTA")
    label  += f" + {transport_type}" if has_coherence else ""
    results = {"mesh": mesh, "solver": label, "transport_type": transport_type,
               "temperatures": {}}

    hdr = f"\n  {'T(K)':<7} {'kappa_xx':>8} {'kappa_yy':>8} {'kappa_zz':>8} {'kappa_iso':>9}"
    if has_coherence:
        hdr += f" {'kappa_intra_iso':>11} {'kappa_inter_iso':>11}"
    log.info(hdr + "  [W/mK]")
    log.info("  " + "─" * (56 if not has_coherence else 80))

    for i, T in enumerate(temps):
        k     = kappa_tot[i]           # shape (6,) guaranteed by _to_voigt
        kIa   = kappa_intra[i]
        kIe   = kappa_inter[i]
        iso   = float((k[0] + k[1] + k[2]) / 3)
        IaIso = float((kIa[0] + kIa[1] + kIa[2]) / 3)
        IeIso = float((kIe[0] + kIe[1] + kIe[2]) / 3)

        entry = {
            "kappa_TOT":       [float(v) for v in k[:6]],
            "kappa_intra":     [float(v) for v in kIa[:6]],
            "kappa_iso":       iso,
            "kappa_intra_iso": IaIso,
        }
        if has_coherence:
            entry["kappa_inter"]     = [float(v) for v in kIe[:6]]
            entry["kappa_inter_iso"] = IeIso

        results["temperatures"][str(int(T))] = entry

        row = (f"  {int(T):<7} {float(k[0]):>8.3f} {float(k[1]):>8.3f} "
               f"{float(k[2]):>8.3f} {iso:>9.3f}")
        if has_coherence:
            row += f" {IaIso:>11.3f} {IeIso:>11.3f}"
        log.info(row)

    return results


# =============================================================================
# Stage 6a: BTE serial / OMP  (all q at once, no per-q resume)
# =============================================================================

def _run_bte_core(
    ph3:         Phono3py,
    cfg:         Config,
    log:         logging.Logger,
    out_dir:     Path,
    gp_list:     list[int] | None = None,
    write_gamma: bool = False,
) -> dict:
    mesh           = cfg.mesh_list
    temps          = cfg.temperature_list
    is_lbte, ctype = _resolve_bte(cfg)

    log.info(f"  Mesh         : {mesh}")
    log.info(f"  Temperatures : {temps} K")
    log.info(f"  Solver       : {_solver_label(cfg)}")
    log.info(f"  Grid points  : {'all' if gp_list is None else gp_list}")

    ph3.mesh_numbers = mesh
    _init_phph(ph3, log)

    run_kw = dict(
        temperatures   = temps,
        is_LBTE        = is_lbte,
        transport_type = ctype,
        is_isotope     = cfg.isotope,
        mass_variances = cfg.mass_variances_parsed,
        write_kappa    = not write_gamma,
    )
    if gp_list is not None:
        run_kw["grid_points"] = gp_list
    if write_gamma:
        run_kw["write_gamma"] = True

    orig_cwd = os.getcwd()
    os.chdir(str(out_dir))
    try:
        t0 = time.time()
        ph3.run_thermal_conductivity(**run_kw)
        log.info(f"  BTE done in {time.time()-t0:.1f}s")
    finally:
        os.chdir(orig_cwd)

    if write_gamma:
        return {"partial": True, "gp": gp_list}

    return _extract_results(
        ph3.thermal_conductivity, temps, mesh, log,
        solver=cfg.solver, transport_type=cfg.transport_type,
    )


# =============================================================================
# Stage 6b: BTE serial_gp  (one q at a time, full resume granularity)
# =============================================================================

def _run_bte_serial_gp(
    ph3:     Phono3py,
    cfg:     Config,
    ckpt:    Checkpoint,
    log:     logging.Logger,
    out_dir: Path,
) -> dict:
    """
    Process every irreducible q-point independently and sequentially.

    _sync_gp_progress() is called by stage_kappa() BEFORE this function,
    so done_gps in checkpoint.json is already up-to-date when we arrive.
    We simply read it here — no second disk scan needed.

    After each q-point:
      - done_gps  (sorted list) is written to checkpoint.json immediately.
      - gp_progress (human-readable summary) is also updated.
    When all q-points are done, calls stage_collect().
    """
    # ── Fast resume: already fully collected — skip grid/ph-ph setup ──────
    # stage_collect() has its own cheap resume check (summary JSON exists +
    # ckpt.done("collect")) that never touches ph3, but it only helps if we
    # reach it before doing grid/ph-ph setup here. Otherwise every
    # invocation — even ones with nothing left to compute — pays for
    # _get_ir_grid_points()'s expensive RTA-object fallback (its own
    # docstring notes this has been observed to actually run part of a
    # real solve, multiple minutes on this system).
    summary = out_dir / "kappa_summary.json"
    if ckpt.done("collect") and summary.exists():
        log.info("  [RESUME] kappa_summary.json already complete — "
                  "skipping grid/ph-ph setup")
        return stage_collect(cfg, ckpt, log, out_dir)

    mesh           = cfg.mesh_list
    tag            = cfg.mesh_tag
    is_lbte, ctype = _resolve_bte(cfg)

    # ── Init ph-ph interaction (done ONCE here; _get_ir_grid_points ──────
    # must NOT call it again or it runs twice — very expensive for large
    # systems and the primary cause of the apparent hang after stage_fc).
    ph3.mesh_numbers = mesh
    _init_phph(ph3, log)

    ir_pts = _get_ir_grid_points(ph3, mesh, log=log)
    n_ir   = len(ir_pts)
    log.info(f"  Irreducible q-points : {n_ir}")

    # ── Read done_gps already synced by stage_kappa ───────────────────────
    # stage_kappa called _sync_gp_progress before dispatching here,
    # so checkpoint.json already reflects disk reality.
    done_gps    = set(ckpt.get("done_gps", []))
    n_cached    = len(done_gps.intersection(set(ir_pts.tolist())))
    n_remaining = n_ir - n_cached
    log.info(f"  Already done : {n_cached}  Remaining : {n_remaining}")

    if n_remaining == 0:
        log.info("  [RESUME] All q-points complete — going to collect")
        return stage_collect(cfg, ckpt, log, out_dir)

    # ── Per-q loop ────────────────────────────────────────────────────────
    t0                = time.time()
    computed, skipped = 0, 0

    orig_cwd = os.getcwd()
    os.chdir(str(out_dir))   # phono3py writes kappa-m{mesh}-g{N}.hdf5 here
    try:
        for i, gp in enumerate(ir_pts):
            gp_int = int(gp)

            if gp_int in done_gps:
                skipped += 1
                if skipped <= 3 or skipped % 20 == 0:
                    log.info(
                        f"  [{i+1:4d}/{n_ir}]  gp={gp_int:<6d}  [cached]"
                    )
                continue

            log.info(f"  [{i+1:4d}/{n_ir}]  gp={gp_int:<6d} …")
            ph3.run_thermal_conductivity(
                temperatures   = cfg.temperature_list,
                is_LBTE        = is_lbte,
                transport_type = ctype,
                is_isotope     = cfg.isotope,
                mass_variances = cfg.mass_variances_parsed,
                grid_points    = [gp_int],
                write_gamma    = True,
                # no output_filename — phono3py writes
                # kappa-m{mesh}-g{N}.hdf5 in cwd automatically
            )
            computed += 1

            # ── Persist progress immediately after each q-point ───────────
            done_gps.add(gp_int)
            elapsed = round(time.time() - t0, 1)
            ckpt.set("done_gps", sorted(done_gps))
            ckpt.set("gp_progress", {
                "n_done":    len(done_gps),
                "n_total":   n_ir,
                "pct":       round(100 * (computed + skipped) / n_ir, 1),
                "computed":  computed,
                "skipped":   skipped,
                "last_gp":   gp_int,
                "elapsed_s": elapsed,
            })

    finally:
        os.chdir(orig_cwd)

    log.info(
        f"  Computed: {computed}  Cached: {skipped}  "
        f"Total: {n_ir}  Elapsed: {time.time()-t0:.1f}s"
    )

    return stage_collect(cfg, ckpt, log, out_dir)


# =============================================================================
# Stage 6c: BTE grid_points parallel
# =============================================================================

def _bte_gp_worker(args: tuple) -> str:
    worker_id, gp_subset, cfg_dict, out_dir_str = args
    import logging as _logging
    cfg     = Config(**cfg_dict)
    out_dir = Path(out_dir_str)
    log     = _logging.getLogger(f"worker_{worker_id}")

    is_lbte, ctype = _resolve_bte(cfg)
    ph3 = load_ph3_from_disk(out_dir, cfg)
    ph3.mesh_numbers = cfg.mesh_list
    _init_phph(ph3, log)

    orig_cwd = os.getcwd()
    os.chdir(str(out_dir))
    try:
        ph3.run_thermal_conductivity(
            temperatures   = cfg.temperature_list,
            is_LBTE        = is_lbte,
            transport_type = ctype,
            is_isotope     = cfg.isotope,
            mass_variances = cfg.mass_variances_parsed,
            grid_points    = gp_subset,
            write_gamma    = True,
        )
    finally:
        os.chdir(orig_cwd)

    tag         = cfg.mesh_tag
    gamma_files = [
        str(out_dir / f"kappa-m{tag}-g{gp}.hdf5")
        for gp in gp_subset
        if (out_dir / f"kappa-m{tag}-g{gp}.hdf5").exists()
    ]
    log.info(f"  Worker {worker_id}: gp={gp_subset}  files={len(gamma_files)}")
    return json.dumps({"worker_id": worker_id,
                       "gp": gp_subset,
                       "gamma_files": gamma_files})


def _balanced_chunks(
    ir_pts:     np.ndarray,
    n_jobs:     int,
    batch_size: int = 1,
) -> list[list[int]]:
    if batch_size > 1:
        batches = [ir_pts[i:i+batch_size].tolist()
                   for i in range(0, len(ir_pts), batch_size)]
        chunks  = [[] for _ in range(min(n_jobs, len(batches)))]
        for i, b in enumerate(batches):
            chunks[i % len(chunks)].extend(b)
        return [c for c in chunks if c]
    chunks = [[] for _ in range(min(n_jobs, len(ir_pts)))]
    for i, gp in enumerate(ir_pts):
        chunks[i % len(chunks)].append(int(gp))
    return [c for c in chunks if c]


def _run_bte_grid_parallel(
    ph3:     Phono3py,
    cfg:     Config,
    ckpt:    Checkpoint,
    log:     logging.Logger,
    out_dir: Path,
) -> dict:
    from multiprocessing import Pool

    # ── Fast resume: already fully collected — skip grid/ph-ph setup ──────
    # See the matching guard in _run_bte_serial_gp for why this must run
    # before _get_ir_grid_points().
    summary = out_dir / "kappa_summary.json"
    if ckpt.done("collect") and summary.exists():
        log.info("  [RESUME] kappa_summary.json already complete — "
                  "skipping grid/ph-ph setup")
        return stage_collect(cfg, ckpt, log, out_dir)

    mesh   = cfg.mesh_list
    n_jobs = cfg.n_workers
    ir_pts = _get_ir_grid_points(ph3, mesh, log=log)
    n_ir   = len(ir_pts)
    log.info(f"  Irreducible q-points : {n_ir}")
    log.info(f"  Workers              : {n_jobs}")
    log.info(f"  Batch size           : {cfg.gp_batch_size}")

    if cfg.gp_list is not None:
        ir_pts = np.array([gp for gp in cfg.gp_list
                           if gp in set(ir_pts.tolist())])
        log.info(f"  User-specified gp    : {ir_pts.tolist()}")

    chunks    = _balanced_chunks(ir_pts, n_jobs, cfg.gp_batch_size)
    cfg_dict  = asdict(cfg)
    args_list = [(i, chunk, cfg_dict, str(out_dir))
                 for i, chunk in enumerate(chunks)]
    log.info(f"  Chunks : {len(chunks)}  sizes={[len(c) for c in chunks]}")

    t0 = time.time()
    if n_jobs > 1:
        with Pool(processes=n_jobs) as pool:
            raw = pool.map(_bte_gp_worker, args_list)
    else:
        raw = [_bte_gp_worker(a) for a in args_list]
    log.info(f"  All workers done in {time.time()-t0:.1f}s")

    all_gamma = []
    for r in raw:
        all_gamma.extend(json.loads(r)["gamma_files"])
    log.info(f"  Files written : {len(all_gamma)}")

    return stage_collect(cfg, ckpt, log, out_dir)


# =============================================================================
# Stage 6 dispatcher
# =============================================================================

def stage_kappa(
    ph3:     Phono3py,
    cfg:     Config,
    ckpt:    Checkpoint,
    log:     logging.Logger,
    out_dir: Path,
) -> dict:
    log.info("─" * 60)
    log.info(f"STAGE 6  Thermal conductivity  [{cfg.parallel_mode}]  "
             f"[{_solver_label(cfg)}]")
    log.info("─" * 60)

    # ── Invalidate cached results from a different BTE config ─────────────
    # Must run before _sync_gp_progress, which otherwise treats any
    # kappa-m{mesh}-g*.hdf5 file on disk as valid regardless of the
    # solver/isotope/mass_variances/temperatures settings that produced it.
    _invalidate_stale_kappa_cache(cfg, ckpt, log, out_dir)

    # ── Always sync disk -> checkpoint before anything else ────────────────
    # Runs for ALL parallel modes. Scans kappa-m*-g*.hdf5 files on disk,
    # merges with done_gps in checkpoint.json, and persists the union back.
    # This ensures backward compatibility: files written by a previous run,
    # a different version, or the phono3py CLI are detected and recorded
    # in checkpoint.json even if done_gps was previously empty or missing.
    done_gps = _sync_gp_progress(out_dir, cfg.mesh_tag, ckpt, log)
    if done_gps:
        log.info(f"  Checkpoint updated: {len(done_gps)} q-point(s) "
                 f"already done — recorded in checkpoint.json")

    mode = cfg.parallel_mode.lower()

    if mode in ("serial", "omp"):
        if mode == "omp":
            log.info(f"  OMP_NUM_THREADS = "
                     f"{os.environ.get('OMP_NUM_THREADS', 'not set')}")
        return _run_bte_core(ph3, cfg, log, out_dir, gp_list=cfg.gp_list)

    elif mode == "serial_gp":
        return _run_bte_serial_gp(ph3, cfg, ckpt, log, out_dir)

    elif mode == "grid_points":
        if cfg.gp_list is not None and cfg.n_workers == 1:
            log.info("  Mode: single grid-point job (SLURM array task)")
            _run_bte_core(ph3, cfg, log, out_dir,
                          gp_list=cfg.gp_list, write_gamma=True)
            return {"partial": True, "gp": cfg.gp_list}
        else:
            log.info("  Mode: Python multi-worker grid-point parallel")
            return _run_bte_grid_parallel(ph3, cfg, ckpt, log, out_dir)

    else:
        raise NotImplementedError(
            f"Unknown parallel_mode: '{mode}'. "
            "Choose: serial | serial_gp | omp | grid_points"
        )


# =============================================================================
# Stage 7: Collect  kappa-m*-g*.hdf5 -> kappa-m*.hdf5
# =============================================================================

def stage_collect(
    cfg:         Config,
    ckpt:        Checkpoint,
    log:         logging.Logger,
    out_dir:     Path,
    gamma_files: list[str] | None = None,   # kept for API compat
) -> dict:
    log.info("─" * 60)
    log.info(f"STAGE 7  Collect  [{_solver_label(cfg)}]")
    log.info("─" * 60)

    # Idempotent: no-op if stage_kappa already ran with this config in this
    # process. Needed here too because `run_collect` can call stage_collect
    # directly without going through stage_kappa first.
    _invalidate_stale_kappa_cache(cfg, ckpt, log, out_dir)

    tag         = cfg.mesh_tag
    mesh        = cfg.mesh_list
    temps       = cfg.temperature_list
    final_kappa = out_dir / f"kappa-m{tag}.hdf5"
    summary     = out_dir / "kappa_summary.json"

    # ── Resume level 1: summary JSON already written ──────────────────────
    if ckpt.done("collect") and summary.exists():
        log.info("  [RESUME] kappa_summary.json exists — loading directly")
        return json.loads(summary.read_text())

    # ── Resume level 2: final kappa HDF5 exists but summary missing ───────
    if ckpt.done("collect") and final_kappa.exists():
        log.info("  [RESUME] Final kappa HDF5 exists — re-extracting results")
        is_lbte, ctype = _resolve_bte(cfg)
        ph3 = load_ph3_from_disk(out_dir, cfg)
        ph3.mesh_numbers = mesh
        _init_phph(ph3, log)
        orig_cwd = os.getcwd()
        os.chdir(str(out_dir))
        try:
            ph3.run_thermal_conductivity(
                temperatures   = temps,
                is_LBTE        = is_lbte,
                transport_type = ctype,
                is_isotope     = cfg.isotope,
                mass_variances = cfg.mass_variances_parsed,
                read_gamma     = True,
                write_kappa    = True,
            )
        finally:
            os.chdir(orig_cwd)
        results = _extract_results(
            ph3.thermal_conductivity, temps, mesh, log,
            solver=cfg.solver, transport_type=cfg.transport_type,
        )
        summary.write_text(json.dumps(results, indent=2))
        ckpt.mark("collect", {"summary": str(summary)})
        return results

    # ── Normal: assemble from per-q files ────────────────────────────────
    gp_files = sorted(out_dir.glob(f"kappa-m{tag}-g*.hdf5"))
    log.info(f"  Found {len(gp_files)} kappa-m{tag}-g*.hdf5 files")
    if not gp_files:
        raise FileNotFoundError(
            f"No kappa-m{tag}-g*.hdf5 files in {out_dir}.\n"
            "Run the BTE step (serial_gp / grid_points) first."
        )

    is_lbte, ctype = _resolve_bte(cfg)
    ph3 = load_ph3_from_disk(out_dir, cfg)
    ph3.mesh_numbers = mesh
    _init_phph(ph3, log)

    orig_cwd = os.getcwd()
    os.chdir(str(out_dir))
    try:
        ph3.run_thermal_conductivity(
            temperatures   = temps,
            is_LBTE        = is_lbte,
            transport_type = ctype,
            read_gamma     = True,
            write_kappa    = True,
        )
    finally:
        os.chdir(orig_cwd)

    results = _extract_results(
        ph3.thermal_conductivity, temps, mesh, log,
        solver=cfg.solver, transport_type=cfg.transport_type,
    )
    summary.write_text(json.dumps(results, indent=2))
    log.info(f"  Saved: {summary}")
    ckpt.mark("collect", {"summary": str(summary)})
    return results


# =============================================================================
# Plots
# =============================================================================

def stage_plots(results: dict, out_dir: Path, log: logging.Logger) -> None:
    if results.get("partial"):
        log.info("  Partial run — skipping plots.")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family":    "sans-serif",
            "font.size":      11,
            "axes.linewidth": 1.4,
            "savefig.dpi":    200,
            "savefig.bbox":   "tight",
        })

        temps  = [float(T) for T in results["temperatures"]]
        kappa  = results["temperatures"]
        has_coherence = any("kappa_inter_iso" in v for v in kappa.values())
        ttype  = results.get("transport_type") or "coherence"

        def _k(key):
            return [kappa[str(int(T))][key] for T in temps]

        kxx  = [kappa[str(int(T))]["kappa_TOT"][0] for T in temps]
        kyy  = [kappa[str(int(T))]["kappa_TOT"][1] for T in temps]
        kzz  = [kappa[str(int(T))]["kappa_TOT"][2] for T in temps]
        kiso = _k("kappa_iso")

        ncols = 2 if has_coherence else 1
        fig, axes = plt.subplots(
            1, ncols, figsize=(7*ncols, 5),
            gridspec_kw=dict(wspace=0.30),
        )
        if ncols == 1:
            axes = [axes]

        ax = axes[0]
        for y, lab, c, m in [
            (kxx,  r"$\kappa_{xx}$",         "#3A86FF", "o"),
            (kyy,  r"$\kappa_{yy}$",         "#E63946", "s"),
            (kzz,  r"$\kappa_{zz}$",         "#2A9D8F", "^"),
            (kiso, r"$\kappa_\mathrm{iso}$", "#888888", "D"),
        ]:
            ax.plot(temps, y, marker=m,
                    linestyle="--" if "iso" in lab else "-",
                    color=c, lw=2, ms=6, label=lab)
        ax.set_yscale("log")
        ax.set_xlabel("Temperature (K)", fontsize=12)
        ax.set_ylabel(r"$\kappa$ (W m$^{-1}$ K$^{-1}$)", fontsize=12)
        ax.set_title(f"Thermal conductivity  [{results.get('solver','')}]",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.text(0.02, 0.97, "(a)", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")

        if has_coherence:
            kIaIso = _k("kappa_intra_iso")
            kIeIso = _k("kappa_inter_iso")
            ax = axes[1]
            for y, lab, c, m, ls in [
                (kiso,   r"$\kappa_\mathrm{TOT}$",   "#3A86FF", "o",  "-"),
                (kIaIso, r"$\kappa_\mathrm{intra}$", "#E63946", "s",  "--"),
                (kIeIso, r"$\kappa_\mathrm{inter}$", "#2A9D8F", "^",  "-."),
            ]:
                ax.plot(temps, y, marker=m, linestyle=ls, color=c,
                        lw=2, ms=6,
                        markerfacecolor="none" if ls != "-" else c,
                        markeredgewidth=1.8, label=lab)
            ax.set_yscale("log")
            ax.set_xlabel("Temperature (K)", fontsize=12)
            ax.set_ylabel(r"$\kappa_\mathrm{iso}$ (W m$^{-1}$ K$^{-1}$)",
                          fontsize=12)
            ax.set_title(f"Intra / Inter decomposition  [{ttype}]",
                         fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, which="both", linestyle="--", alpha=0.3)
            ax.text(0.02, 0.97, "(b)", transform=ax.transAxes,
                    fontsize=12, fontweight="bold", va="top")

        plt.savefig(str(out_dir / "kappa_vs_T.png"))
        plt.close()
        log.info("  Saved: kappa_vs_T.png")

    except Exception as e:
        log.warning(f"  Plot failed: {e}")


# =============================================================================
# Pipeline entry points
# =============================================================================

def run_full(cfg: Config) -> dict:
    out_dir = Path(cfg.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")

    log  = setup_logging(out_dir)
    ckpt = Checkpoint(out_dir / "checkpoint.json")

    log.info("=" * 60)
    log.info("  Thermal Transport Pipeline")
    log.info("=" * 60)
    print_version_banner(log)
    log.info(f"  Structure     : {cfg.structure}")
    log.info(f"  Calculator    : {cfg.calc_type.upper()}"
             + (f" / {cfg.mace_model}" if cfg.calc_type == "mace"
                else f" / {cfg.uma_model} ({cfg.uma_task})"))
    log.info(f"  DFT-D3        : {cfg.dftd3}")
    log.info(f"  Supercell     : {cfg.supercell}")
    log.info(f"  Cutoff pair   : {cfg.cutoff_pair} Å")
    log.info(f"  Mesh          : {cfg.mesh}")
    log.info(f"  Temperatures  : {cfg.temperatures} K")
    log.info(f"  Solver        : {_solver_label(cfg)}")
    log.info(f"  Isotope       : {cfg.isotope}"
             + (f"  (mass_variances={cfg.mass_variances_parsed})"
                if cfg.isotope and cfg.mass_variances_parsed else ""))
    log.info(f"  Parallel mode : {cfg.parallel_mode}")
    log.info(f"  Workers       : {cfg.n_workers}")
    log.info(f"  Resume        : {cfg.resume}")

    t0 = time.time()
    atoms   = stage_read    (cfg,      log, out_dir)
    relaxed = stage_relax   (atoms,   cfg, ckpt, log, out_dir)
    ph3     = stage_generate(relaxed, cfg, ckpt, log, out_dir)
    ph3     = stage_forces  (ph3,     cfg, ckpt, log, out_dir)
    ph3     = stage_fc      (ph3,     cfg, ckpt, log, out_dir)
    results = stage_kappa   (ph3,     cfg, ckpt, log, out_dir)
    stage_plots(results, out_dir, log)

    if not results.get("partial"):
        (out_dir / "kappa_summary.json").write_text(
            json.dumps(results, indent=2)
        )

    log.info("=" * 60)
    log.info(f"  DONE  ({time.time()-t0:.1f}s)")
    log.info(f"  Results: {out_dir.resolve()}")
    log.info("=" * 60)
    return results


def run_bte(cfg: Config) -> dict:
    out_dir = Path(cfg.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log  = setup_logging(out_dir, name="bte")
    ckpt = Checkpoint(out_dir / "checkpoint.json")

    log.info("=" * 60)
    log.info(f"  BTE-only  [{_solver_label(cfg)}]")
    log.info("=" * 60)
    print_version_banner(log)
    log.info(f"  Mesh          : {cfg.mesh}")
    log.info(f"  Temperatures  : {cfg.temperatures} K")
    log.info(f"  Parallel mode : {cfg.parallel_mode}")
    log.info(f"  Grid points   : {cfg.gp if cfg.gp else 'all'}")

    ph3     = load_ph3_from_disk(out_dir, cfg)
    results = stage_kappa(ph3, cfg, ckpt, log, out_dir)
    stage_plots(results, out_dir, log)
    return results


def run_collect(cfg: Config) -> dict:
    out_dir = Path(cfg.out_dir).resolve()
    log     = setup_logging(out_dir, name="collect")
    print_version_banner(log)
    ckpt    = Checkpoint(out_dir / "checkpoint.json")
    results = stage_collect(cfg, ckpt, log, out_dir)
    stage_plots(results, out_dir, log)
    return results


# =============================================================================
# CLI
# =============================================================================

def _add_calc_args(p: argparse.ArgumentParser) -> None:
    grp = p.add_argument_group("Calculator")
    grp.add_argument(
        "--calc_type", default="mace", choices=["mace", "uma"],
        help="mace: MACECalculator (needs --mace_model). "
             "uma: FAIRChem UMA (needs --hf_token).",
    )
    grp.add_argument("--mace_model",  default="",
                     help="Path to MACE model file")
    grp.add_argument("--mace_head",   default="",
                     help="MACE fine-tuning head, e.g. 'omat_pbe'")
    grp.add_argument("--mace_device", default="cuda",
                     choices=["cpu", "cuda"])
    grp.add_argument("--mace_dtype",  default="float64",
                     choices=["float32", "float64"])
    grp.add_argument("--uma_model",   default="uma-s-1p2",
                     choices=["uma-s-1p2", "uma-m-1p1"])
    grp.add_argument("--uma_task",    default="omc",
                     help="FAIRChem task name, e.g. 'omc', 's2ef'")
    grp.add_argument("--uma_device",  default="cuda",
                     choices=["cpu", "cuda"])
    grp.add_argument("--hf_token",
                     default=os.environ.get("HF_TOKEN", ""),
                     help="HuggingFace token for UMA (or set HF_TOKEN env var)")
    grp.add_argument("--dftd3", action="store_true",
                     help="Add Grimme D3-BJ dispersion (PBE params)")


def _add_bte_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mesh",         default="11 11 11")
    p.add_argument("--temperatures", default="300")
    p.add_argument("--solver",       default="rta", choices=["rta", "lbte"],
                   help="rta: single-mode RTA. lbte: iterative BTE.")
    p.add_argument(
        "--transport_type",
        default = None,
        choices = ["SMM19", "NJC23", "IBDB19"],
        help    = (
            "Include coherence kappa_C via an inter-band transport formulation. "
            "Omit for standard particle-like transport (kappa_P only). "
            "SMM19  : Simoncelli-Marzari-Mauri (2019) Wigner transport "
            "         equation — the original/default coherence formulation. "
            "NJC23  : alternative inter-band transport formulation. "
            "IBDB19 : Isaeva-Barbalinardo-Donadio-Baroni (2019) formulation."
        ),
    )
    p.add_argument("--isotope",      action="store_true",
                   help="Include phonon-isotope scattering "
                        "(is_isotope=True in phono3py).")
    p.add_argument(
        "--mass_variances",
        default = "",
        help    = (
            "Space-separated per-element isotope mass-variance (g-factors), "
            "one value per species in the primitive-cell element order. "
            "Only used with --isotope. Empty = phono3py's built-in "
            "natural-abundance values."
        ),
    )
    p.add_argument("--symprec",      type=float, default=1e-5)
    p.add_argument("--out_dir",      default="results")
    p.add_argument("--resume",       action="store_true")
    p.add_argument(
        "--parallel_mode",
        default="serial",
        choices=["serial", "serial_gp", "omp", "grid_points"],
        help=(
            "serial     : all q at once, no per-q resume. "
            "serial_gp  : one q at a time; checkpoint + disk scan on restart "
            "             -> per-q resume, crash-safe. "
            "omp        : same as serial; set OMP_NUM_THREADS externally. "
            "grid_points: split q across Python workers or SLURM arrays."
        ),
    )
    p.add_argument("--gp",           default="",
                   help="Grid points for this job (space-separated). "
                        "Empty = all irreducible q-points.")
    p.add_argument("--gp_batch_size", type=int, default=1)
    p.add_argument("--n_workers",     type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog        = "agent_for_thermal_transport_v2.py",
        description = "MACE / UMA + phono3py thermal transport pipeline.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
parallel_mode summary
─────────────────────
  serial      all q at once — fast, no per-q resume
  serial_gp   one q at a time — crash-safe, per-q resume via checkpoint + disk
  omp         same as serial with OpenMP threading
  grid_points split q across Python workers / SLURM arrays

Monitor serial_gp progress
───────────────────────────
  python3 -c "
  import json; from pathlib import Path
  d = json.loads(Path('results/checkpoint.json').read_text())
  p = d.get('gp_progress', {})
  print(f\"{p.get('n_done','?')}/{p.get('n_total','?')} q-pts | \
{p.get('pct','?')}% | last={p.get('last_gp','?')} | {p.get('elapsed_s','?')}s\")
  "
""",
    )
    sub = root.add_subparsers(dest="command", required=True)

    # ── full ──────────────────────────────────────────────────────────────
    p_full = sub.add_parser("full",
        help="Full pipeline: relax -> displacements -> forces -> FC -> BTE -> kappa",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_full.add_argument("--structure",        required=True)
    p_full.add_argument("--no_relax",         action="store_true")
    p_full.add_argument("--relax_fmax",       type=float, default=0.001)
    p_full.add_argument("--relax_steps",      type=int,   default=500)
    p_full.add_argument("--relax_pressure",   type=float, default=0.0)
    p_full.add_argument("--supercell",        default="2 2 2")
    p_full.add_argument("--primitive_matrix", default="auto")
    p_full.add_argument("--amplitude",        type=float, default=0.03)
    p_full.add_argument("--cutoff_pair",      type=float, default=5.0)
    _add_calc_args(p_full)
    _add_bte_args(p_full)

    # ── bte ───────────────────────────────────────────────────────────────
    p_bte = sub.add_parser("bte",
        help="BTE only — FC2/FC3 must already be on disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_bte.add_argument("--structure",        default="POSCAR",
                       help="Unit cell POSCAR — required when phono3py_disp.yaml "
                            "is absent (e.g. FC2/FC3 from SCPH/hiphive workflow)")
    p_bte.add_argument("--supercell",        default="2 2 2")
    p_bte.add_argument("--primitive_matrix", default="auto")
    _add_calc_args(p_bte)
    _add_bte_args(p_bte)

    # ── collect ───────────────────────────────────────────────────────────
    p_col = sub.add_parser("collect",
        help="Collect kappa-m*-g*.hdf5 -> final kappa.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_col.add_argument("--structure",        default="POSCAR",
                       help="Unit cell POSCAR — required when phono3py_disp.yaml "
                            "is absent (e.g. FC2/FC3 from SCPH/hiphive workflow)")
    p_col.add_argument("--supercell",        default="2 2 2")
    p_col.add_argument("--primitive_matrix", default="auto")
    _add_calc_args(p_col)
    _add_bte_args(p_col)

    return root


def args_to_config(a: argparse.Namespace) -> Config:
    return Config(
        structure        = getattr(a, "structure",        "POSCAR"),
        out_dir          = a.out_dir,
        calc_type        = a.calc_type,
        mace_model       = a.mace_model,
        mace_head        = a.mace_head,
        mace_device      = a.mace_device,
        mace_dtype       = a.mace_dtype,
        uma_model        = a.uma_model,
        uma_task         = a.uma_task,
        uma_device       = a.uma_device,
        hf_token         = a.hf_token,
        dftd3            = a.dftd3,
        no_relax         = getattr(a, "no_relax",         False),
        relax_fmax       = getattr(a, "relax_fmax",       0.001),
        relax_steps      = getattr(a, "relax_steps",      500),
        relax_pressure   = getattr(a, "relax_pressure",   0.0),
        supercell        = getattr(a, "supercell",        "2 2 2"),
        primitive_matrix = getattr(a, "primitive_matrix", "auto"),
        symprec          = getattr(a, "symprec",          1e-5),
        amplitude        = getattr(a, "amplitude",        0.03),
        cutoff_pair      = getattr(a, "cutoff_pair",      5.0),
        mesh             = a.mesh,
        temperatures     = a.temperatures,
        solver           = a.solver,
        transport_type   = a.transport_type,
        isotope          = a.isotope,
        mass_variances   = a.mass_variances,
        parallel_mode    = a.parallel_mode,
        n_workers        = a.n_workers,
        gp               = a.gp,
        gp_batch_size    = a.gp_batch_size,
        resume           = a.resume,
    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    cfg    = args_to_config(args)

    if args.command == "full":
        run_full(cfg)
    elif args.command == "bte":
        run_bte(cfg)
    elif args.command == "collect":
        run_collect(cfg)
