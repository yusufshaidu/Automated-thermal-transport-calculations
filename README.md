# Thermal Transport Toolkit

MACE/UMA machine-learned interatomic potentials + phono3py, for computing
temperature-dependent lattice thermal conductivity kappa(T), including
self-consistent phonon (SCPH) renormalization for strongly anharmonic
materials.

The repository contains **two independent, complementary scripts**, two
post-processing utilities, and one shared helper module:

| File | Purpose |
|------|---------|
| [`thermal_transport_agent.py`](#thermal_transport_agentpy) | End-to-end pipeline: relax -> generate displaced supercells -> evaluate forces -> **finite-displacement** FC2/FC3 (phono3py) -> solve the phonon BTE -> kappa(T). Also runs BTE-only or collect-only against FC2/FC3 produced elsewhere (e.g. by the SCPH script). |
| [`generate_scph_fc2_fc3_agent.py`](#generate_scph_fc2_fc3_agentpy) | Produces **temperature-dependent, anharmonically renormalized** FC2 (plus a jointly-fit FC3/FC4…) via a self-consistent harmonic (SCPH) loop + [hiphive](https://hiphive.materialsmodeling.org/) fitting. Feeds its `fc2.hdf5`/`fc3.hdf5` into `thermal_transport_agent.py bte` for the actual kappa(T) solve. |
| [`plot_scph_free_energy.py`](#plot_scph_free_energypy) | Post-processes the `.fcp` checkpoints saved by `generate_scph_fc2_fc3_agent.py` and plots free energy F(T) and minimum phonon frequency vs. SCPH iteration — convergence/stability diagnostics that need no new MLIP evaluations. |
| [`plot_scph_bands.py`](#plot_scph_bandspy) | Overlays the phonon band structure from the same `.fcp` checkpoints along one auto-detected high-symmetry path, colored by SCPH iteration. |
| [`phono3py_compat.py`](#phono3py_compatpy) | Version-compatibility shim (phono3py v3.x vs v4.x). Imported by both pipeline scripts above — **must live in the same directory**. |

Use `thermal_transport_agent.py` on its own for standard 0 K (or
low-anharmonicity) finite-displacement force constants. Use
`generate_scph_fc2_fc3_agent.py` first, then `thermal_transport_agent.py bte`,
when force constants need to be renormalized at temperature (soft modes,
strongly anharmonic lattices, imaginary frequencies from a static FC2).

---

## Table of Contents

- [Installation](#installation)
- [`thermal_transport_agent.py`](#thermal_transport_agentpy)
  - [Quick start](#quick-start)
  - [Subcommands](#subcommands)
  - [Pipeline stages](#pipeline-stages-full)
  - [Calculator options](#calculator-options)
  - [BTE / transport options](#bte--transport-options)
  - [Parallel modes](#parallel-modes)
  - [Checkpointing, resume, and stale-cache invalidation](#checkpointing-resume-and-stale-cache-invalidation)
  - [Output files](#output-files)
  - [Full CLI reference](#full-cli-reference)
  - [SLURM examples](#slurm-examples)
- [`generate_scph_fc2_fc3_agent.py`](#generate_scph_fc2_fc3_agentpy)
  - [What the SCPH loop does](#what-the-scph-loop-does)
  - [CLI reference](#cli-reference)
  - [Restart / seed / skip modes](#restart--seed--skip-modes)
  - [Output files](#output-files-1)
  - [Usage examples](#usage-examples)
- [Connecting the two scripts](#connecting-the-two-scripts)
- [`plot_scph_free_energy.py`](#plot_scph_free_energypy)
- [`plot_scph_bands.py`](#plot_scph_bandspy)
- [`phono3py_compat.py`](#phono3py_compatpy)
- [Tips and troubleshooting](#tips-and-troubleshooting)

---

## Installation

```bash
conda create -n thermal python=3.10
conda activate thermal

# PyTorch (CPU or CUDA — pick one)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install torch --index-url https://download.pytorch.org/whl/cu118

# MACE
pip install mace-torch

# phono3py / phonopy
pip install phono3py phonopy h5py

# ASE
pip install ase

# hiphive + trainstation (needed only for generate_scph_fc2_fc3_agent.py
# and the plot_scph_*.py post-processing scripts)
pip install hiphive trainstation

# seekpath (needed only for plot_scph_bands.py — auto high-symmetry path)
pip install seekpath

# optional — UMA calculator backend (--calc_type uma)
pip install fairchem-core

# optional — Grimme D3(BJ) dispersion (--dftd3 / --include_d3)
pip install dftd3-python   # provides the `dftd3.ase.DFTD3` calculator
```

Both scripts import `phono3py_compat.py` at module load time — keep it in the
**same directory** as `thermal_transport_agent.py` and
`generate_scph_fc2_fc3_agent.py`.

---

## `thermal_transport_agent.py`

### Quick start

```bash
python thermal_transport_agent.py full \
    --structure    POSCAR              \
    --calc_type    mace                \
    --mace_model   path/to/mace.model  \
    --supercell    "2 2 2"             \
    --cutoff_pair  5.0                 \
    --mesh         "19 19 19"          \
    --temperatures "100 200 300 400 500" \
    --solver       rta                 \
    --transport_type SMM19             \
    --out_dir      results/            \
    --resume
```

`--resume` sets the dataclass config flag that gates checkpoint-based stage
skipping. Its CLI default is `False` even on a first run, so **pass
`--resume` explicitly** any time you might re-invoke the same `--out_dir`
later — see [Checkpointing, resume, and stale-cache
invalidation](#checkpointing-resume-and-stale-cache-invalidation).

### Subcommands

```
python thermal_transport_agent.py {full,bte,collect} ...
```

| Subcommand | Does | Use it for |
|---|---|---|
| `full` | relax -> displacements -> forces -> FC2/FC3 -> BTE -> kappa(T) -> plot | A new structure, starting from scratch. |
| `bte` | loads FC2/FC3 (from disk) -> BTE -> kappa(T) -> plot | Re-running the BTE with a different mesh, solver, isotope/coherence settings, or temperatures, without recomputing forces — including FC2/FC3 produced by `generate_scph_fc2_fc3_agent.py`. |
| `collect` | assembles `kappa-m*-g*.hdf5` partial files -> final kappa(T) -> plot | After many `bte --parallel_mode serial_gp/grid_points` jobs (e.g. a SLURM array) finish independently. |

`bte` and `collect` default `--structure` to `"POSCAR"` — only required when
`phono3py_disp.yaml` is absent from `--out_dir` (e.g. FC2/FC3 came from the
SCPH/hiphive workflow rather than `full`'s own displacement generation).

### Pipeline stages (`full`)

1. **Read structure** — reads `--structure` (POSCAR/CIF) with ASE, round-trips it through VASP format to `POSCAR-unitcell` for canonical formatting.
2. **Variable-cell relaxation** — `ase.filters.FrechetCellFilter` + `BFGS` (`fmax=--relax_fmax`, `steps=--relax_steps`, target pressure `--relax_pressure` GPa). Skip with `--no_relax`. Writes `POSCAR-relaxed`.
3. **Generate displaced supercells** — builds a `Phono3py` object (`--supercell`, `--primitive_matrix`, `--symprec`) and calls `generate_displacements(distance=--amplitude, is_plusminus=True, is_diagonal=True, cutoff_pair_distance=--cutoff_pair)`. Writes `phono3py_disp.yaml` and `supercells/disp-NNNNN.vasp`.
4. **Evaluate forces** — MACE/UMA (+/- D3) forces on the perfect supercell (residual subtracted from every displaced cell) and on each displaced supercell, cached per-displacement to `forces_cache/`; parallelized over `--n_workers` with `multiprocessing.Pool`. Writes `FORCES_FC3`.
5. **Force constants** — `ph3.produce_fc3(symmetrize_fc3r=True, is_compact_fc=False)` produces FC2 and FC3 together. Writes `fc2.hdf5`, `fc3.hdf5`.
6. **BTE solve** — dispatches to one of four code paths by `--parallel_mode` (see [Parallel modes](#parallel-modes)); resolves `--solver`/`--transport_type`/`--isotope`/`--mass_variances` into a single `run_thermal_conductivity()` call (see [BTE / transport options](#bte--transport-options)).
7. **Collect** — for `serial_gp`/`grid_points` runs, assembles per-q `kappa-m{tag}-g*.hdf5` files into the final `kappa-m{tag}.hdf5` and `kappa_summary.json`.
8. **Plot** — `kappa_vs_T.png`: kappa_xx/kappa_yy/kappa_zz/kappa_iso vs T always; a second panel with kappa_TOT/kappa_intra/kappa_inter (coherence decomposition) is added whenever `--transport_type` was set.

`bte` runs only stage 6 (+ plot), loading FC2/FC3 from disk. `collect` runs
only stage 7 (+ plot).

### Calculator options

| Flag | Default | Notes |
|---|---|---|
| `--calc_type` | `mace` | `mace` or `uma` |
| `--mace_model` | `""` | Path to a MACE model file (required for `--calc_type mace`) |
| `--mace_head` | `""` | MACE fine-tuning head, e.g. `omat_pbe` |
| `--mace_device` | `cuda` | `cpu` or `cuda` |
| `--mace_dtype` | `float64` | `float32` or `float64` |
| `--uma_model` | `uma-s-1p2` | `uma-s-1p2` or `uma-m-1p1` |
| `--uma_task` | `omc` | FAIRChem UMA task name, e.g. `omc`, `s2ef` |
| `--uma_device` | `cuda` | `cpu` or `cuda` |
| `--hf_token` | `$HF_TOKEN` | HuggingFace token, required for UMA |
| `--dftd3` | off | Adds a Grimme D3(BJ)/PBE dispersion correction on top of the base calculator |

### BTE / transport options

```
--solver          rta | lbte           (default: rta)
--transport_type   SMM19 | NJC23 | IBDB19   (default: omitted -> kappa_P only)
--isotope                                    (default: off)
--mass_variances  "g1 g2 ..."                (per-element, primitive-cell order)
```

| `--solver` | `--transport_type` | Method | Output |
|---|---|---|---|
| `rta` | *(omitted)* | Single-mode RTA | kappa_P only |
| `rta` | `SMM19` / `NJC23` / `IBDB19` | RTA + coherence | kappa_P + kappa_C |
| `lbte` | *(omitted)* | Iterative LBTE | kappa_P (full BTE) |
| `lbte` | `SMM19` / `NJC23` / `IBDB19` | LBTE + coherence | kappa_P + kappa_C (full BTE) |

- **`SMM19`** — Simoncelli–Marzari–Mauri (2019) Wigner transport equation; the original/default coherence formulation.
- **`NJC23`** — an alternative inter-band transport formulation.
- **`IBDB19`** — Isaeva–Barbalinardo–Donadio–Baroni (2019) formulation.

`--isotope` enables `is_isotope=True` (phonon-isotope scattering) in
phono3py. `--mass_variances` overrides the natural-abundance g-factors used
for isotope scattering — space-separated, one value per element in
primitive-cell order; leave empty to use phono3py's built-in values. It has
no effect without `--isotope`.

### Parallel modes

| `--parallel_mode` | Behaviour |
|---|---|
| `serial` (default) | All irreducible q-points in one `run_thermal_conductivity()` call. No per-q resume. |
| `omp` | Same code path as `serial`; phono3py threads internally over `OMP_NUM_THREADS` (set externally). Best for one node, many cores. |
| `serial_gp` | Loops over irreducible q-points **one at a time**, writing `kappa-m{tag}-g{N}.hdf5` and updating `checkpoint.json` after every point. Crash-safe — a kill loses at most one q-point; automatically calls `collect` at the end. |
| `grid_points` | Splits irreducible q-points across `--n_workers` (a `multiprocessing.Pool`) or, when `--gp` is set with `--n_workers 1`, computes just that subset of grid points as a single job (e.g. one SLURM array task) and leaves collection to a separate `collect` invocation. |

`--gp_batch_size N` groups N consecutive q-points per worker call in
`grid_points` mode to reduce per-call overhead.

Monitor `serial_gp` progress without touching the running job:

```bash
python3 -c "
import json; from pathlib import Path
d = json.loads(Path('results/checkpoint.json').read_text())
p = d.get('gp_progress', {})
print(f\"{p.get('n_done','?')}/{p.get('n_total','?')} q-pts | \
{p.get('pct','?')}% | last={p.get('last_gp','?')} | {p.get('elapsed_s','?')}s\")
"
```

### Checkpointing, resume, and stale-cache invalidation

Every stage writes a completion marker to `checkpoint.json`. Pass `--resume`
to skip stages already marked done and to reload cached per-displacement
forces (`forces_cache/`) — without it, stages are always recomputed even if
`checkpoint.json` exists.

`serial_gp` and `grid_points` additionally reconcile `checkpoint.json`
against whatever `kappa-m{tag}-g*.hdf5` files actually exist on disk (disk is
the source of truth), so a job can be restarted after a crash without
recomputing already-finished q-points.

**Stale-cache protection**: on every `bte`/`collect` run, the script
fingerprints the physics-relevant settings (`mesh`, `solver`,
`transport_type`, `isotope`, `mass_variances`, `temperatures`) and compares
them against what's recorded in `checkpoint.json`. If you change any of
these while pointing `--out_dir` at a directory with prior kappa results, the
old `kappa-m*-g*.hdf5` / `kappa-m*.hdf5` / `kappa_summary.json` /
`kappa_vs_T.png` files are moved aside into `_stale_kappa_<timestamp>/`, the
BTE is recomputed from scratch, and a `[CACHE INVALIDATED]` warning is
logged. This prevents silently reusing numbers from a previous
`--solver`/`--transport_type`/`--isotope` setting.

### Output files

```
results/
├── pipeline.log               # log of all stages
├── config.json                # saved Config (full pipeline only)
├── checkpoint.json            # stage completion, gp progress, kappa fingerprint
│
├── POSCAR-unitcell            # canonical input structure
├── POSCAR-relaxed             # relaxed (or unrelaxed copy if --no_relax)
├── relax_BFGS.log
│
├── phono3py_disp.yaml         # displacement dataset
├── supercells/disp-00001.vasp ...
│
├── forces_cache/forces_perfect.npy
├── forces_cache/forces_00000.npy ...
├── FORCES_FC3
│
├── fc2.hdf5                   # 2nd-order force constants
├── fc3.hdf5                   # 3rd-order force constants
│
├── kappa-m{tag}-g{N}.hdf5     # per-q partial results (serial_gp / grid_points)
├── kappa-m{tag}.hdf5          # final assembled kappa (phono3py native format)
│
├── kappa_summary.json         # human-readable kappa(T) results
├── kappa_vs_T.png             # kappa vs T plot
└── _stale_kappa_<timestamp>/  # old results moved aside on a settings change
```

`mesh_tag` is the mesh numbers concatenated, e.g. mesh `"19 19 19"` -> tag
`"191919"`.

`kappa_summary.json`:

```json
{
  "mesh": [19, 19, 19],
  "solver": "RTA + SMM19 (kappa_P + kappa_C) + isotope",
  "transport_type": "SMM19",
  "temperatures": {
    "300": {
      "kappa_TOT":       [1.23, 1.23, 0.85, 0.0, 0.0, 0.0],
      "kappa_intra":     [1.10, 1.10, 0.76, 0.0, 0.0, 0.0],
      "kappa_inter":     [0.13, 0.13, 0.09, 0.0, 0.0, 0.0],
      "kappa_iso":       1.10,
      "kappa_intra_iso": 0.99,
      "kappa_inter_iso": 0.12
    }
  }
}
```

Tensor entries are ordered `[xx, yy, zz, yz, xz, xy]` in W m^-1 K^-1.
`kappa_intra`/`kappa_inter` (and their `_iso` isotropic averages) are only
present when a coherence `--transport_type` was used.

### Full CLI reference

Shared across `full`/`bte`/`collect`:

| Parameter | Default | Description |
|---|---|---|
| `--mesh` | `"11 11 11"` | BTE q-point mesh |
| `--temperatures` | `"300"` | Space-separated temperatures (K) |
| `--solver` | `rta` | `rta` or `lbte` |
| `--transport_type` | *(none)* | `SMM19`, `NJC23`, or `IBDB19` — adds kappa_C |
| `--isotope` | off | Phonon-isotope scattering |
| `--mass_variances` | `""` | Per-element isotope g-factors (needs `--isotope`) |
| `--symprec` | `1e-5` | Symmetry precision (Å) |
| `--out_dir` | `results` | Output directory |
| `--resume` | off | Enable checkpoint/cache reuse |
| `--parallel_mode` | `serial` | `serial`, `omp`, `serial_gp`, `grid_points` |
| `--gp` | `""` | Grid points for this job (space-separated); empty = all |
| `--gp_batch_size` | `1` | Grid points per worker call |
| `--n_workers` | `1` | Worker processes (forces + `grid_points` BTE) |
| `--calc_type` | `mace` | `mace` or `uma` |
| `--mace_model` / `--mace_head` / `--mace_device` / `--mace_dtype` | see above | MACE options |
| `--uma_model` / `--uma_task` / `--uma_device` / `--hf_token` | see above | UMA options |
| `--dftd3` | off | Add D3(BJ) dispersion |

`full`-only:

| Parameter | Default | Description |
|---|---|---|
| `--structure` | required | Input POSCAR or CIF |
| `--no_relax` | off | Skip relaxation |
| `--relax_fmax` | `0.001` | Force convergence threshold (eV/Å) |
| `--relax_steps` | `500` | Max optimiser steps |
| `--relax_pressure` | `0.0` | Target pressure (GPa) |
| `--supercell` | `"2 2 2"` | Supercell matrix (3 diagonal or 9 full) |
| `--primitive_matrix` | `"auto"` | Primitive matrix or `"auto"` |
| `--amplitude` | `0.03` | Displacement distance (Å) |
| `--cutoff_pair` | `5.0` | 3rd-order pair cutoff (Å); <=0 = no cutoff |

`bte`/`collect`-only:

| Parameter | Default | Description |
|---|---|---|
| `--structure` | `"POSCAR"` | Only required if `phono3py_disp.yaml` is absent from `--out_dir` |
| `--supercell` | `"2 2 2"` | Must match the FC2/FC3 on disk |
| `--primitive_matrix` | `"auto"` | Must match the FC2/FC3 on disk |

### SLURM examples

**Full pipeline, OpenMP BTE:**

```bash
#!/bin/bash
#SBATCH --job-name=thermal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=16

python thermal_transport_agent.py full \
    --structure   POSCAR             \
    --mace_model  mace.model         \
    --supercell   "2 2 2"            \
    --cutoff_pair 5.0                \
    --mesh        "19 19 19"         \
    --temperatures "100 200 300 400 500" \
    --solver      rta --transport_type SMM19 \
    --parallel_mode omp              \
    --out_dir     results/           \
    --resume
```

**Forces on one node, BTE as a q-point array:**

```bash
# Step 1 — forces + FC2/FC3
python thermal_transport_agent.py full \
    --structure POSCAR --mace_model mace.model \
    --supercell "2 2 2" --cutoff_pair 5.0 \
    --mesh "19 19 19" --temperatures "300" \
    --solver rta --transport_type SMM19 \
    --n_workers 8 \
    --out_dir results/ --resume
```

```bash
#!/bin/bash
#SBATCH --job-name=bte_gp
#SBATCH --array=0-200        # upper bound = n_irreducible_q - 1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

GP=$SLURM_ARRAY_TASK_ID

python thermal_transport_agent.py bte \
    --out_dir      results/          \
    --mesh         "19 19 19"        \
    --temperatures "100 200 300 400 500" \
    --solver       rta --transport_type SMM19 \
    --parallel_mode grid_points      \
    --gp           "$GP"
```

```bash
#!/bin/bash
#SBATCH --job-name=collect
#SBATCH --dependency=afterok:<array_job_id>

python thermal_transport_agent.py collect \
    --out_dir      results/          \
    --mesh         "19 19 19"        \
    --temperatures "100 200 300 400 500" \
    --solver       rta --transport_type SMM19
```

---

## `generate_scph_fc2_fc3_agent.py`

Generates **temperature-dependent** force constants: a self-consistent
harmonic loop refines FC2 at each requested temperature by iteratively
generating thermally-displaced structures, evaluating MLIP forces on them,
and refitting — then jointly fits FC3 (and higher orders, if requested) to
the accumulated configurations with [hiphive](https://hiphive.materialsmodeling.org/).
Output is exported to phono3py-format `fc2.hdf5`/`fc3.hdf5`, ready for
`thermal_transport_agent.py bte`.

### What the SCPH loop does

For each temperature `T` in `--temperatures`, `run_scph_and_collect()`:

1. **Initializes** 2nd-order parameters either from small-amplitude
   (0.001 Å) rattled structures fit with a hiphive `Optimizer`, or — if
   `--init_fc2` is given — by algebraically projecting an existing
   `fc2.hdf5` onto the cluster-space basis (no forces needed).
2. Repeats for `i = 0 .. --n_iterations - 1`:
   - Builds the FC2 array from the current parameters and symmetrizes it.
   - Generates `--n_structures` thermally-displaced structures at `T`
     (hiphive's phonon rattler for classical statistics, or phonopy's
     random-displacement generator for classical **or** quantum
     Bose-Einstein statistics with `--qm_statistics`).
   - Evaluates MLIP forces on each (the expensive step) and saves the
     tagged configs to `configs/config_T{T:.0f}_N{n_structures}_iter{i}.extxyz`.
   - Refits 2nd-order parameters by least squares, then mixes with the
     previous iteration: `params_new = alpha * params_fit + (1 - alpha) * params_old`
     (`--alpha`, default `0.2` — simple linear/Picard mixing).
   - Logs RMSE, relative parameter change `|delta_params| / |params_old|`, and
     max displacement. **There is no automatic convergence check** — the
     loop always runs the full `--n_iterations`; use the printed relative
     change to judge convergence by eye.
   - Every `--ckpt` iterations, checkpoints parameters to
     `fcp_scph/scph_T{T:.0f}_iter{i}.fcp`.
3. After the loop, collects configs from the **last `--n_collect`
   iterations only** (later iterations sample the thermally-converged
   configuration space, giving a better-conditioned dataset without extra
   force evaluations) and jointly fits a multi-order
   `ForceConstantPotential` (FC2 .. FC`{len(cutoffs)+1}`) to them via
   least squares (`--train_size` controls the training fraction).
4. Exports the fit to phono3py-format `fc2.hdf5`/`fc3.hdf5` (generic HDF5
   for order > 3), and prints follow-up commands: a
   `thermal_transport_agent.py bte` invocation and a version-appropriate
   native `phono3py`/`phono3py-init` CLI command.

Higher orders (FC3+) are **not** part of the SCPH self-consistency loop —
only the 2nd-order cutoff (`cutoffs[0]`) is iterated; FC3+ is fit once,
jointly, at the end.

### CLI reference

There are no subcommands — a single flat argument list.

**Structure**

| Flag | Default | Description |
|---|---|---|
| `-prim` / `--prim_file` | required | Primitive structure file |
| `-sdim` / `--sdim` | `"2 2 2"` | Supercell dimensions |
| `-pa` / `--primitive_matrix` | `"auto"` | Primitive matrix keyword or 9 numbers |
| `-tolerance` / `--symprec` | `1e-3` | Symmetry precision |
| `-o` / `--outdir` | required | Output directory |

**Calculator**

| Flag | Default | Description |
|---|---|---|
| `--calc_type` | `mace` | `mace` or `uma` |
| `--model` | `""` | MACE model path (ignored with `--skip_scph`) |
| `--head` | `omat_pbe` | MACE head / UMA task name (ignored with `--skip_scph`) |
| `--device` | `cuda` | `cpu` or `cuda` (ignored with `--skip_scph`) |
| `--hf_token` | `$HF_TOKEN` | UMA HuggingFace token |
| `--include_d3` | off | Add Grimme D3(BJ) dispersion |

**SCPH loop**

| Flag | Default | Description |
|---|---|---|
| `-temps` / `--temperatures` | required | Space-separated temperatures (K) |
| `-N` / `--n_structures` | `100` | Displaced structures per iteration (ignored with `--skip_scph`) |
| `-niter` / `--n_iterations` | `50` | SCPH iterations (ignored with `--skip_scph`) |
| `-alpha` / `--alpha` | `0.2` | Mixing/momentum factor (ignored with `--skip_scph`) |
| `-cutoffs` / `--cutoffs` | required | Space-separated cluster cutoffs (Å); N cutoffs fit orders 2..N+1 |
| `--qm_statistics` | off | Quantum Bose-Einstein vs. classical Maxwell-Boltzmann displacement amplitude (ignored with `--skip_scph`) |
| `--imag_freq_factor` | `1.0` | — |
| `-ckpt` / `--ckpt` | `2` | Checkpoint FCP every N iterations (ignored with `--skip_scph`) |
| `--resume` | off | Auto-resume: reuse the highest `fcp_scph/scph_T{T}_iter*.fcp` checkpoint under `--outdir` per temperature and continue from there. Raise `--n_iterations` beyond the previous run's to extend it. Overridden by explicit `-nstart`/`-init`; ignored with `--skip_scph`/`--fc2_only`. |
| `-nstart` / `--nstart` | `0` | Manual restart: iteration to resume at. Takes precedence over `--resume` |
| `-init` / `--initial_parameter_file` | *(none)* | Manual restart: path to a saved `fcp_scph/scph_T{T}_iter{i}.fcp` checkpoint to resume from |
| `--init_fc2` | *(none)* | Seed the harmonic model from an existing `fc2.hdf5` |
| `--fc2_only` | off | Requires `--init_fc2`; single displacement batch, no SCPH loop |

**Higher-order fitting**

| Flag | Default | Description |
|---|---|---|
| `--n_collect` | `10` | Configs from the last N iterations used for the higher-order fit |
| `--train_size` | `1.0` | Fraction of collected data used for training |
| `--select_best_iteration` | off | Anchor the `--n_collect` window on the lowest-F(T) iteration instead of the last one (see below) |
| `--fe_mesh` | `"10 10 10"` | q-point mesh for `--select_best_iteration`'s per-iteration free energy calc |

**Skip-SCPH**

| Flag | Default | Description |
|---|---|---|
| `--skip_scph` | off | Skip the SCPH loop, reuse previously saved configs |
| `--configs_dir` | *(none)* | Directory with `config_T{T}_N*_iter*.extxyz` files |
| `--configs` | *(none)* | Comma-separated explicit paths/globs; overrides `--configs_dir` |

Validation: `--n_collect` is clamped (with a warning) to `--n_iterations` if
larger; `--fc2_only` requires `--init_fc2` and is mutually exclusive with
`--skip_scph`.

### Selecting the best iteration for the higher-order fit

F(T) is not guaranteed to decrease monotonically between SCPH iterations --
each iteration's parameters are fit to a different random batch of thermal
displacements, so the free energy naturally fluctuates with that sampling
noise. By default, `--n_collect` always anchors its window on the *last*
iteration, which may not be the best one.

`--select_best_iteration` tracks F(T) at every iteration (a phonopy mesh sum
on `--fe_mesh` -- cheap, no extra MLIP evaluations, since it reuses the FC2
already built that iteration) and instead anchors the `--n_collect` window
on whichever iteration had the *lowest* F(T), reusing that iteration's
already-saved configs (again, no extra MLIP evaluations). Combined with
`--resume`, F(T) for pre-restart iterations is recomputed from their
checkpoints so the best-iteration search still spans the whole run -- except
iteration 0's, which has no corresponding checkpoint and can't be recovered
after a restart.

### Restart / seed / skip modes

- **Continue an interrupted or too-short run**: add `--resume` and (if you
  want more iterations than originally requested) raise `--n_iterations`.
  For each temperature, it finds the highest-numbered
  `fcp_scph/scph_T{T}_iter{i}.fcp` checkpoint already on disk, projects its
  FC2 back onto the ClusterSpace to seed the parameters, and continues the
  loop at iteration `i + 1` — the completed iterations are not redone, and
  their already-saved `.extxyz` configs are folded back into `--n_collect`'s
  window so the higher-order fit still sees the full run. If
  `--n_iterations` isn't raised past `i + 1`, it prints a warning and
  no-ops (nothing left to run). If no checkpoint exists yet for that
  temperature, it just starts from scratch.
- **Restart at a specific iteration with a specific checkpoint** (manual
  equivalent of `--resume`, e.g. to roll back a few iterations): `-nstart N
  -init <fcp_scph/scph_T{T}_iter{N-1}.fcp>` resumes the loop at iteration
  `N`, seeding parameters from that checkpoint's FC2. Explicit `-nstart`/
  `-init` always take precedence over `--resume`.
- **Seed from a converged FC2 at another temperature/setting**:
  `--init_fc2 previous_run/T300/fc2.hdf5` initializes the loop without
  rattled-structure fitting; add `--fc2_only` to skip iterative refinement
  entirely and just generate one batch of thermal displacements from that
  FC2 for the joint higher-order fit.
- **Refit with different cutoffs without new force evaluations**:
  `--skip_scph --configs_dir <dir>` (or `--configs <glob1,glob2,...>`)
  reuses previously saved `.extxyz` configs — useful for adding a 4th-order
  term or changing `--cutoffs` without re-running MLIP forces.

### Output files

```
<outdir>/
└── T{T:.0f}/
    ├── configs/
    │   └── config_T{T:.0f}_N{n_structures}_iter{i}.extxyz
    ├── fcp_scph/
    │   ├── scph_T{T:.0f}_iter{i}.fcp      # checkpoint every --ckpt iterations
    │   └── scph_T{T:.0f}_final.fcp        # final harmonic-only FCP
    ├── fcp_order2to{max_order}.fcp        # joint FC2..FC{max_order} potential
    ├── fc2.hdf5                            # phono3py-format FC2
    ├── fc3.hdf5                            # phono3py-format FC3 (if max_order >= 3)
    └── fc{order}.hdf5                      # generic HDF5, order > 3
```

No JSON summaries or plots are produced — only console/log output (cluster
space orbit info, per-iteration RMSE/convergence diagnostics, and the
recommended follow-up commands).

### Usage examples

**From scratch:**

```bash
python generate_scph_fc2_fc3_agent.py \
    -prim POSCAR-unitcell \
    -sdim "2 2 2" \
    --model mace.model --head omat_pbe \
    -cutoffs "6.0 4.5" \
    -temps "100 200 300" \
    -N 100 -niter 50 \
    --n_collect 10 \
    -o output/
```

Add `--qm_statistics` for quantum (Bose-Einstein) displacement amplitudes
instead of classical.

**Refit with different cutoffs, no new MLIP calls:**

```bash
python generate_scph_fc2_fc3_agent.py \
    -prim POSCAR-unitcell \
    -sdim "2 2 2" \
    -cutoffs "6.0 3.5 3.5" \
    -temps "100 200 300" \
    -o output/ \
    --skip_scph --configs_dir output_prev/T300/configs
```

**Seed from an existing FC2, still refining:**

```bash
python generate_scph_fc2_fc3_agent.py \
    -prim POSCAR-unitcell -sdim "2 2 2" \
    --model mace.model --head omat_pbe \
    -cutoffs "6.0 4.5" -temps "300" -N 100 -niter 50 \
    -o output/ \
    --init_fc2 previous_run/T300/fc2.hdf5
```

**Seed from an existing FC2, skip refinement (`--fc2_only`):**

```bash
python generate_scph_fc2_fc3_agent.py \
    -prim POSCAR-unitcell -sdim "2 2 2" \
    --model mace.model --head omat_pbe \
    -cutoffs "6.0 4.5" -temps "300" -N 100 \
    -o output/ \
    --init_fc2 previous_run/T300/fc2.hdf5 --fc2_only
```

---

## Connecting the two scripts

1. Run `generate_scph_fc2_fc3_agent.py` to get temperature-renormalized
   `T{T}/fc2.hdf5` and `T{T}/fc3.hdf5`.
2. Point `thermal_transport_agent.py bte` at that directory to solve the
   BTE and get kappa(T) at that temperature. Since the SCPH output directory
   has no `phono3py_disp.yaml`, pass `--structure` explicitly (the same
   primitive structure file) along with matching `--supercell` /
   `--primitive_matrix`:

```bash
python thermal_transport_agent.py bte \
    --structure    POSCAR-unitcell   \
    --supercell    "2 2 2"           \
    --out_dir      output/T300/      \
    --mesh         "11 11 11"        \
    --temperatures "300"             \
    --solver       rta --transport_type SMM19 \
    --resume
```

(This is exactly the command `generate_scph_fc2_fc3_agent.py` prints at the
end of each temperature's run — adjust `--out_dir`/paths to match your
actual layout.)

Before spending a BTE run on a given temperature, it's worth checking that
the SCPH loop actually converged at that temperature — see
[`plot_scph_free_energy.py`](#plot_scph_free_energypy) below.

---

## `plot_scph_free_energy.py`

The SCPH loop has no automatic convergence check — it just runs
`--n_iterations` and prints the relative parameter change for you to judge
by eye. This script computes two more physical diagnostics instead: for each
checkpointed `fcp_scph/scph_T{T}_iter{i}.fcp` (and `..._final.fcp`), it
rebuilds the FC2, symmetrizes it, and computes on a phonopy q-point mesh (a)
the harmonic free energy F(T) and (b) the minimum phonon frequency anywhere
on the mesh, *excluding* the 3 acoustic bands at Gamma (which are trivially
~0 for any structure, stable or not, by translational invariance — including
them would make the minimum always read ~0 regardless of real instabilities
elsewhere in the Brillouin zone). Plotting both vs. iteration shows whether
the loop actually plateaued, and whether any mode is still (or newly)
dynamically unstable (negative/imaginary frequency). No new MLIP evaluations
are needed.

```bash
python plot_scph_free_energy.py \
    -prim POSCAR-unitcell -sdim "2 2 2" -o output/ \
    -temps "100 200 300" --mesh "20 20 20"
```

`-prim`/`-sdim`/`-pa` must match the `generate_scph_fc2_fc3_agent.py` run
that produced the checkpoints, since they're used to rebuild the same
supercell the FC2 must match. Each FC2 read from a `.fcp` is checked against
the expected `(N, N, 3, 3)` shape before use — note this check is on the FC2
array itself, not on `fcp.primitive_structure` (hiphive's own
symmetry-reduced primitive, e.g. 1 atom for a cubic lattice), which is not
comparable to the supercell size and would give a false mismatch.

Outputs per temperature: `<outdir>/T{T:.0f}/scph_convergence.json` (raw
iteration -> free energy / min-frequency values) and `scph_convergence.png`
(two stacked panels vs. iteration, final value marked as a dashed line).
`--classical` switches from quantum (Bose-Einstein, default) to classical
(Boltzmann) statistics for the free energy. A flat F(T) curve means the SCPH
loop converged; a still-drifting one means raise `--n_iterations` or lower
`--alpha`. A minimum frequency that stays negative means the structure is
still dynamically unstable at that temperature even after SCPH renormalization.

---

## `plot_scph_bands.py`

Overlays the phonon band structure from every checkpointed `.fcp` (same
files as above) along a single high-symmetry path, colored by SCPH
iteration, with the final fit drawn in bold black. This shows directly how
each branch moves (softening, hardening, avoided crossings) as the SCPH loop
converges — a more detailed view than the single min-frequency number from
`plot_scph_free_energy.py`.

```bash
python plot_scph_bands.py \
    -prim POSCAR-unitcell -sdim "2 2 2" -o output/ \
    -temps "300" --npoints 101
```

The high-symmetry path is auto-detected once (via phonopy + `seekpath`, from
the primitive structure — `pip install seekpath`) and reused for every
iteration, so all curves share exactly the same q-points. `--stride N` plots
only every Nth checkpointed iteration (the final fit is always included) to
keep the plot legible when there are many checkpoints. Output:
`<outdir>/T{T:.0f}/bands_vs_scph_iteration.png`.

---

## `phono3py_compat.py`

A small compatibility shim so both scripts work across phono3py v3.x and
v4.x, which changed several APIs (CLI split into `phono3py-init` +
`phono3py`; compact-FC became the v4 default vs. full FC in v3; default
`primitive_matrix` changed from identity to `"auto"`; Rust backend by
default in v4; the grid/tetrahedron-method/kaccum modules moved from
`phono3py` to `phonopy`).

Exported functions:

- `get_phono3py_version()` — installed phono3py version as a tuple of ints.
- `is_v4_or_later()` — `True` if major version >= 4.
- `print_version_banner(log=None)` — logs/prints the detected version and
  compatibility mode.
- `get_thermal_conductivity_RTA_compat(interaction, **kwargs)` — imports
  `get_thermal_conductivity_RTA` from whichever module path it lives at in
  the installed version.
- `get_ir_grid_points_compat(bz_grid)` — imports `get_ir_grid_points` from
  `phonopy.phonon.grid` (v4+) or `phono3py.phonon.grid` (v3), used to get
  irreducible q-points in phono3py's BZGrid numbering.
- `recommend_bte_cli(mesh, prim_file, dim, pa, tmin, tmax, tstep, full_fc)` —
  returns a ready-to-run, version-appropriate native `phono3py`/
  `phono3py-init` CLI command for computing kappa(T) directly from
  `fc2.hdf5`/`fc3.hdf5`, outside the Python pipeline.

---

## Tips and troubleshooting

**MACE on GPU — always use `--n_workers 1`.** GPU MACE cannot be shared
across `multiprocessing.Pool` workers (CUDA context conflicts). For BTE
parallelism on GPU-relax/force nodes, use `--parallel_mode omp` with
`OMP_NUM_THREADS` set externally instead of `grid_points`.

**`--resume` must be passed explicitly, every run.** Its CLI default is
`False`; omitting it recomputes every stage even if `checkpoint.json`
already records completed work.

**Changing BTE settings on an existing `--out_dir`.** This is safe — a
solver/transport_type/isotope/mass_variances/temperatures change triggers
automatic stale-cache invalidation (old kappa files moved to
`_stale_kappa_<timestamp>/`). Purely computational settings
(`parallel_mode`, `n_workers`) don't trigger it.

**Large residual forces on the relaxed structure.** Tighten relaxation:

```bash
--relax_fmax 0.0001 --relax_steps 1000
```

**How many irreducible q-points will `grid_points`/`serial_gp` need?**
Check the log after the BTE stage initializes — it logs the count. For a
19x19x19 mesh in a typical low-symmetry structure, expect on the order of
hundreds.

**Cutoff-pair / cutoffs convergence.** Sweep `--cutoff_pair` (finite
displacement) or the last entry of `-cutoffs` (SCPH FC3) and check kappa(T)
convergence before trusting a production run.

**Mesh convergence.** Always check kappa(T) vs. mesh density with `bte`
reruns (cheap — force constants aren't recomputed):

```bash
for mesh in "9 9 9" "11 11 11" "15 15 15" "19 19 19"; do
    python thermal_transport_agent.py bte \
        --out_dir results/ --mesh "$mesh" \
        --temperatures "300" --solver rta --transport_type SMM19
done
```

**Imaginary phonon frequencies from a static (0 K) FC2.** This is the
signal to switch from `thermal_transport_agent.py full` to
`generate_scph_fc2_fc3_agent.py` — SCPH renormalizes FC2 at the target
temperature, which typically removes thermally-stabilized soft modes.

**D3 dispersion.** `--dftd3` (in `thermal_transport_agent.py`) /
`--include_d3` (in `generate_scph_fc2_fc3_agent.py`) both import
`dftd3.ase.DFTD3` — install the `dftd3-python` package, not `torch-dftd`,
to provide this module.
