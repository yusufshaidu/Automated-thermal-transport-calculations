# Thermal Transport Toolkit

MACE/UMA machine-learned interatomic potentials + phono3py for lattice
thermal conductivity kappa(T). `thermal_transport_agent.py` runs the full
finite-displacement pipeline (relax -> forces -> FC2/FC3 -> BTE -> kappa(T)).
`generate_scph_fc2_fc3_agent.py` produces temperature-renormalized force
constants via a self-consistent phonon (SCPH) loop; its output feeds into
`thermal_transport_agent.py bte`. `plot_scph_free_energy.py`/
`plot_scph_bands.py` are convergence
diagnostics for an SCPH run. `phono3py_compat.py` is a shared
phono3py v3.x/v4.x compatibility helper and must stay alongside the other
scripts.

## Installation

```bash
conda create -n thermal python=3.10 && conda activate thermal
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or cu118 for GPU
pip install mace-torch phono3py phonopy h5py ase
pip install hiphive trainstation        # generate_scph_fc2_fc3_agent.py + plot_scph_*.py
pip install seekpath                    # plot_scph_bands.py only
pip install fairchem-core                # optional: --calc_type uma
pip install dftd3-python                 # optional: --dftd3 / --include_d3
```

## `thermal_transport_agent.py`

```bash
python thermal_transport_agent.py full \
    --structure    POSCAR              \
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

Subcommands: `full` (relax through kappa(T)), `bte` (BTE only, from
existing FC2/FC3), `collect` (assemble partial `bte` results from a
`--parallel_mode serial_gp`/`grid_points` run). `--resume` must be passed
explicitly to enable checkpoint reuse. Run with `--help` for the full flag
list.

Rerunning `bte` with only `--transport_type` changed reuses cached phonon
lifetimes (cheap); changing mesh/solver/isotope/mass_variances/temperatures
triggers a full recompute.

Notes:
- `--cutoff_frequency` (default `1e-2` THz) excludes near-zero modes from
  every BTE/coherence sum. SCPH-fit FC2s can leave a larger spurious
  Gamma-acoustic residual than finite-displacement FC2s; raise this
  (e.g. `0.05`-`0.1`) if `NJC23` disagrees sharply with `IBDB19`/`SMM19`.
- `bte`/`collect` symmetrize loaded FC2/FC3 by default
  (`--skip_fc_symmetrize` to disable) — important for hiphive-fit SCPH
  force constants, which aren't guaranteed symmetric on load.

## `generate_scph_fc2_fc3_agent.py`

```bash
python generate_scph_fc2_fc3_agent.py \
    -prim POSCAR-unitcell -sdim "2 2 2" \
    --model mace.model --head omat_pbe \
    -cutoffs "6.0 4.5" -temps "100 200 300" \
    -N 100 -niter 50 -o output/
```

Writes `output/T{T}/fc2.hdf5` and `fc3.hdf5`. Useful flags: `--resume`
(extend a run), `--select_best_iteration` (anchor the higher-order fit on
the lowest-free-energy stable iteration instead of the last one),
`--skip_scph --configs_dir <dir>` (refit with different `-cutoffs` reusing
saved configs), `--init_fc2 <fc2.hdf5>` (seed from a converged FC2 at
another temperature). Run with `--help` for the full list.

`-tolerance`/`--symprec` (default `1e-3`) should match across this script,
`thermal_transport_agent.py bte --symprec`, and `plot_scph_*.py
-tolerance` — a mismatch can make the FC fit and downstream BTE disagree
about the structure's symmetry.

## `plot_scph_free_energy.py` / `plot_scph_bands.py`

Read saved `fcp_scph/scph_T{T}_iter{i}.fcp` checkpoints, no new MLIP calls.

```bash
python plot_scph_free_energy.py -prim POSCAR-unitcell -sdim "2 2 2" \
    -o output/ -temps "100 200 300" --mesh "20 20 20"

python plot_scph_bands.py -prim POSCAR-unitcell -sdim "2 2 2" \
    -o output/ -temps "300" --npoints 101
```

`-prim`/`-sdim`/`-pa`/`-tolerance` must match the SCPH run being inspected.
Outputs land in `output/T{T}/`.

## `combine_scph_fc2_fc3.py`

Extracts the lowest-F(T), stable FC2 from `fcp_scph/scph_T{T}_iter*.fcp`
checkpoints (not the joint FC2..FC3 refit's `fc2.hdf5`) and can pair it with
an FC3 from a separate finite-displacement `thermal_transport_agent.py` run.

```bash
python combine_scph_fc2_fc3.py \
    -prim POSCAR-unitcell -sdim "2 2 2" -o output/ -temps "300" \
    --fc3_dir results_finite_disp/
```

Writes `output/T300/fc2_scph_best.hdf5` and, with `--fc3_dir`, a
`combined_scph_fc2_finite_fc3/` dir plus the matching `bte` command.
`-prim`/`-sdim`/`-pa`/`-tolerance` must match the SCPH run; make sure
`--fc3_dir`'s run used the same supercell/primitive matrix/symprec too —
only the atom count is checked automatically.

## `scan_relax_fmax_imaginary_modes.py`

For every `--fmax`/`--noise` combination, relaxes independently from
`--structure`, fits a finite-displacement FC2, and reports the minimum
non-acoustic frequency on `--mesh`. `--noise` rattles the structure
(`ase.Atoms.rattle`, seeded by `--noise_seed`) before each relaxation.

```bash
python scan_relax_fmax_imaginary_modes.py \
    --structure POSCAR --out_dir relax_scan/ \
    --fmax "0.05 0.01 0.005 0.001 0.0001" \
    --noise "0.0 0.02 0.05" \
    --supercell "2 2 2" --mesh "20 20 20" \
    --mace_model path/to/mace.model --mace_device cuda
```

Writes `fmax_{f}_noise_{n}/{POSCAR-relaxed, fc2.hdf5, result.json}` plus
`relax_scan_summary.{json,png}`; a negative `min_frequency_THz` means
imaginary modes. Reruns skip already-done combinations (`--overwrite` to
force). Calculator flags match `thermal_transport_agent.py`.

## `phonons_from_fc2.py`

Reports band structure (with an exact-q imaginary-mode check along
`--band_path`, not a mesh minimum), DOS, thermal properties, and MSD from a
saved `fc2.hdf5` or `FORCE_SETS`. Exits non-zero if any band-path q-point is
imaginary.

```bash
python phonons_from_fc2.py \
    --indir output/T300 --out_dir analysis/T300 \
    --supercell "2 2 2" --primitive_matrix auto \
    --band_path hex --dos --thermal --msd
```

Writes `bands.{yaml,png}` always; `total_dos.{dat,png}` with `--dos`;
`thermal_properties.{dat,yaml,png}` (+ `debye_fit.png` unless
`--debye_tmax 0`) with `--thermal`; `msd_vs_T.dat` with `--msd`.
`--band_path` is `hex` (G-M-K-G-A-L-H-A-L-M-K-H) or `ortho`
(G-P-Z-Q-G-F-P1-Q1-L-Z). `--mesh` (used only by `--dos`/`--thermal`) warns if
it can't land on `hex`'s K/H points (1/3, 1/3, \*) -- the band-path check is
the one that determines stability, not the mesh.

## `plot_coherence_regime.py`

Compares transport-type formulations (SMM19/NJC23/IBDB19) from an existing
`kappa-m*.hdf5` (no new BTE run).

```bash
python plot_coherence_regime.py --kappa_hdf5 results/kappa-m191919.hdf5 --temperature 300
```

## `plot_cumulative_kappa_inter.py`

Shows which band-pair frequency gaps each `--transport_type`'s
`kappa_inter` comes from. Reuses cached gamma; needs FC2/FC3 on disk.

```bash
python plot_cumulative_kappa_inter.py \
    --out_dir results/ --structure POSCAR-unitcell \
    --supercell "2 2 2" --mesh "19 19 19" --temperature 300 --solver rta
```
