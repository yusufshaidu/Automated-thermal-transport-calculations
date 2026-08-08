# Thermal Transport Toolkit

MACE/UMA machine-learned interatomic potentials + phono3py for lattice
thermal conductivity kappa(T). `thermal_transport_agent.py` runs the full
finite-displacement pipeline (relax -> forces -> FC2/FC3 -> BTE -> kappa(T)).
`generate_scph_fc2_fc3_agent.py` instead produces temperature-renormalized
force constants via a self-consistent phonon (SCPH) loop, for strongly
anharmonic materials -- its output feeds into `thermal_transport_agent.py
bte`. Two small utilities (`plot_scph_free_energy.py`, `plot_scph_bands.py`)
post-process SCPH checkpoints to check convergence. `phono3py_compat.py` is
a shared helper (phono3py v3.x/v4.x compatibility) and must stay in the same
directory as the other scripts.

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
explicitly on every invocation to enable checkpoint reuse. Run with `--help`
(or `full --help` / `bte --help`) for the full flag list, including
`--isotope`, `--parallel_mode`, and SLURM-array usage notes.

## `generate_scph_fc2_fc3_agent.py`

```bash
python generate_scph_fc2_fc3_agent.py \
    -prim POSCAR-unitcell -sdim "2 2 2" \
    --model mace.model --head omat_pbe \
    -cutoffs "6.0 4.5" -temps "100 200 300" \
    -N 100 -niter 50 -o output/
```

Writes `output/T{T}/fc2.hdf5` and `fc3.hdf5`, ready for
`thermal_transport_agent.py bte`. Useful flags: `--resume` (extend a run --
raise `-niter` and rerun with `--resume` to continue from the last
checkpoint instead of starting over), `--select_best_iteration` (anchor the
higher-order fit on the lowest-free-energy iteration instead of the last
one, since F(T) fluctuates iteration to iteration), `--skip_scph
--configs_dir <dir>` (refit with different `-cutoffs` reusing saved
configs, no new MLIP calls), `--init_fc2 <fc2.hdf5>` (seed from a converged
FC2 at another temperature). Run with `--help` for the full list.

## `plot_scph_free_energy.py` / `plot_scph_bands.py`

Convergence diagnostics for an SCPH run, reading its saved
`fcp_scph/scph_T{T}_iter{i}.fcp` checkpoints (no new MLIP evaluations):

```bash
# free energy F(T) and min phonon frequency vs. SCPH iteration
python plot_scph_free_energy.py -prim POSCAR-unitcell -sdim "2 2 2" \
    -o output/ -temps "100 200 300" --mesh "20 20 20"

# phonon band structure overlaid across SCPH iterations
python plot_scph_bands.py -prim POSCAR-unitcell -sdim "2 2 2" \
    -o output/ -temps "300" --npoints 101
```

`-prim`/`-sdim`/`-pa` must match the `generate_scph_fc2_fc3_agent.py` run
being inspected. Outputs land in `output/T{T}/`.
