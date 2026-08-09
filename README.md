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

Changing only `--transport_type` between `bte` reruns (same mesh/solver/
isotope/mass_variances/temperatures) reuses the already-computed phonon
lifetimes and only recomputes the kappa tensor — the expensive
phonon-phonon scattering step is not rerun. Changing anything else
(mesh, solver, isotope, mass_variances, temperatures) still triggers a
full recompute.

`--cutoff_frequency` (default `1e-2` THz -- matching the *native* `phono3py`
CLI's own default, set explicitly in its `cui/phono3py_script.py`, **not**
the bare `Phono3py()` Python class's internal default of `1e-4` that this
script would otherwise silently inherit) excludes modes at/below it from
every BTE/coherence sum, since the Gamma-acoustic modes (exactly 0 by
translational invariance) are always present in any mesh. A
finite-displacement FC2's Gamma-acoustic residual is usually symmetrized far
below either threshold; a `generate_scph_fc2_fc3_agent.py` FC2 (fit via
hiphive regression, not analytically projected) can leave a larger -- still
spurious, still meaningless -- residual that slips above `1e-4` but not
`1e-2`. That matters specifically for `--transport_type NJC23`: its
heat-capacity prefactor `(w_i+w_j)^2/4` does not vanish as `w_i -> 0`, so a
near-zero mode that isn't excluded can dominate `kappa_inter` with a number
that's really just numerical noise from the FC2 fit, not physics
(`IBDB19`'s prefactor `w_i*w_j` vanishes in the same limit, so it's far less
exposed to this). If NJC23 disagrees dramatically with IBDB19/SMM19 on an
SCPH-derived FC2 but the two agree closely on a finite-displacement FC2 of
the same material run through the native `phono3py` CLI, this mismatched
default is the first thing to suspect -- raise `--cutoff_frequency` further
(e.g. to `0.05`-`0.1`) if the residual is larger still.

Both the `bte` and `collect` subcommands also symmetrize a loaded FC2/FC3
(`ph3.symmetrize_fc2()`/`symmetrize_fc3()`) right after reading them from
disk, before anything else uses them. A finite-displacement FC3 already went
through `produce_fc3(symmetrize_fc3r=True)` before being saved, so this is
a cheap no-op there; a `generate_scph_fc2_fc3_agent.py` FC2/FC3 (fit via
hiphive regression) is not guaranteed to satisfy phono3py's translational/
permutation symmetry convention at all, and an unsymmetrized FC2 is the
*real* source of the Gamma-acoustic residual discussed above, not just
numerical noise -- on one real SCPH-fit MOF-274 FC2, symmetrizing dropped
the Gamma-acoustic frequencies from ~0.0016-0.0026 THz to ~1e-6 THz.
Pass `--skip_fc_symmetrize` to disable this (e.g. to reproduce old results,
or if a non-hiphive FC2/FC3 source is already known to be exactly symmetric
and the extra pass isn't worth its cost on a very large supercell).

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

`-tolerance`/`--symprec` (default `1e-3`) controls symmetry detection for
both hiphive's fit and the internal `Phonopy` object -- for floppy/relaxed
structures (e.g. MOFs) that need a looser tolerance to find their true
symmetry, pass the same value to `thermal_transport_agent.py bte --symprec`
and to `plot_scph_free_energy.py`/`plot_scph_bands.py -tolerance` too, or
the FC fit and the downstream BTE/diagnostics can silently disagree about
the structure's symmetry.

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

`-prim`/`-sdim`/`-pa`/`-tolerance` must match the `generate_scph_fc2_fc3_agent.py`
run being inspected. Outputs land in `output/T{T}/`.

## `plot_coherence_regime.py`

Diagnoses why `--transport_type` (SMM19/NJC23/IBDB19) gives substantially
different `kappa_inter`, from an existing `kappa-m*.hdf5` -- no new BTE run
needed.

```bash
python plot_coherence_regime.py --kappa_hdf5 results/kappa-m191919.hdf5 --temperature 300
```

Writes a linewidth-vs-frequency-gap scatter, the NJC23/IBDB19
prefactor-ratio curve, and a heatmap of where the two formulations disagree
most, with a phonon-DOS panel for context. `--annotate_threshold` (default
`1e-3` eV/K) labels the biggest-disagreement bins; negative disables it.
`--cutoff_frequency` (default `1e-2` THz) excludes near-zero modes, same as
`thermal_transport_agent.py`.

## `plot_cumulative_kappa_inter.py`

Shows which band-pair frequency gaps each `--transport_type`'s
`kappa_inter` comes from. Reuses cached gamma, so no new phonon-phonon
scattering is computed, but needs FC2/FC3 on disk (phono3py environment).

```bash
python plot_cumulative_kappa_inter.py \
    --out_dir results/ --structure POSCAR-unitcell \
    --supercell "2 2 2" --mesh "19 19 19" --temperature 300 --solver rta
```

Writes `kappa_inter_vs_dw_T{T}.png`: cumulative `kappa_inter` vs.
frequency-gap cutoff, one line per formulation. `--temperature` picks one
temperature out of the cached gamma file automatically.
