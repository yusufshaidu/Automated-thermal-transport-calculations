"""Compatibility helpers for phono3py v3.x vs v4.x API/CLI differences."""

from __future__ import annotations

import logging


# =============================================================================
# Version detection
# =============================================================================

def get_phono3py_version() -> tuple[int, ...]:
    """Return phono3py version as a tuple of ints, e.g. (4, 3, 1)."""
    import phono3py
    raw = phono3py.__version__
    parts = []
    for p in raw.split("."):
        digits = "".join(c for c in p if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) if parts else (0,)


def is_v4_or_later() -> bool:
    """True if the installed phono3py is version 4.0.0 or later."""
    return get_phono3py_version()[0] >= 4


def print_version_banner(log: logging.Logger | None = None) -> str:
    """Print/log the detected phono3py version and its key behavioural mode."""
    import phono3py
    version = phono3py.__version__
    mode = (
        "v4+  (CLI split: phono3py-init/phono3py, compact-FC default, "
        "Rust backend)"
        if is_v4_or_later() else
        "v3.x (single phono3py CLI, full-FC default, C/legacy backend)"
    )
    msg = f"  phono3py version: {version}   [{mode}]"
    if log is not None:
        log.info(msg)
    else:
        print(msg)
    return version


# =============================================================================
# get_thermal_conductivity_RTA — multi-path import
# =============================================================================

def get_thermal_conductivity_RTA_compat(interaction, **kwargs):
    """Call get_thermal_conductivity_RTA regardless of which module path it lives at in the installed phono3py version."""
    last_err = None
    candidates = [
        "phono3py.conductivity.rta_init",
        "phono3py.conductivity.rta",
        "phono3py.phonon3.conductivity_RTA",
    ]
    for mod_path in candidates:
        try:
            mod = __import__(mod_path, fromlist=["get_thermal_conductivity_RTA"])
            func = getattr(mod, "get_thermal_conductivity_RTA")
            return func(interaction, **kwargs)
        except (ImportError, AttributeError) as e:
            last_err = e
            continue

    version = ".".join(map(str, get_phono3py_version()))
    raise ImportError(
        f"Could not find get_thermal_conductivity_RTA in any known module "
        f"path (tried: {candidates}).\n"
        f"Installed phono3py version: {version}\n"
        f"Last error: {last_err}\n"
        f"This usually means the thermal-conductivity module layout changed "
        f"in this phono3py release. Run:\n"
        f"  python -c \"import phono3py.conductivity as c; print(dir(c))\"\n"
        f"to find the new location and update phono3py_compat.py."
    )


# =============================================================================
# get_ir_grid_points — multi-path import (grid module moved phono3py -> phonopy in v4)
# =============================================================================

def get_ir_grid_points_compat(bz_grid):
    """Return irreducible grid point indices, regardless of which module get_ir_grid_points() lives in for the installed phono3py/phonopy version."""
    last_err = None
    candidates = (
        ["phonopy.phonon.grid", "phono3py.phonon.grid"]
        if is_v4_or_later() else
        ["phono3py.phonon.grid", "phonopy.phonon.grid"]
    )
    for mod_path in candidates:
        try:
            mod = __import__(mod_path, fromlist=["get_ir_grid_points"])
            func = getattr(mod, "get_ir_grid_points")
            return func(bz_grid)
        except (ImportError, AttributeError) as e:
            last_err = e
            continue

    version = ".".join(map(str, get_phono3py_version()))
    raise ImportError(
        f"Could not find get_ir_grid_points in any known module path "
        f"(tried: {candidates}).\n"
        f"Installed phono3py version: {version}\n"
        f"Last error: {last_err}\n"
        f"Run:\n"
        f"  python -c \"from phonopy.phonon.grid import BZGrid; "
        f"print([a for a in dir(BZGrid) if not a.startswith('_')])\"\n"
        f"to inspect the installed BZGrid API and update phono3py_compat.py."
    )


# =============================================================================
# CLI command recommendations (version-aware)
# =============================================================================

def recommend_bte_cli(
    mesh:      str,
    prim_file: str,
    dim:       str,
    pa:        str,
    tmin:      float = 25,
    tmax:      float = 400,
    tstep:     float = 25,
    full_fc:   bool   = True,
) -> str:
    """Return a phono3py CLI command string appropriate for the detected phono3py version to run thermal conductivity from existing fc2.hdf5 / fc3.hdf5."""
    fc_flag = " --full-fc" if (full_fc and is_v4_or_later()) else ""

    if is_v4_or_later():
        return f"""
  # phono3py >= 4: CLI is split into phono3py-init (setup) and phono3py (run).
  # Step 1 — generate phono3py_disp.yaml (no actual displacements needed
  #          since fc2.hdf5 / fc3.hdf5 already exist from the Python API):
  phono3py-init --dim {dim} --pa {pa} -c {prim_file}

  # Step 2 — run thermal conductivity using the existing FC2/FC3:
  phono3py --fc2 --fc3 --mesh {mesh} --br --wigner{fc_flag} \\
      --tmin {tmin} --tmax {tmax} --tstep {tstep}
""".strip("\n")
    else:
        return f"""
  # phono3py < 4: single CLI command.
  phono3py --fc2 --fc3 --mesh {mesh} --br --wigner \\
      --dim {dim} --pa {pa} -c {prim_file} \\
      --tmin {tmin} --tmax {tmax} --tstep {tstep}
""".strip("\n")
