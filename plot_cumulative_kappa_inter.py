"""Plots cumulative kappa_inter vs frequency-gap cutoff, comparing SMM19/NJC23/IBDB19."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import h5py
import numpy as np

from thermal_transport_agent import Config, load_ph3_from_disk, _init_phph


def detect_stored_temperatures(out_dir: str, mesh_tag: str) -> list[float]:
    """Reads which temperatures are already stored in the cached gamma file(s) in out_dir."""
    candidates = [os.path.join(out_dir, f"kappa-m{mesh_tag}.hdf5")]
    candidates += sorted(glob.glob(os.path.join(out_dir, f"kappa-m{mesh_tag}-g*.hdf5")))
    for path in candidates:
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                return [float(t) for t in f["temperature"][()]]
    raise FileNotFoundError(
        f"No kappa-m{mesh_tag}.hdf5 or kappa-m{mesh_tag}-g*.hdf5 found in {out_dir} "
        "to detect which temperatures were already computed."
    )


def compute_pair_kappa(ph3, cfg: Config, all_temps: list[float], T: float, transport_type: str):
    """Runs read_gamma=True for one transport_type and returns per-band-pair kappa_inter data."""
    is_lbte = cfg.solver.lower() == "lbte"
    i_temp  = int(np.argmin(np.abs(np.array(all_temps) - T)))

    ph3.mesh_numbers = cfg.mesh_list
    _init_phph(ph3, None)

    orig_cwd = os.getcwd()
    os.chdir(cfg.out_dir)
    try:
        ph3.run_thermal_conductivity(
            temperatures   = all_temps,
            is_LBTE        = is_lbte,
            transport_type = transport_type,
            is_isotope     = cfg.isotope,
            mass_variances = cfg.mass_variances_parsed,
            read_gamma     = True,
            write_kappa    = False,
        )
    finally:
        os.chdir(orig_cwd)

    tc = ph3.thermal_conductivity
    freqs = tc.frequencies          # (num_gp, num_band)
    mkm   = tc.mode_kappa_matrix    # (num_sigma, num_temp, num_gp, num_band, num_band, 6)
    num_mesh_points = int(np.prod(cfg.mesh_list))

    i_sigma = 0
    mkm_xx = mkm[i_sigma, i_temp, :, :, :, :2].mean(axis=-1) / num_mesh_points
    mkm_zz = mkm[i_sigma, i_temp, :, :, :, 2] / num_mesh_points
    kappa_inter_xx_total = float(np.mean(tc.kappa_inter[i_sigma, i_temp, :2]))
    kappa_inter_zz_total = float(tc.kappa_inter[i_sigma, i_temp, 2])

    num_gp, num_band, _ = mkm_xx.shape
    num_gp, num_band, _ = mkm_zz.shape
    iu, ju = np.triu_indices(num_band, k=1)

    dw_parts, val_parts_xx, val_parts_zz = [], [], []
    for gp in range(num_gp):
        dw_parts.append(np.abs(freqs[gp, iu] - freqs[gp, ju]))
        val_parts_xx.append(mkm_xx[gp, iu, ju] + mkm_xx[gp, ju, iu])
        val_parts_zz.append(mkm_zz[gp, iu, ju] + mkm_zz[gp, ju, iu])

    dw  = np.concatenate(dw_parts)
    val_xx = np.concatenate(val_parts_xx)
    val_zz = np.concatenate(val_parts_zz)

    return dw, val_xx, val_zz, kappa_inter_xx_total, kappa_inter_zz_total


def compute_pair_kappa_2d(ph3, cfg: Config, all_temps: list[float], T: float, transport_type: str):
    """Like compute_pair_kappa, but keeps omega_i, omega_j, and BZ weight for an omega_i vs omega_j heatmap."""
    is_lbte = cfg.solver.lower() == "lbte"
    i_temp  = int(np.argmin(np.abs(np.array(all_temps) - T)))

    ph3.mesh_numbers = cfg.mesh_list
    _init_phph(ph3, None)

    orig_cwd = os.getcwd()
    os.chdir(cfg.out_dir)
    try:
        ph3.run_thermal_conductivity(
            temperatures   = all_temps,
            is_LBTE        = is_lbte,
            transport_type = transport_type,
            is_isotope     = cfg.isotope,
            mass_variances = cfg.mass_variances_parsed,
            read_gamma     = True,
            write_kappa    = False,
        )
    finally:
        os.chdir(orig_cwd)

    tc     = ph3.thermal_conductivity
    freqs  = tc.frequencies      # (num_gp, num_band)
    weight = tc.grid_weights     # (num_gp,)
    mkm    = tc.mode_kappa_matrix
    num_mesh_points = int(np.prod(cfg.mesh_list))

    i_sigma = 0
    mkm_xx = mkm[i_sigma, i_temp, :, :, :, :2].mean(axis=-1) / num_mesh_points
    mkm_zz = mkm[i_sigma, i_temp, :, :, :, 2] / num_mesh_points

    num_gp, num_band, _ = mkm_xx.shape
    iu, ju = np.triu_indices(num_band, k=1)

    wi_list, wj_list, xx_list, zz_list, w_list = [], [], [], [], []
    for gp in range(num_gp):
        wi, wj = freqs[gp, iu], freqs[gp, ju]
        vxx = mkm_xx[gp, iu, ju] + mkm_xx[gp, ju, iu]
        vzz = mkm_zz[gp, iu, ju] + mkm_zz[gp, ju, iu]
        # both orderings, so the plot is symmetric about the wi == wj diagonal
        wi_list.extend([wi, wj])
        wj_list.extend([wj, wi])
        xx_list.extend([vxx, vxx])
        zz_list.extend([vzz, vzz])
        w_list.extend([np.full(iu.shape, weight[gp])] * 2)

    return (np.concatenate(wi_list), np.concatenate(wj_list),
            np.concatenate(xx_list), np.concatenate(zz_list),
            np.concatenate(w_list), freqs, weight)


def plot_kappa_inter_diff_heatmap(wi, wj, diff, weight, freq, gp_weight, T, out_png,
                                   component_label, tt_a, tt_b, n_bins: int = 60,
                                   annotate_threshold: float | None = None):
    """Plots omega_i vs omega_j heatmap colored by <kappa_inter[tt_a] - kappa_inter[tt_b]>_q."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wmax = max(wi.max(), wj.max())
    bins = np.linspace(0, wmax, n_bins + 1)

    sum_wd, _, _ = np.histogram2d(wi, wj, bins=bins, weights=diff * weight)
    sum_w,  _, _ = np.histogram2d(wi, wj, bins=bins, weights=weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = sum_wd / sum_w
    avg = np.ma.masked_invalid(avg)

    # Signed: an arbitrary pair of transport_types isn't guaranteed a fixed-sign difference.
    vmax = float(np.abs(avg.compressed()).max()) if avg.count() else 1.0
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad(color="#bbbbbb")

    fig = plt.figure(figsize=(6.5, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0.06)
    ax_dos = fig.add_subplot(gs[0])
    ax     = fig.add_subplot(gs[1], sharex=ax_dos)

    n_band = freq.shape[1]
    dos_w  = np.repeat(gp_weight, n_band)
    dos, _ = np.histogram(freq.ravel(), bins=bins, weights=dos_w)
    ax_dos.bar(bins[:-1], dos, width=np.diff(bins), align="edge", color="steelblue")
    ax_dos.set_ylabel("DOS")
    ax_dos.set_yticks([])
    ax_dos.tick_params(labelbottom=False)
    for spine in ("top", "right", "left"):
        ax_dos.spines[spine].set_visible(False)
    ax_dos.set_title(f"<kappa_inter[{tt_a}] - kappa_inter[{tt_b}]>_q  "
                      f"({component_label}, T = {T:.0f} K)")

    pc = ax.pcolormesh(bins, bins, avg.T, cmap=cmap, vmin=-vmax, vmax=vmax, shading="flat")
    ax.set_xlabel("omega_i  [THz]")
    ax.set_ylabel("omega_j  [THz]")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bins[0], bins[-1])
    cbar = fig.colorbar(pc, ax=[ax_dos, ax], fraction=0.046, pad=0.02)
    cbar.set_label(f"mean kappa_inter[{tt_a}] - kappa_inter[{tt_b}]  [W/m/K]  (q-averaged)")

    if annotate_threshold is not None:
        centers = (bins[:-1] + bins[1:]) / 2
        filled = ~avg.mask
        candidates = [(m, n) for m, n in zip(*np.where(filled & (np.abs(avg.data) > annotate_threshold)))
                      if m < n]
        candidates.sort(key=lambda mn: abs(avg.data[mn]), reverse=True)
        accepted: list[tuple[int, int]] = []
        for m, n in candidates:
            if any(abs(m - am) <= 3 and abs(n - an) <= 3 for am, an in accepted):
                continue
            accepted.append((m, n))
        for m, n in accepted:
            ax.annotate(
                f"({centers[m]:.0f}, {centers[n]:.0f})\n{avg[m, n]:.1e}",
                xy=(centers[m], centers[n]), fontsize=6, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            )

    fig.text(0.02, 0.01, "gray = no band pairs in this bin", fontsize=7, color="dimgray")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cumulative kappa_inter vs frequency-gap cutoff, across transport_types.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out_dir", required=True,
                   help="Directory with fc2.hdf5/fc3.hdf5 (and phono3py_disp.yaml, if present)")
    p.add_argument("--structure", default="POSCAR",
                   help="Unit cell file -- only needed if phono3py_disp.yaml is absent from --out_dir")
    p.add_argument("--supercell", default="2 2 2")
    p.add_argument("--primitive_matrix", default="auto")
    p.add_argument("--symprec", type=float, default=1e-5)
    p.add_argument("--cutoff_frequency", type=float, default=1e-2,
                   help="Modes at/below this frequency (THz) are excluded from "
                        "coherence sums -- matches the native phono3py CLI's "
                        "own default (NOT the bare Phono3py() class default "
                        "of 1e-4). Raise further if a hiphive-fit SCPH FC2 "
                        "leaves a spurious near-zero Gamma-acoustic residual "
                        "above this -- that residual can dominate NJC23 "
                        "(whose prefactor doesn't vanish as omega->0) "
                        "without affecting IBDB19 much.")
    p.add_argument("--mesh", default="11 11 11",
                   help="Must match the mesh already used for gamma in --out_dir")
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--solver", default="rta", choices=["rta", "lbte"])
    p.add_argument("--isotope", action="store_true")
    p.add_argument("--mass_variances", default="")
    p.add_argument("--transport_types", default="SMM19 NJC23 IBDB19",
                   help="Space-separated subset of SMM19, NJC23, IBDB19")
    p.add_argument("--out", default=None, help="Output PNG (default: <out_dir>/kappa_inter_vs_dw.png)")
    p.add_argument("--heatmap_pair", default="NJC23 IBDB19",
                   help="Two transport_types to compare in the omega_i vs omega_j "
                        "kappa_inter-difference heatmap. Empty string disables it.")
    p.add_argument("--heatmap_annotate_threshold", type=float, default=None,
                   help="Label heatmap bins with |mean kappa_inter difference| above "
                        "this value [W/m/K] (default: no annotation)")

    args = p.parse_args()

    cfg = Config(
        structure=args.structure, out_dir=args.out_dir, supercell=args.supercell,
        primitive_matrix=args.primitive_matrix, symprec=args.symprec,
        cutoff_frequency=args.cutoff_frequency, mesh=args.mesh,
        temperatures=str(args.temperature), solver=args.solver, isotope=args.isotope,
        mass_variances=args.mass_variances,
    )
    ph3 = load_ph3_from_disk(Path(cfg.out_dir), cfg)

    all_temps = detect_stored_temperatures(cfg.out_dir, cfg.mesh_tag)
    print(f"  Detected {len(all_temps)} temperature(s) already in cached gamma: {all_temps}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for tt in args.transport_types.split():
        dw, val_xx, val_zz, total_xx, total_zz = compute_pair_kappa(ph3, cfg, all_temps, args.temperature, tt)
        order = np.argsort(dw)
        dw_sorted, cum_xx = dw[order], np.cumsum(val_xx[order])
        dw_sorted, cum_zz = dw[order], np.cumsum(val_zz[order])
        print(f"  {tt}: sanity check cum_xx[-1]={cum_xx[-1]:.4f}  "
              f" cum_zz[-1]={cum_zz[-1]:.4f}  "
              f"kappa_inter_xx={total_xx:.4f}  "
              f"kappa_inter_zz={total_zz:.4f}  "
              f"(ratio={cum_xx[-1]/total_xx if total_xx else float('nan'):.3f}, should be ~1.0) "
              f"(ratio={cum_zz[-1]/total_zz if total_zz else float('nan'):.3f}, should be ~1.0)")

        ax.plot(dw_sorted, cum_xx, label=tt+' xx')
        ax.plot(dw_sorted, cum_zz, label=tt +' zz')

    #ax.set_xscale("log")
    ax.set_xlabel("|dw| cutoff = |omega_s - omega_s'|  [THz]")
    ax.set_ylabel("Cumulative kappa_inter (iso)  [W/m/K]")
    ax.set_title(f"Where kappa_inter comes from  (T = {args.temperature:.0f} K)")
    ax.legend()
    fig.tight_layout()

    out_png = args.out or os.path.join(args.out_dir, f"kappa_inter_vs_dw_T{args.temperature:.0f}.png")
    fig.savefig(out_png, dpi=150)
    print(f"  wrote {out_png}")

    if args.heatmap_pair.strip():
        tt_a, tt_b = args.heatmap_pair.split()
        print(f"\n  omega_i vs omega_j kappa_inter heatmap: {tt_a} - {tt_b}")
        wi_a, wj_a, xx_a, zz_a, w_a, freq_a, gpw_a = compute_pair_kappa_2d(ph3, cfg, all_temps, args.temperature, tt_a)
        wi_b, wj_b, xx_b, zz_b, w_b, freq_b, gpw_b = compute_pair_kappa_2d(ph3, cfg, all_temps, args.temperature, tt_b)
        # wi/wj/w are identical between the two calls (same mesh, same gamma) --
        # only the mode_kappa values (xx/zz) differ by transport_type.
        for label, val_a, val_b in (("xx+yy", xx_a, xx_b), ("zz", zz_a, zz_b)):
            out_hm = os.path.join(args.out_dir,
                f"kappa_inter_diff_{tt_a}_vs_{tt_b}_{label}_T{args.temperature:.0f}.png")
            plot_kappa_inter_diff_heatmap(
                wi_a, wj_a, val_a - val_b, w_a, freq_a, gpw_a, args.temperature, out_hm,
                label, tt_a, tt_b, annotate_threshold=args.heatmap_annotate_threshold,
            )


if __name__ == "__main__":
    main()
