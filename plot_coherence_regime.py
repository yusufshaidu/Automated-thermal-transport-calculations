"""Diagnose why SMM19/NJC23/IBDB19 coherence kappa can diverge, from an existing kappa-m*.hdf5.

No new phono3py run needed -- reads gamma (linewidths) and frequency,
already saved by any past `thermal_transport_agent.py bte` run, regardless
of which --transport_type produced the file.

NJC23 and IBDB19 share the same Lorentzian resonance kernel
    L(s,s') = g / (dw^2 + g^2),   g = gamma_s + gamma_s',  dw = omega_s - omega_s'
and differ only in their heat-capacity-matrix prefactor: NJC23 uses
(w_s+w_s')^2/4, IBDB19 uses w_s*w_s'. By AM-GM these agree when w_s ~ w_s'
and diverge more the further apart the two frequencies are -- but that only
matters for band pairs where L(s,s') is non-negligible, i.e. where the
linewidth broadening g is comparable to the gap dw. This script plots that
regime directly.

python plot_coherence_regime.py --kappa_hdf5 results/kappa-m191919.hdf5 --temperature 300
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np
from phonopy.physical_units import get_physical_units


def load_gamma_frequency(kappa_hdf5: str, temperature: float):
    with h5py.File(kappa_hdf5, "r") as f:
        temps = f["temperature"][()]
        it = int(np.argmin(np.abs(temps - temperature)))
        if abs(temps[it] - temperature) > 1e-6:
            print(f"  Note: using closest available temperature {temps[it]:.1f} K "
                  f"(requested {temperature:.1f} K)")
        gamma = f["gamma"][it]          # (n_gp, n_band)
        freq  = f["frequency"][()]      # (n_gp, n_band)
        weight = f["weight"][()]        # (n_gp,)
    return gamma, freq, weight, float(temps[it])


def pairwise_regime(gamma: np.ndarray, freq: np.ndarray, weight: np.ndarray):
    """Off-diagonal (s != s') g, |dw|, freq ratio, resonance kernel L, and
    BZ weight for every band pair at every q-point."""
    n_gp, n_band = freq.shape
    iu, ju = np.triu_indices(n_band, k=1)   # unique unordered pairs, s < s'

    g_list, dw_list, r_list, w_list = [], [], [], []
    for gp in range(n_gp):
        fi, fj = freq[gp, iu], freq[gp, ju]
        g_list.append(gamma[gp, iu] + gamma[gp, ju])
        dw_list.append(np.abs(fi - fj))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(fi > fj, fi / fj, fj / fi)  # fold to >= 1
        r_list.append(ratio)
        w_list.append(np.full(iu.shape, weight[gp]))

    g_sum = np.concatenate(g_list)
    dw    = np.concatenate(dw_list)
    ratio = np.concatenate(r_list)
    w     = np.concatenate(w_list)

    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = g_sum / (dw**2 + g_sum**2)
    kernel = np.where(np.isfinite(kernel), kernel, 0.0)
    ratio  = np.where(np.isfinite(ratio), ratio, 1.0)

    return g_sum, dw, ratio, kernel, w


def plot_regime_scatter(g_sum, dw, kernel, weight, T, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nonzero = (g_sum > 0) & (dw > 0)
    g_sum, dw, kernel, weight = g_sum[nonzero], dw[nonzero], kernel[nonzero], weight[nonzero]

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(dw, g_sum, c=np.log10(kernel + 1e-30), s=8 + 40 * (weight / weight.max()),
                     cmap="viridis", alpha=0.7, linewidths=0)
    lims = [min(dw.min(), g_sum.min()), max(dw.max(), g_sum.max())]
    ax.plot(lims, lims, "k--", lw=1, label="g = |dw|  (ambiguous regime)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|dw| = |omega_s - omega_s'|  [THz]")
    ax.set_ylabel("g = gamma_s + gamma_s'  [THz]")
    ax.set_title(f"Band-pair resonance regime at T = {T:.0f} K")
    ax.legend(loc="lower right", fontsize=8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log10(Lorentzian kernel L)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"  wrote {out_png}")

    # Fraction of total kernel weight sitting in the "ambiguous" regime
    # (within a factor of 3 of g == |dw|), where NJC23 and IBDB19 disagree
    # most and where a naive g>>dw or g<<dw argument doesn't decide things.
    ratio = g_sum / dw
    ambiguous = (ratio > 1 / 3) & (ratio < 3)
    total_w = np.sum(kernel * weight)
    ambig_w = np.sum(kernel[ambiguous] * weight[ambiguous])
    frac = ambig_w / total_w if total_w > 0 else float("nan")
    print(f"  Fraction of resonance-kernel weight with g within 3x of |dw| "
          f"(the regime where formulas disagree most): {frac:.1%}")
    return frac


def pairwise_cqij(freq: np.ndarray, weight: np.ndarray, T: float,
                   cutoff_frequency: float = 1e-2):
    """
    Off-diagonal (s != s') omega_s, omega_s', and C_NJC23 - C_IBDB19 (the
    heat-capacity-matrix entry each uses in the shared Lorentzian kernel),
    for every band pair at every q-point. Needs only frequency + T -- no
    gamma, no velocities -- since both formulas share the exact factor
    -1/T * (n_s - n_s')/(w_s - w_s'), differing only in a prefactor
    ((w_s+w_s')^2/4 for NJC23, w_s*w_s' for IBDB19). Mirrored across the
    diagonal (both (i,j) and (j,i)) so the returned arrays plot symmetrically.

    cutoff_frequency (THz) mirrors the native phono3py CLI's own default of
    1e-2 (set explicitly in cui/phono3py_script.py -- NOT the bare
    Phono3py() class's internal default of 1e-4): any pair where either
    mode is at/below this is zeroed out entirely, matching
    compute_bulk_cv_matrix()'s condition_2d masking in
    conductivity/heat_capacity_solvers.py. Without this, a near-zero
    Gamma-acoustic mode (always present, exactly 0 by symmetry) would show
    up here as a huge but *fictitious* divergence: NJC23's prefactor
    (w_i+w_j)^2/4 does not vanish as w_i -> 0, so this diagnostic would
    otherwise report a difference phono3py's real calculation never sees.
    """
    KB, THZ_TO_EV = get_physical_units().KB, get_physical_units().THzToEv

    n_gp, n_band = freq.shape
    iu, ju = np.triu_indices(n_band, k=1)

    def bose_einstein(e):
        with np.errstate(divide="ignore", over="ignore"):
            return 1.0 / (np.exp(e / (KB * T)) - 1.0)

    # (n_i - n_j)/(e_i - e_j) is a removable 0/0 singularity as e_i -> e_j
    # (both formulas' shared factor). Its analytic limit is dn/de evaluated
    # at e = e_i, i.e. -n(n+1)/(KB*T) -- NOT zero, and in fact this is where
    # the mode heat capacity is largest, so zeroing it would hide exactly
    # the near-degenerate pairs that matter most for the coherence term.
    eps = 1e-8   # eV; typical adjacent-mode gaps are >= 1e-4 eV (~0.02 THz)

    wi_list, wj_list, diff_list, w_list = [], [], [], []
    for gp in range(n_gp):
        wi, wj = freq[gp, iu], freq[gp, ju]
        ei, ej = wi * THZ_TO_EV, wj * THZ_TO_EV
        de = ei - ej
        degenerate = np.abs(de) < eps

        ni, nj = bose_einstein(ei), bose_einstein(ej)
        n_mid = bose_einstein(np.where(degenerate, (ei + ej) / 2, ei))

        with np.errstate(divide="ignore", invalid="ignore"):
            finite_diff_quotient = (ni - nj) / de
        limit_quotient = -n_mid * (n_mid + 1) / (KB * T)  # = dn/de at e_i == e_j
        quotient = np.where(degenerate, limit_quotient, finite_diff_quotient)

        shared   = -quotient / T
        c_njc23  = (ei + ej) ** 2 / 4 * shared
        c_ibdb19 = ei * ej * shared
        diff     = c_njc23 - c_ibdb19

        below_cutoff = (wi <= cutoff_frequency) | (wj <= cutoff_frequency)
        diff = np.where(below_cutoff, 0.0, diff)

        # both orderings, so the plot is symmetric about the wi == wj diagonal
        wi_list.extend([wi, wj])
        wj_list.extend([wj, wi])
        diff_list.extend([diff, diff])
        w_list.extend([np.full(iu.shape, weight[gp])] * 2)

    return (np.concatenate(wi_list), np.concatenate(wj_list),
            np.concatenate(diff_list), np.concatenate(w_list))


def plot_cqij_heatmap(wi, wj, diff, weight, freq, gp_weight, T, out_png,
                       n_bins: int = 60, annotate_threshold: float | None = 1e-1):
    """omega_s vs omega_s', colored by (C_NJC23 - C_IBDB19) averaged over q.

    A gray cell means no band pair in the mesh has that (omega_i, omega_j)
    combination -- which is ambiguous on its own: it could mean this exact
    combination is rare, or it could mean one (or both) of those
    frequencies simply doesn't exist anywhere in the phonon spectrum (a
    real gap, common in MOFs where light-atom X-H stretches sit far above
    the framework/lattice modes). The DOS panel on top disambiguates: if
    it's flat zero over some range, any gray cell touching that range is a
    genuine spectral gap, not a binning artifact.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wmax = max(wi.max(), wj.max())
    bins = np.linspace(0, wmax, n_bins + 1)

    sum_wd, _, _ = np.histogram2d(wi, wj, bins=bins, weights=diff * weight)
    sum_w,  _, _ = np.histogram2d(wi, wj, bins=bins, weights=weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = sum_wd / sum_w
    # diff is provably >= 0 (NJC23's prefactor >= IBDB19's by AM-GM); clip
    # away any roundoff-negative noise so 0 always means "equal", not "close to 0".
    avg = np.ma.masked_invalid(np.clip(avg, 0, None))

    # White at 0 ("the formulas agree here"), saturating to dark red as the
    # gap grows; masked (no band pairs in this bin) rendered as a visually
    # distinct flat gray so it's never confused with a real zero-difference bin.
    cmap = matplotlib.colormaps["OrRd"].copy()
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
    ax_dos.set_title(f"<C_NJC23 - C_IBDB19>_q  at T = {T:.0f} K")

    pc = ax.pcolormesh(bins, bins, avg.T, cmap=cmap, vmin=0, shading="flat")
    ax.set_xlabel("omega_i  [THz]")
    ax.set_ylabel("omega_j  [THz]")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bins[0], bins[-1])
    # No set_aspect("equal") here: it repads the axes box to keep a square
    # aspect ratio, which throws off the horizontal alignment with the DOS
    # panel above even though they nominally share an x-axis.
    cbar = fig.colorbar(pc, ax=[ax_dos, ax], fraction=0.046, pad=0.02)
    cbar.set_label("mean C_NJC23 - C_IBDB19  [eV/K]  (q-averaged)")

    if annotate_threshold is not None:
        centers = (bins[:-1] + bins[1:]) / 2
        filled = ~avg.mask
        # avg[m, n] <-> (omega_i bin m, omega_j bin n); only m < n so each
        # physical pair (mirrored onto both (m,n) and (n,m)) is considered once.
        candidates = [(m, n) for m, n in zip(*np.where(filled & (avg.data > annotate_threshold)))
                      if m < n]

        # Greedy non-max suppression, highest value first: skip a candidate
        # if it's within 1 bin of an already-accepted one. Deterministic
        # (unlike a per-bin local-window check, which can accept ties on
        # both sides of a plateau) and turns a broad contiguous region
        # above threshold into one label at its peak instead of an
        # unreadable stack of overlapping labels.
        candidates.sort(key=lambda mn: avg.data[mn], reverse=True)
        accepted: list[tuple[int, int]] = []
        for m, n in candidates:
            nms_radius = 3   # bins; wide enough that neighboring labels don't overlap on screen
            if any(abs(m - am) <= nms_radius and abs(n - an) <= nms_radius for am, an in accepted):
                continue
            accepted.append((m, n))

        for m, n in accepted:
            ax.annotate(
                f"({centers[m]:.0f}, {centers[n]:.0f})\n{avg[m, n]:.1e}",
                xy=(centers[m], centers[n]), fontsize=6, ha="center", va="center",
                color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            )
    fig.text(0.02, 0.01,
              "top panel: phonon DOS (flat zero = real spectral gap)   |   "
              "white = formulas agree   |   gray = no band pairs in this bin",
              fontsize=7, color="dimgray")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  wrote {out_png}")


def plot_prefactor_ratio(ratio, kernel, weight, out_png):
    """Analytic NJC23/IBDB19 prefactor ratio vs frequency mismatch, with a
    histogram of the material's own band pairs (weighted by resonance
    kernel * BZ weight, i.e. how much each pair actually matters) overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = np.logspace(-1.5, 1.5, 400)                 # omega_s/omega_s' >= 1
    njc23_over_ibdb19 = (1 + r) ** 2 / (4 * r)       # = (a+b)^2/4 / (a*b), scale-free

    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax1.plot(r, njc23_over_ibdb19, "b-", lw=2, label="NJC23 / IBDB19 prefactor ratio")
    ax1.axhline(1.0, color="gray", ls=":", lw=1)
    ax1.set_xscale("log")
    ax1.set_xlabel("omega_s / omega_s'  (folded to >= 1)")
    ax1.set_ylabel("NJC23 / IBDB19 heat-capacity-matrix prefactor", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    resonance_weight = kernel * weight
    nonzero = resonance_weight > 0
    if nonzero.any():
        bins = np.logspace(-1.5, 1.5, 40)
        ax2.hist(ratio[nonzero], bins=bins, weights=resonance_weight[nonzero],
                 color="orange", alpha=0.4, label="this material's band pairs\n(weighted by resonance kernel)")
    ax2.set_ylabel("Resonance-weighted band-pair density", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"  wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Diagnose SMM19/NJC23/IBDB19 coherence divergence from an existing kappa-m*.hdf5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--kappa_hdf5", required=True,
                   help="Path to an existing kappa-m*.hdf5 (any transport_type -- "
                        "gamma/frequency don't depend on it)")
    p.add_argument("--temperature", type=float, required=True,
                   help="Temperature (K) to analyze (closest available is used)")
    p.add_argument("--out_prefix", default=None,
                   help="Output file prefix (default: alongside --kappa_hdf5)")
    p.add_argument("--annotate_threshold", type=float, default=10000,
                   help="Label omega_i/omega_j heatmap bins with mean "
                        "C_NJC23 - C_IBDB19 [eV/K] above this value. "
                        "Set to a negative number to disable annotation.")
    p.add_argument("--cutoff_frequency", type=float, default=1e-2,
                   help="Modes at/below this frequency (THz) are excluded, "
                        "matching the native phono3py CLI's own default "
                        "(NOT the bare Phono3py() class default of 1e-4) -- "
                        "without this, the always-present Gamma-acoustic "
                        "mode (exactly 0 by symmetry) would make this "
                        "diagnostic report a fictitious divergence that "
                        "never shows up in the actual reported kappa_inter.")

    args = p.parse_args()
    prefix = args.out_prefix or args.kappa_hdf5.rsplit(".", 1)[0]
    annotate_threshold = args.annotate_threshold if args.annotate_threshold >= 0 else None

    gamma, freq, weight, T = load_gamma_frequency(args.kappa_hdf5, args.temperature)
    g_sum, dw, ratio, kernel, w = pairwise_regime(gamma, freq, weight)

    plot_regime_scatter(g_sum, dw, kernel, w, T, f"{prefix}_regime_T{T:.0f}.png")
    plot_prefactor_ratio(ratio, kernel, w, f"{prefix}_prefactor_ratio.png")

    wi, wj, diff, w_cqij = pairwise_cqij(freq, weight, T, cutoff_frequency=args.cutoff_frequency)
    plot_cqij_heatmap(wi, wj, diff, w_cqij, freq, weight, T, f"{prefix}_cqij_diff_T{T:.0f}.png",
                       annotate_threshold=annotate_threshold)


if __name__ == "__main__":
    main()
