"""
Produces a two-panel figure comparing the estimated vs. simulated zonal velocity
normalised by the Lorentz force magnitude, for:
  (a) zonal-flow regime      (omega/Omega = 7.59)
  (b) inertial-wave regime   (omega/Omega = 1)

Requires the MagIC Python post-processing library and simulation output on the
local scratch disk. See README for details.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import roots_legendre
from magic import MagicTs, MagicGraph, Surf

# ── Physical constants (non-dimensionalised MagIC units) ─────────────────────
SHELL_VOLUME   = 1.415966e+03
SHELL_SURFACE  = 1.551404e+03
OMEGA_RATIO_ZF = 7.59   # omega/Omega for zonal-flow runs

# ── Plot styling ──────────────────────────────────────────────────────────────
FONT_LARGE  = 24
FONT_MEDIUM = 19
FONT_SMALL  = 18
MARKER_SIZE = 10

EK_COLORS = {
    1e-4: "#0072B2",
    1e-5: "#E69F00",
    1e-6: "#009E73",
    1e-7: "#D55E00",
}

# ── Simulation data root ──────────────────────────────────────────────────────
RUNS_ROOT = "/scratch/delangen/magic_runs"

# ── Run definitions ───────────────────────────────────────────────────────────
# Keys are (ek, b); values are (run_dir, tag).

RUNS_ZF = {
    (1e-4, 5e-6): ("paper_final/res_part3", "ek1e-4_zonal_flow_b5e-6"),
    (1e-4, 5e-7): ("ek1e-4_freqs",          "ek1e-4_zonal_flow"),
    (1e-4, 5e-8): ("paper_final/res_part2", "ek1e-4_zonal_flow_b5e-8_2"),
    (1e-5, 5e-6): ("paper_final/res_part3", "ek1e-5_zonal_flow_b5e-6"),
    (1e-5, 5e-7): ("paper_final/res_part1", "ek1e-5_zonal_flow_b5e-7"),
    (1e-5, 5e-8): ("paper_final/res_part1", "ek1e-5_zonal_flow_b5e-8"),
    (1e-6, 5e-6): ("zonal_flow",            "ek1e-6_zonal_flow_b5e-6"),
    (1e-6, 5e-7): ("zonal_flow",            "ek1e-6"),
    (1e-6, 5e-8): ("ek1e-6_rmvar",          "ek1e-6_zonal_flow_b5e-8_2"),
    (1e-7, 5e-7): ("ek1e-7",                "ek1e-7_zonal_flow_2"),
}

# Keys are (pm, ek, b); values are (run_dir, tag, is_orbital).
# is_orbital=True for runs at the orbital frequency, where the Lorentz force
# is time-averaged over multiple snapshots rather than taken from a single one.
RUNS_IW = {
    (1e-2,         1e-4, 5e-7): ("ek1e-4_freqs",         "ek1e-4_freq1.592e3_pm1e-2",  False),
    (1e-3,         1e-5, 5e-7): ("ratio_1",               "ek1e-5_freq15e3",             False),
    (1e-4,         1e-6, 5e-7): ("ek1e-6_rmvar",          "ek1e-6_freq1.59e5_pm1e-4",   False),
    (1e-5,         1e-7, 5e-7): ("ek1e-7",                "ek1e-7_freq1.59e6_pm1e-5_2", False),
    (2e-5*np.pi,   1e-5, 5e-7): ("paper_final/res_part1", "ek1e-5_omega_o_timeavg",      True),
    (2e-5*np.pi,   1e-5, 5e-6): ("ratio_1",               "ek1e-5_ratio1_rm1_b5e-6",    False),
    (2e-5*np.pi,   1e-5, 5e-8): ("ratio_1",               "ek1e-5_ratio1_rm1_b5e-8",    False),
    (2e-6*np.pi,   1e-6, 5e-7): ("ek1e-6_rmvar",          "ek1e-6_freq1.5e5_rm1_correct",False),
    (2e-6*np.pi,   1e-6, 5e-6): ("ek1e-6_rmvar",          "ek1e-6_freq1.5e5_rm1_b5e-6", False),
    (2e-6*np.pi,   1e-6, 5e-8): ("ek1e-6_rmvar",          "ek1e-6_freq1.5e5_rm1_b5e-8_2",False),
    (2e-4*np.pi,   1e-4, 5e-7): ("paper_final/res_part2", "ek1e-4_omega_o_timeavg",      True),
    (2e-7*np.pi,   1e-7, 5e-7): ("ek1e-7",                "ek1e-7_freq1.592e6_rm1_2",   False),
}


# ── Quadrature ────────────────────────────────────────────────────────────────
def _quadrature_weights(n_phi, n_theta, n_r, g):
    """
    Compute the 3D quadrature weight tensor for a MagIC grid.
    Uses uniform weights in phi, Gauss-Legendre in theta (colatitude), and
    Clenshaw-Curtis weights for Chebyshev-Gauss-Lobatto points in r.

    Parameters
    ----------
    n_phi, n_theta, n_r : int
        Grid dimensions in the azimuthal, colatitudinal, and radial directions.
    g : MagicGraph
        MagIC graphic object providing the radial grid via g.radius.

    Returns
    -------
    W : np.ndarray, shape (n_phi, n_theta, n_r)
        Quadrature weight tensor including the r^2 sin(theta) volume element.
    """

    w_phi = np.ones(n_phi) * (2 * np.pi / n_phi)

    _, w_theta = roots_legendre(n_theta)

    r = g.radius
    n = n_r - 1
    w_r = np.zeros(n_r)
    for k in range(n_r):
        theta_k = k * np.pi / n
        w_r[k] = 1.0
        for j in range(1, n // 2 + 1):
            b = 2.0 if j < n // 2 else 1.0
            w_r[k] -= b * np.cos(2 * j * theta_k) / (4 * j**2 - 1)
        w_r[k] /= n
    w_r[0]  /= 2
    w_r[-1] /= 2
    w_r *= abs(r[-1] - r[0]) / 2

    w_r_vol = w_r * r**2

    return w_phi[:, None, None] * w_theta[None, :, None] * w_r_vol[None, None, :]


def compute_shell_mean(field, g):
    """Volume-weighted mean of a scalar field on a MagIC spherical shell grid."""
    W = _quadrature_weights(*field.shape, g)
    return (W * field).sum() / W.sum()


def compute_shell_rms(field, g):
    """Volume-weighted RMS of a scalar field on a MagIC spherical shell grid."""
    W = _quadrature_weights(*field.shape, g)
    return np.sqrt((W * field**2).sum() / W.sum())


# ── Lorentz force ─────────────────────────────────────────────────────────────
def compute_lorentz_mean(tag, ek, pm, ivar=None):
    """Return the volume mean of the azimuthal Lorentz force from a time-averaged graphic file."""
    gr = MagicGraph(ivar=ivar, tag=tag)
    return compute_shell_mean(gr.LFphi, gr)


def compute_lorentz_rms(tag, ek, pm, ivar=None):
    """Return the volume RMS of the rescaled azimuthal Lorentz force from a single snapshot."""
    s1 = Surf(ivar=ivar, tag=tag) if ivar is not None else Surf(tag=tag)
    data, *_ = s1.surf(field='Lorentz_p', r=0.99, iplot=False)
    data_rescaled = data / (ek * pm)
    gr = MagicGraph(ivar=ivar, tag=tag)
    return compute_shell_rms(data_rescaled, gr)


def compute_lorentz_rms_timeavg(tag, ek, pm, ivarmax):
    """Return the Lorentz force RMS averaged over snapshots 1 through ivarmax."""
    rms_values = [compute_lorentz_rms(tag, ek, pm, ivar=i) for i in range(1, ivarmax + 1)]
    return np.mean(rms_values)


# ── Run loaders ───────────────────────────────────────────────────────────────
def load_lorentz_zf(tag, ek, pm):
    """Load time-averaged azimuthal Lorentz force for a zonal-flow run."""
    try:
        return compute_lorentz_mean(tag, ek, pm)
    except Exception:
        return compute_lorentz_mean(tag, ek, pm, ivar=30)


def load_lorentz_iw(tag, ek, pm, is_orbital=False):
    """Load Lorentz force RMS for an inertial-wave run.
    For orbital-frequency runs, time-averages over multiple snapshots (14 for
    ek=1e-4, 30 otherwise) rather than using a single snapshot."""
    if is_orbital:
        ivarmax = 14 if tag == "ek1e-4_omega_o_timeavg" else 30
        return compute_lorentz_rms_timeavg(tag, ek, pm, ivarmax)
    try:
        return compute_lorentz_rms(tag, ek, pm)
    except Exception:
        return compute_lorentz_rms(tag, ek, pm, ivar=30)



def sci_label(ek):
    """Format an Ekman number as a LaTeX scientific-notation string."""
    exp   = int(np.floor(np.log10(ek)))
    coeff = ek / 10**exp
    return fr'$Ek = {coeff:.0f} \times 10^{{{exp}}}$'


# ── Main figure ───────────────────────────────────────────────────────────────
def plot_forcing_vs_velocity():
    """
    Two-panel figure: estimated vs. simulated velocity ratio as a function of
    the dimensionless Lorentz force, for zonal-flow (a) and inertial-wave (b)
    regimes across a range of Ekman numbers.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    # ── Panel (a): Zonal Flow ─────────────────────────────────────────────────
    for (ek, b), (run_dir, tag) in RUNS_ZF.items():
        freq  = OMEGA_RATIO_ZF / (2 * np.pi) * ek
        pm    = 1 / freq
        color = EK_COLORS[ek]

        os.chdir(os.path.join(RUNS_ROOT, run_dir))
        ts   = MagicTs(field="e_kin", tag=tag, iplot=False)
        ekin = np.average(ts.ekin_tor_axi[-50:])
        lorentz   = load_lorentz_zf(tag=tag, ek=ek, pm=pm)

        u_sim = np.sqrt(2 * ekin / SHELL_VOLUME)
        u_exp = -lorentz * np.sqrt(ek) * SHELL_VOLUME / SHELL_SURFACE

        ax1.scatter(
            -lorentz * ek**2, u_exp / u_sim,
            color=color, s=MARKER_SIZE**2,
            label=sci_label(ek) if b == 5e-7 else None,
        )

    ax1.set_xscale("log")
    ax1.set_ylim(0.5, 1.1)
    ax1.set_xlabel(r"$F_{L}$", fontsize=FONT_LARGE)
    ax1.set_ylabel(r"$U_{\mathrm{est}}/U_{\mathrm{sim}}$", fontsize=FONT_LARGE)
    ax1.tick_params(labelsize=FONT_MEDIUM)
    ax1.grid(linestyle="--", alpha=0.5)
    ax1.set_title(rf"Zonal flow ($\omega/\Omega = {OMEGA_RATIO_ZF}$)", fontsize=FONT_MEDIUM)
    ax1.text(0.04, 0.96, "(a)", transform=ax1.transAxes,
             fontsize=FONT_LARGE, fontweight="bold", va="top")

    # ── Panel (b): Inertial Wave ──────────────────────────────────────────────
    for (pm, ek, b), (run_dir, tag, is_orbital) in RUNS_IW.items():
        rm    = pm / (2 * np.pi * ek)
        color = EK_COLORS[ek]

        os.chdir(os.path.join(RUNS_ROOT, run_dir))
        ts      = MagicTs(field="e_kin", tag=tag, iplot=False)
        ekin    = np.average(ts.ekin_tot[-10:])
        lorentz = load_lorentz_iw(tag=tag, ek=ek, pm=pm, is_orbital=is_orbital)

        u_sim    = np.sqrt(2 * ekin / SHELL_VOLUME)
        u_exp    = lorentz * np.sqrt(ek) * SHELL_VOLUME / SHELL_SURFACE
        marker   = "*" if is_orbital else ("o" if round(rm, 2) == 1.00 else "s")
        dot_size = (MARKER_SIZE * 2)**2 if is_orbital else MARKER_SIZE**2

        ax2.scatter(lorentz * ek**2, u_exp / u_sim,
                    color=color, marker=marker, s=dot_size)

    ax2.set_xscale("log")
    ax2.set_xlim(1e-15, 1e-10)
    ax2.set_xlabel(r"$F_{L}$", fontsize=FONT_LARGE)
    ax2.tick_params(labelsize=FONT_MEDIUM)
    ax2.grid(linestyle="--", alpha=0.5)
    ax2.set_title(r"Inertial wave ($\omega/\Omega = 1$)", fontsize=FONT_MEDIUM)
    ax2.text(0.04, 0.96, "(b)", transform=ax2.transAxes,
             fontsize=FONT_LARGE, fontweight="bold", va="top")

    # ── Legend ────────────────────────────────────────────────────────────────
    ek_handles, _ = ax1.get_legend_handles_labels()
    marker_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="black",
               markersize=MARKER_SIZE, label=r"$Rm = 15.9$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=MARKER_SIZE, label=r"$Rm = 1.0$"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
               markersize=MARKER_SIZE * 2, label=r"$Rm = 1.0$, $\omega_o$"),
    ]
    ax1.legend(handles=ek_handles + marker_handles, fontsize=FONT_SMALL, ncol=2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_forcing_vs_velocity()