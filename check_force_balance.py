"""
Produces a two-panel figure comparing the estimated vs. simulated zonal velocity
as function of the non-dimensional Lorentz force, for:
  a) zonal-flow regime      (omega/Omega = 7.59)
  b) inertial-wave regime   (omega/Omega = 1)

Requires the MagIC Python post-processing library and simulation output.
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

# Colorblind-friendly palette
EK_COLORS = {
    1e-4: "#0072B2",
    1e-5: "#E69F00",
    1e-6: "#009E73",
    1e-7: "#D55E00",
}

# ── Simulation data root ──────────────────────────────────────────────────────
SIM_DIR = "/scratch/delangen/magic_runs"

# ── Run definitions ───────────────────────────────────────────────────────────
# Zonal flow runs (omega/Omega = 7.59). Each entry is (ek, freq, b, rm).
ZF_RUNS = [
    (1e-4, 1.2080e4, 5e-6, 1),
    (1e-4, 1.2080e4, 5e-7, 1),
    (1e-4, 1.2080e4, 5e-8, 1),
    (1e-5, 1.2080e5, 5e-6, 1),
    (1e-5, 1.2080e5, 5e-7, 1),
    (1e-5, 1.2080e5, 5e-8, 1),
    (1e-6, 1.2080e6, 5e-6, 1),
    (1e-6, 1.2080e6, 5e-7, 1),
    (1e-6, 1.2080e6, 5e-8, 1),
    (1e-7, 1.2080e7, 5e-7, 1),
]

# Inertial wave runs (omega/Omega = 1). Each entry is (ek, freq, b, rm, is_orbital).
# is_orbital=True for runs at the orbital frequency rather than the synodic frequency.
IW_RUNS = [
    (1e-4, 1.5915e3, 5e-7,  1,    False),
    (1e-4, 1.5915e3, 5e-7,  15.9, False),
    (1e-5, 1.5915e4, 5e-7,  1,    False),
    (1e-5, 1.5915e4, 5e-6,  1,    False),
    (1e-5, 1.5915e4, 5e-8,  1,    False),
    (1e-5, 1.5915e4, 5e-7,  15.9, False),
    (1e-6, 1.5915e5, 5e-7,  1,    False),
    (1e-6, 1.5915e5, 5e-6,  1,    False),
    (1e-6, 1.5915e5, 5e-8,  1,    False),
    (1e-6, 1.5915e5, 5e-7,  15.9, False),
    (1e-7, 1.5915e6, 5e-7,  1,    False),
    (1e-7, 1.5915e6, 5e-7,  15.9, False),
    (1e-4, 1.5915e3, 5e-7,  1,    True),
    (1e-5, 1.5915e4, 5e-7,  1,    True),
]


# ── Path helper ───────────────────────────────────────────────────────────────
def get_run_path(regime, ek, freq, b, rm):
    """Return the full path to a simulation directory."""
    folder = f"ek{ek:.0e}_freq{freq:.3g}_b{b:.0e}_rm{rm}"
    return os.path.join(SIM_DIR, regime, folder)


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
    g :                    MagicGraph
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
def compute_lorentz_mean(ivar=None):
    """Return the volume mean of the azimuthal Lorentz force from a time-averaged graphic file."""
    gr = MagicGraph(ivar=ivar)
    return compute_shell_mean(gr.LFphi, gr)


def compute_lorentz_rms(ek, pm, ivar=None):
    """Return the volume RMS of the azimuthal Lorentz force from a single snapshot graphic file."""
    s1 = Surf(ivar=ivar) if ivar is not None else Surf()
    data, *_ = s1.surf(field='Lorentz_p', r=0.99, iplot=False)
    data_rescaled = data / (ek * pm)
    gr = MagicGraph(ivar=ivar)
    return compute_shell_rms(data_rescaled, gr)


def compute_lorentz_rms_timeavg(ek, pm, ivarmax):
    """Return the volume RMS of the azimuthal Lorentz force averaged over snapshots 1 to ivarmax."""
    rms_values = [compute_lorentz_rms(ek, pm, ivar=i) for i in range(1, ivarmax + 1)]
    return np.mean(rms_values)


# ── Run loaders ───────────────────────────────────────────────────────────────
def load_lorentz_zf(ek, pm):
    """Load time-averaged azimuthal Lorentz force for a zonal-flow run."""
    try:
        return compute_lorentz_mean()
    except Exception:
        return compute_lorentz_mean(ivar=30)


def load_lorentz_iw(ek, pm, is_orbital=False):
    """Load Lorentz force RMS for an inertial-wave run.
    For orbital-frequency runs, time-averages over the first 14 snapshots
    rather than using a single snapshot."""
    if is_orbital:
        return compute_lorentz_rms_timeavg(ek, pm, ivarmax=14)
    try:
        return compute_lorentz_rms(ek, pm)
    except Exception:
        return compute_lorentz_rms(ek, pm, ivar=30)


def sci_label(ek):
    """Format an Ekman number as a LaTeX scientific-notation string."""
    exp   = int(np.floor(np.log10(ek)))
    coeff = ek / 10**exp
    return fr'$Ek = {coeff:.0f} \times 10^{{{exp}}}$'


# ── Main figure ───────────────────────────────────────────────────────────────
def plot_velocity_ratio():
    """
    Two-panel figure: estimated vs. simulated velocity ratio as a function of
    the dimensionless Lorentz force, for zonal-flow (a) and inertial-wave (b)
    regimes across a range of Ekman numbers.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    # ── Panel (a): Zonal Flow ─────────────────────────────────────────────────
    for ek, freq, b, rm in ZF_RUNS:
        pm    = 1 / freq
        color = EK_COLORS[ek]

        os.chdir(get_run_path("zonal_flow", ek, freq, b, rm))
        ts      = MagicTs(field="e_kin", iplot=False)
        ekin    = np.average(ts.ekin_tor_axi[-50:])
        lorentz = load_lorentz_zf(ek=ek, pm=pm)

        # u_sim: RMS velocity derived from volume-averaged kinetic energy (E_kin = 0.5 * V * u^2)
        u_sim = np.sqrt(2 * ekin / SHELL_VOLUME)
        # u_exp: estimated zonal velocity from balancing Lorentz force against viscous force
        # (see de Langen et al., submitted)
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
    for ek, freq, b, rm, is_orbital in IW_RUNS:
        pm    = 1 / freq
        color = EK_COLORS[ek]

        os.chdir(get_run_path("inertial_wave", ek, freq, b, rm))
        ts      = MagicTs(field="e_kin", iplot=False)
        ekin    = np.average(ts.ekin_tot[-10:])
        lorentz = load_lorentz_iw(ek=ek, pm=pm, is_orbital=is_orbital)

        # u_sim: RMS velocity derived from volume-averaged kinetic energy (E_kin = 0.5 * V * u^2)
        u_sim    = np.sqrt(2 * ekin / SHELL_VOLUME)
        # u_exp: estimated velocity from balancing Lorentz force against viscous force
        # (see de Langen et al., submitted)
        u_exp    = lorentz * np.sqrt(ek) * SHELL_VOLUME / SHELL_SURFACE
        marker   = "*" if is_orbital else ("o" if rm == 1 else "s")
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
    plot_velocity_ratio()