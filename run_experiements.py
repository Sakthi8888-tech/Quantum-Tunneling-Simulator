"""
Generates all figures and result data for the quantum tunneling project.

Outputs (in ./figures):
    01_wavefunction_snapshots.png  - |psi|^2 before / after hitting the barrier
    02_spacetime_density.png       - space-time heatmap of |psi(x,t)|^2
    03_transmission_vs_energy.png  - numeric vs analytic transmission coefficient
    results_summary.txt            - printed numeric results
"""

import time as _time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantum_tunneling import (
    QuantumTunnelingSimulator,
    SimulationConfig,
    analytic_transmission,
)

FIG_DIR = "figures"


def run_main_demo():
    """Single representative run: E = k0^2/2 = 2.0, barrier height V0 = 4.0."""
    cfg = SimulationConfig(
        x_min=-100.0, x_max=100.0, n_points=6000,
        k0=2.0, barrier_height=4.0, n_steps=3000,
    )
    sim = QuantumTunnelingSimulator(cfg)

    psi_initial = sim.psi.copy()
    norm0 = sim.norm()

    times, history = sim.run(save_every=15)
    T, R = sim.transmission_reflection()
    norm_final = sim.norm()

    # --- Figure 1: initial vs final snapshots ------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sim.x, np.abs(psi_initial) ** 2, label="t = 0 (incident)", color="#1f77b4")
    ax.plot(sim.x, np.abs(sim.psi) ** 2, label=f"t = {times[-1]:.1f} (final)", color="#d62728")
    ax.fill_between(sim.x, 0, sim.V / sim.V.max() * np.abs(psi_initial).max() ** 2 * 1.1,
                     step="mid", color="gray", alpha=0.3, label="barrier (scaled)")
    ax.set_xlabel("x (a.u.)")
    ax.set_ylabel(r"$|\psi(x)|^2$")
    ax.set_title(f"Wave packet tunneling through a barrier  (E={cfg.k0**2/2:.2f}, V0={cfg.barrier_height})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/01_wavefunction_snapshots.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: space-time density heatmap ------------------------------
    density = np.abs(history) ** 2
    fig, ax = plt.subplots(figsize=(9, 6))
    extent = [sim.x[0], sim.x[-1], times[-1], times[0]]
    im = ax.imshow(density, aspect="auto", extent=extent, cmap="inferno")
    ax.axvline(cfg.barrier_center - cfg.barrier_width / 2, color="cyan", lw=1, ls="--")
    ax.axvline(cfg.barrier_center + cfg.barrier_width / 2, color="cyan", lw=1, ls="--")
    ax.set_xlabel("x (a.u.)")
    ax.set_ylabel("time (a.u.)")
    ax.set_title("Space-time evolution of |\u03c8(x,t)|\u00b2")
    fig.colorbar(im, ax=ax, label=r"$|\psi|^2$")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/02_spacetime_density.png", dpi=150)
    plt.close(fig)

    return {
        "E": cfg.k0 ** 2 / 2,
        "V0": cfg.barrier_height,
        "T_numeric": T,
        "R_numeric": R,
        "norm0": norm0,
        "norm_final": norm_final,
    }


def run_energy_scan():
    """Sweep incident energy across the barrier height and compare to theory."""
    V0 = 4.0
    barrier_width = 2.0
    k0_values = np.linspace(0.8, 4.2, 18)

    x0 = -15.0
    sigma = 3.0
    target_x = barrier_width / 2 + 4 * sigma + 10  # where packet should end up

    T_numeric = []
    energies = []

    for k0 in k0_values:
        t_final = (target_x - x0) / k0
        n_steps = max(int(t_final / 0.01), 200)

        cfg = SimulationConfig(
            x_min=-90.0, x_max=max(60.0, target_x + 15),
            n_points=4000,
            dt=0.01, n_steps=n_steps,
            x0=x0, sigma=sigma, k0=k0,
            barrier_height=V0, barrier_width=barrier_width,
        )
        sim = QuantumTunnelingSimulator(cfg)
        sim.run(n_steps=n_steps, save_every=n_steps)  # only need final state
        T, R = sim.transmission_reflection()
        T_numeric.append(T)
        energies.append(k0 ** 2 / 2)

    energies = np.array(energies)
    T_numeric = np.array(T_numeric)
    T_analytic = analytic_transmission(energies, V0, barrier_width)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(energies / V0, T_analytic, "-", color="#333333", lw=2, label="Analytic (plane wave)")
    ax.plot(energies / V0, T_numeric, "o", color="#d62728", ms=6, label="Numeric (wave packet)")
    ax.axvline(1.0, color="gray", ls="--", lw=1, label="E = V0")
    ax.set_xlabel(r"$E / V_0$")
    ax.set_ylabel("Transmission coefficient T")
    ax.set_title(f"Transmission through a rectangular barrier (V0={V0}, w={barrier_width})")
    ax.legend()
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/03_transmission_vs_energy.png", dpi=150)
    plt.close(fig)

    mae = np.mean(np.abs(T_numeric - T_analytic))
    return energies, T_numeric, T_analytic, mae


if __name__ == "__main__":
    import os

    os.makedirs(FIG_DIR, exist_ok=True)

    t0 = _time.time()
    print("Running main demonstration (single wave packet vs barrier)...")
    demo_results = run_main_demo()
    print(f"  done in {_time.time() - t0:.1f}s")

    t1 = _time.time()
    print("Running energy scan (numeric vs analytic transmission)...")
    energies, T_numeric, T_analytic, mae = run_energy_scan()
    print(f"  done in {_time.time() - t1:.1f}s")

    lines = []
    lines.append("QUANTUM TUNNELING SIMULATION - RESULTS SUMMARY")
    lines.append("=" * 50)
    lines.append("")
    lines.append("Main demo run:")
    lines.append(f"  Incident energy E      = {demo_results['E']:.3f}")
    lines.append(f"  Barrier height V0      = {demo_results['V0']:.3f}")
    lines.append(f"  Transmission T (num)   = {demo_results['T_numeric']:.4f}")
    lines.append(f"  Reflection R (num)     = {demo_results['R_numeric']:.4f}")
    lines.append(f"  Norm conservation      = {demo_results['norm0']:.6f} -> {demo_results['norm_final']:.6f}")
    lines.append("")
    lines.append("Energy scan (E/V0, T_numeric, T_analytic):")
    for e, tn, ta in zip(energies, T_numeric, T_analytic):
        lines.append(f"  E/V0={e/4.0:5.2f}   T_num={tn:.4f}   T_analytic={ta:.4f}")
    lines.append("")
    lines.append(f"Mean absolute error (numeric vs analytic): {mae:.5f}")

    summary = "\n".join(lines)
    print("\n" + summary)

    with open("results_summary.txt", "w") as f:
        f.write(summary + "\n")