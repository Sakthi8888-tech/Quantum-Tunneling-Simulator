"""
Quantum Tunneling Simulator


Solves the 1D time-dependent Schrodinger equation

    i * dpsi/dt = -1/2 * d^2 psi/dx^2 + V(x) * psi        (atomic units: hbar = m = 1)

for a Gaussian wave packet incident on a rectangular potential barrier,
using the Crank-Nicolson finite-difference scheme (unconditionally stable,
unitary / norm-conserving to machine precision).

The module also provides the analytic plane-wave transmission coefficient
for a rectangular barrier, used to validate the numerical results.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

# numpy >= 2.0 renamed trapz -> trapezoid; keep this module working on both
_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class SimulationConfig:
    """Physical and numerical parameters for one tunneling simulation."""

    x_min: float = -40.0
    x_max: float = 50.0
    n_points: int = 3000

    dt: float = 0.01
    n_steps: int = 3000

    x0: float = -15.0        # initial center of the wave packet
    sigma: float = 3.0       # initial spatial width of the wave packet
    k0: float = 2.0          # initial mean wavenumber -> mean energy E = k0^2 / 2

    barrier_center: float = 0.0
    barrier_width: float = 2.0
    barrier_height: float = 4.0


class QuantumTunnelingSimulator:
    """Propagates a Gaussian wave packet through a rectangular barrier."""

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.x = np.linspace(config.x_min, config.x_max, config.n_points)
        self.dx = self.x[1] - self.x[0]

        self.V = self._build_potential()
        self.psi = self._initial_wavepacket()
        self._build_propagator()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _build_potential(self) -> np.ndarray:
        cfg = self.cfg
        V = np.zeros_like(self.x)
        mask = np.abs(self.x - cfg.barrier_center) < cfg.barrier_width / 2
        V[mask] = cfg.barrier_height
        return V

    def _initial_wavepacket(self) -> np.ndarray:
        cfg = self.cfg
        norm = (2 * np.pi * cfg.sigma ** 2) ** (-0.25)
        psi = (
            norm
            * np.exp(-((self.x - cfg.x0) ** 2) / (4 * cfg.sigma ** 2))
            * np.exp(1j * cfg.k0 * self.x)
        ).astype(complex)
        psi[0] = psi[-1] = 0.0
        return psi

    def _build_propagator(self) -> None:
        """Pre-factorize the Crank-Nicolson tridiagonal system.

        (I + i*dt/2 * H) psi_{n+1} = (I - i*dt/2 * H) psi_n
        """
        dx2 = self.dx ** 2
        N = self.cfg.n_points
        dt = self.cfg.dt

        main_diag = 1.0 / dx2 + self.V
        off_diag = -1.0 / (2 * dx2) * np.ones(N - 1)

        self.A_main = 1.0 + 1j * dt / 2 * main_diag
        self.A_off = 1j * dt / 2 * off_diag
        self.B_main = 1.0 - 1j * dt / 2 * main_diag
        self.B_off = -1j * dt / 2 * off_diag

        # banded storage expected by scipy.linalg.solve_banded: (upper, diag, lower)
        self.ab = np.zeros((3, N), dtype=complex)
        self.ab[0, 1:] = self.A_off
        self.ab[1, :] = self.A_main
        self.ab[2, :-1] = self.A_off

    # ------------------------------------------------------------------ #
    # Propagation
    # ------------------------------------------------------------------ #
    def _rhs(self) -> np.ndarray:
        psi = self.psi
        rhs = self.B_main * psi
        rhs[1:] += self.B_off * psi[:-1]
        rhs[:-1] += self.B_off * psi[1:]
        return rhs

    def step(self) -> None:
        rhs = self._rhs()
        self.psi = solve_banded((1, 1), self.ab, rhs)
        self.psi[0] = self.psi[-1] = 0.0

    def run(self, n_steps: int = None, save_every: int = 20):
        """Propagate the wave packet, returning (times, psi_history)."""
        n_steps = self.cfg.n_steps if n_steps is None else n_steps
        history = [self.psi.copy()]
        times = [0.0]
        for n in range(1, n_steps + 1):
            self.step()
            if n % save_every == 0:
                history.append(self.psi.copy())
                times.append(n * self.cfg.dt)
        return np.array(times), np.array(history)

    # ------------------------------------------------------------------ #
    # Observables
    # ------------------------------------------------------------------ #
    def norm(self) -> float:
        return float(_trapz(np.abs(self.psi) ** 2, self.x))

    def transmission_reflection(self):
        """Split probability across the barrier's right edge."""
        prob = np.abs(self.psi) ** 2
        right_edge = self.cfg.barrier_center + self.cfg.barrier_width / 2
        T = _trapz(prob[self.x > right_edge], self.x[self.x > right_edge])
        R = _trapz(prob[self.x <= right_edge], self.x[self.x <= right_edge])
        total = T + R
        return T / total, R / total


def analytic_transmission(E, V0: float, w: float) -> np.ndarray:
    """Exact plane-wave transmission coefficient for a rectangular barrier.

    E < V0 (tunneling regime):
        T = [1 + V0^2 sinh^2(k2 w) / (4 E (V0 - E))]^-1,  k2 = sqrt(2(V0-E))

    E > V0 (over-barrier regime, resonances):
        T = [1 + V0^2 sin^2(k2 w) / (4 E (E - V0))]^-1,   k2 = sqrt(2(E-V0))
    """
    E = np.atleast_1d(np.asarray(E, dtype=float))
    T = np.ones_like(E)

    below = E < V0
    if np.any(below):
        Eb = E[below]
        k2 = np.sqrt(2 * (V0 - Eb))
        T[below] = 1.0 / (
            1.0 + (V0 ** 2 * np.sinh(k2 * w) ** 2) / (4 * Eb * (V0 - Eb))
        )

    above = E > V0
    if np.any(above):
        Ea = E[above]
        k2 = np.sqrt(2 * (Ea - V0))
        T[above] = 1.0 / (
            1.0 + (V0 ** 2 * np.sin(k2 * w) ** 2) / (4 * Ea * (Ea - V0))
        )

    return T