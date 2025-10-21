from __future__ import annotations

class PDController:
    """Simple Proportional-Derivative (PD) controller.

    Usage:
        ctrl = PDController(kp=0.15, kd=0.6)
        u = ctrl(reference, observation)
    The controller stores the previous error internally so it can compute
    the discrete derivative term.
    """

    def __init__(self, kp: float = 0.15, kd: float = 0.6):
        self.kp = kp
        self.kd = kd
        self._prev_error = 0.0

    def reset(self) -> None:
        """Reset internal state (previous error)."""
        self._prev_error = 0.0

    def __call__(self, reference: float, observation: float) -> float:
        """Compute control action for one timestep.

        Args:
            reference: desired reference r[t]
            observation: measured output y[t]

        Returns:
            control action u[t]
        """
        error = reference - observation
        derivative = error - self._prev_error
        u = self.kp * error + self.kd * derivative
        self._prev_error = error
        return u
