"""Physical-control adapter for the shared differentiable inverse QP."""

import torch
from micrompc.solver.inverse_qp import DifferentiableInverseQP as InverseQP


class DifferentiableInverseQP:
    """Lift quadrotor states and recover normalized physical controls."""

    def __init__(self, horizon, control_mean, control_std, regularization):
        self.horizon = horizon
        self.control_mean = control_mean.detach().clone()
        self.control_std = control_std.detach().clone()
        self.solver = InverseQP(horizon, regularization)

    def solve(self, model, initial_state, target_states):
        latent = model.encoder(initial_state)
        states = []
        for _ in range(self.horizon):
            latent = model.A(latent)
            states.append(model.C(latent))
        return self.solver.solve(
            model.A.weight,
            model.B.weight,
            model.C.weight,
            torch.stack(states, dim=1),
            target_states,
            control_mean=self.control_mean,
            control_std=self.control_std,
        )

    def solve_from_rollout(
        self, model, predicted_states, demonstrated_controls, target_states
    ):
        return self.solver.solve(
            model.A.weight,
            model.B.weight,
            model.C.weight,
            predicted_states,
            target_states,
            control_mean=self.control_mean,
            control_std=self.control_std,
            demonstrated_controls=demonstrated_controls,
        )

    def normalize_controls(self, controls):
        return (controls - self.control_mean.to(controls)) / self.control_std.to(
            controls
        )

    def denormalize_controls(self, controls):
        return controls * self.control_std.to(controls) + self.control_mean.to(controls)
