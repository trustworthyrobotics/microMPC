"""Normalized-control adapter for the shared differentiable inverse QP."""

import torch
from micrompc.solver.inverse_qp import DifferentiableInverseQP as InverseQP


class DifferentiableInverseQP:
    """Lift visual Markov states and solve in normalized action coordinates."""

    def __init__(self, horizon, regularization):
        self.horizon = horizon
        self.solver = InverseQP(horizon, regularization)

    def solve(self, model, initial_state, target_states):
        latent = model.lift_state(initial_state.float())
        a, b, c = model.A.weight.float(), model.B.weight.float(), model.C.weight.float()
        bias = model.C.bias.float() if model.C.bias is not None else None
        states = []
        for _ in range(self.horizon):
            latent = torch.nn.functional.linear(latent, a)
            states.append(torch.nn.functional.linear(latent, c, bias))
        return self.solver.solve(
            a, b, c, torch.stack(states, dim=1), target_states.float()
        )
