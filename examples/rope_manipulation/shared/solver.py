"""Convert the rope model to the shared batched primal-dual solver."""

import argparse
import jax
import numpy as np
import torch
from micrompc.solver.batched_primal_dual import BatchedPrimalDualMPC
from .perception import _torch_numpy


def make_solver(
    args: argparse.Namespace,
    model: torch.nn.Module,
    planning: dict[str, object],
    device: jax.Device,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> BatchedPrimalDualMPC:
    return BatchedPrimalDualMPC(
        a=_torch_numpy(model.predictor.A.weight, np.float64),
        b=_torch_numpy(model.predictor.B.weight, np.float64),
        c=_torch_numpy(model.predictor.C.weight, np.float64),
        c_bias=_torch_numpy(model.predictor.C.bias, np.float64),
        lift_params=tuple(
            (_torch_numpy(layer.weight), _torch_numpy(layer.bias))
            for layer in model.predictor.encoder
            if isinstance(layer, torch.nn.Linear)
        ),
        position_dim=int(planning["embed_dim"]),
        device=device,
        horizon=args.horizon,
        q_terminal=args.q_terminal,
        q_stage=args.q_stage,
        r_control=args.r_control,
        regularization=args.qp_regularization,
        q_velocity=args.q_velocity,
        q_acceleration=args.q_acceleration,
        rho=args.pd_rho,
        steps=args.pd_steps,
        mode=args.pd_mode,
        abs_tol=args.pd_abs_tol,
        rel_tol=args.pd_rel_tol,
        action_mean=action_mean,
        action_std=action_std,
    )
