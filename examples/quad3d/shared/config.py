"""Load embodiment standards and complete, typed program YAML configurations.

Configuration precedence: command line > selected YAML. Every program argument must
have a YAML entry; unknown/missing keys fail before models or environments load.
Repository-relative artifact paths resolve independently of the working directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

from .paths import resolve_path


def load_yaml(path: str | Path) -> dict:
    with resolve_path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def parse_configured_args(
    parser: argparse.ArgumentParser, default_path: Path, argv=None
) -> argparse.Namespace:
    """Apply complete YAML defaults, then let argparse apply CLI overrides."""
    parser.add_argument(
        "--config",
        default=str(default_path),
        help="Program YAML; defaults to the file alongside this script.",
    )
    arguments = sys.argv[1:] if argv is None else argv
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config", default=str(default_path))
    selected, _ = selector.parse_known_args(arguments)
    # --help works even when a user is repairing an invalid configuration.
    if "--help" in arguments or "-h" in arguments:
        return parser.parse_args(arguments)
    document = load_yaml(selected.config)
    if document.get("schema_version") != 1:
        parser.error("Program YAML schema_version must be 1")
    values = {}
    for section, entries in document.items():
        if section == "schema_version":
            continue
        if not isinstance(entries, dict):
            parser.error(f"Config section {section!r} must be a mapping")
        overlap = values.keys() & entries.keys()
        if overlap:
            parser.error(f"Repeated YAML options: {sorted(overlap)}")
        values.update(entries)
    actions = {a.dest: a for a in parser._actions if a.dest not in {"help", "config"}}
    missing, unknown = actions.keys() - values.keys(), values.keys() - actions.keys()
    if missing or unknown:
        parser.error(
            f"YAML options: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    for key, value in values.items():
        action = actions[key]
        try:
            if isinstance(
                action,
                (
                    argparse._StoreTrueAction,
                    argparse._StoreFalseAction,
                    argparse.BooleanOptionalAction,
                ),
            ):
                if not isinstance(value, bool):
                    raise ValueError("must be true or false")
            elif value is not None:
                vector = isinstance(action.nargs, int) or action.nargs in ("+", "*")
                sequence = value if vector else [value]
                if vector and not isinstance(value, list):
                    raise ValueError("must be a YAML list")
                if isinstance(action.nargs, int) and len(sequence) != action.nargs:
                    raise ValueError(f"must contain {action.nargs} values")
                converted = []
                for item in sequence:
                    if isinstance(item, bool):
                        raise ValueError("boolean used for a non-boolean argument")
                    if action.type is int and (not isinstance(item, int)):
                        raise ValueError("must be an integer")
                    result = action.type(item) if action.type else item
                    if action.choices is not None and result not in action.choices:
                        raise ValueError(f"must be one of {action.choices}")
                    converted.append(result)
                values[key] = converted if vector else converted[0]
        except (ValueError, TypeError) as exc:
            parser.error(f"YAML {key}: {exc}")
    parser.set_defaults(**values)
    return parser.parse_args(arguments)


def add_planning_args(parser):
    """Common physical objective, runtime, and PD configuration options."""
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction)
    if "model_path" not in {a.dest for a in parser._actions}:
        parser.add_argument("--model-path", type=str)
    if "device" not in {a.dest for a in parser._actions}:
        parser.add_argument("--device", type=str, choices=("cpu", "cuda"))
    if "precision" not in {a.dest for a in parser._actions}:
        parser.add_argument("--precision", type=int, choices=(32, 64))
    if "horizon" not in {a.dest for a in parser._actions}:
        parser.add_argument("--horizon", type=int)
    if "max_mpc_steps" not in {a.dest for a in parser._actions}:
        parser.add_argument("--max-mpc-steps", type=int)
    if "seed" not in {a.dest for a in parser._actions}:
        parser.add_argument("--seed", type=int)
    if "dt" not in {a.dest for a in parser._actions}:
        parser.add_argument("--dt", type=float)
    if "goal_tol" not in {a.dest for a in parser._actions}:
        parser.add_argument("--goal-tol", type=float)
    if "control_weight" not in {a.dest for a in parser._actions}:
        parser.add_argument("--control-weight", type=float)
    if "q_state" not in {a.dest for a in parser._actions}:
        parser.add_argument("--q-state", type=float, nargs=12)
    if "q_terminal" not in {a.dest for a in parser._actions}:
        parser.add_argument("--q-terminal", type=float, nargs=12)
    if "u_min" not in {a.dest for a in parser._actions}:
        parser.add_argument("--u-min", type=float, nargs=4)
    if "u_max" not in {a.dest for a in parser._actions}:
        parser.add_argument("--u-max", type=float, nargs=4)
    if "control_reference" not in {a.dest for a in parser._actions}:
        parser.add_argument("--control-reference", type=float, nargs=4)
    if "cache_dir" not in {a.dest for a in parser._actions}:
        parser.add_argument("--cache-dir", type=Path)
    if "pd_rho" not in {a.dest for a in parser._actions}:
        parser.add_argument("--pd-rho", type=float)
    if "pd_iterations" not in {a.dest for a in parser._actions}:
        parser.add_argument("--pd-iterations", type=int)
    if "pd_mode" not in {a.dest for a in parser._actions}:
        parser.add_argument("--pd-mode", type=str, choices=("fixed", "converged"))
    if "pd_primal_tol" not in {a.dest for a in parser._actions}:
        parser.add_argument("--pd-primal-tol", type=float)
    if "pd_fixed_point_tol" not in {a.dest for a in parser._actions}:
        parser.add_argument("--pd-fixed-point-tol", type=float)
    if "pd_step_safety" not in {a.dest for a in parser._actions}:
        parser.add_argument("--pd-step-safety", type=float)
    if "pd_extrapolation" not in {a.dest for a in parser._actions}:
        parser.add_argument("--pd-extrapolation", type=float)
    if "warm_start" not in {a.dest for a in parser._actions}:
        parser.add_argument("--warm-start", action=argparse.BooleanOptionalAction)
    if "workspace_lower" not in {a.dest for a in parser._actions}:
        parser.add_argument("--workspace-lower", type=float, nargs=3)
    if "workspace_upper" not in {a.dest for a in parser._actions}:
        parser.add_argument("--workspace-upper", type=float, nargs=3)


def validate_planning_args(parser, args):
    import math

    if min(args.horizon, args.max_mpc_steps, args.pd_iterations) < 1:
        parser.error("horizon, max_mpc_steps and pd_iterations must be positive")
    if (
        min(
            args.dt,
            args.goal_tol,
            args.pd_rho,
            args.pd_primal_tol,
            args.pd_fixed_point_tol,
        )
        <= 0
    ):
        parser.error(
            "time step, goal tolerance, rho and PD tolerances must be positive"
        )
    if not 0 < args.pd_step_safety < 1 or not 0 <= args.pd_extrapolation <= 1:
        parser.error("invalid PD step safety or extrapolation")
    if any(
        not math.isfinite(v)
        for v in [
            args.pd_rho,
            args.dt,
            args.goal_tol,
            args.control_weight,
            *args.q_state,
            *args.q_terminal,
            *args.u_min,
            *args.u_max,
            *args.control_reference,
        ]
    ):
        parser.error("physical and solver parameters must be finite")
    if args.control_weight <= 0:
        parser.error("control_weight must be positive")
    if any(lo >= hi for lo, hi in zip(args.u_min, args.u_max)):
        parser.error("each control lower bound must be below its upper bound")
    if any(lo >= hi for lo, hi in zip(args.workspace_lower, args.workspace_upper)):
        parser.error("workspace lower bounds must be below upper bounds")
    for key in (
        "model_path",
        "cache_dir",
        "video_path",
        "results_path",
        "collision_plot_path",
        "npz_path",
        "debug_numerics_log",
    ):
        if hasattr(args, key) and getattr(args, key) is not None:
            setattr(args, key, resolve_path(getattr(args, key)))
    return args
