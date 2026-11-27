"""Strict YAML defaults with explicit command-line overrides."""

from pathlib import Path
import argparse
import yaml


def parse_config(parser, relative_path):
    default = Path(__file__).resolve().parents[1] / relative_path
    parser.add_argument("--config", type=Path, default=default)
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=default)
    selected, _ = preliminary.parse_known_args()
    with selected.config.open() as stream:
        values = yaml.safe_load(stream) or {}
    actions = {a.dest: a for a in parser._actions}
    unknown = set(values) - set(actions)
    if unknown:
        parser.error(f"Unknown configuration keys: {sorted(unknown)}")
    for key, value in values.items():
        action = actions[key]
        if action.type is not None and value is not None:
            value = action.type(value)
        if action.choices is not None and value not in action.choices:
            parser.error(f"Invalid {key}: {value}")
        parser.set_defaults(**{key: value})
    args = parser.parse_args()
    from micrompc.paths import resolve_path

    for action in parser._actions:
        if action.type is Path:
            value = getattr(args, action.dest, None)
            if value is not None:
                setattr(args, action.dest, resolve_path(value))
    return args
