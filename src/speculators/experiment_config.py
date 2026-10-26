"""Explicit YAML inheritance for individually runnable experiment configurations."""

import os
from pathlib import Path

import yaml


def merge_config(left, right):
    result = dict(left)
    for key, value in right.items():
        result[key] = (
            merge_config(result[key], value)
            if isinstance(result.get(key), dict) and isinstance(value, dict)
            else value
        )
    return result


def load_experiment_yaml(path, *, _parents=()):  # noqa: C901 -- parent resolution, cycles and environment expansion
    """Resolve parents relative to the declaring file; runtime paths use cwd.

    Lists replace rather than append. Cycles and unknown fields remain errors;
    the caller validates the resulting mapping against its own strict schema.
    """
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    parents = values.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
        raise ValueError(f"extends must be a path or list of paths: {path}")
    result = {}
    for parent in parents:
        result = merge_config(
            result,
            load_experiment_yaml(path.parent / parent, _parents=(*_parents, path)),
        )
    result = merge_config(result, values)

    def nested(value):
        if isinstance(value, list):
            return [nested(v) for v in value]
        if not isinstance(value, dict):
            return value
        value = dict(value)
        parents = value.pop("extends", [])
        parents = [parents] if isinstance(parents, str) else parents
        if not isinstance(parents, list) or not all(
            isinstance(p, str) for p in parents
        ):
            raise ValueError("Nested extends must be a path or list of paths")
        inherited = {}
        for parent in parents:
            inherited = merge_config(
                inherited,
                load_experiment_yaml(path.parent / parent, _parents=(*_parents, path)),
            )
        return merge_config(inherited, {k: nested(v) for k, v in value.items()})

    result = nested(result)

    def expand(value):
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, str):
            value = os.path.expandvars(value)
            if "${" in value:
                raise ValueError(f"Set the environment variable used in {value}")
        return value

    return expand(result)
