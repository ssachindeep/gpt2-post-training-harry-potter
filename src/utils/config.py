import argparse
import yaml
from pathlib import Path


def _set_nested(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _parse_value(s):
    if s is None or s.lower() == "null":
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def load_config(config_path: str, overrides: list[str] | None = None) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for override in overrides or []:
        key_path, _, raw_value = override.partition("=")
        keys = key_path.strip().split(".")
        _set_nested(cfg, keys, _parse_value(raw_value.strip()))

    return cfg


def get_config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Dot-notation override, e.g. --override training.epochs=5")
    return parser


def parse_config(argv=None) -> dict:
    parser = get_config_parser()
    parser.add_argument("--help", "-h", action="help")
    args, _ = parser.parse_known_args(argv)
    return load_config(args.config, args.override)
