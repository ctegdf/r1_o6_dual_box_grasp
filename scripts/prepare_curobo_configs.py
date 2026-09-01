#!/usr/bin/env python3
"""Materialize portable cuRobo YAML templates with absolute project paths."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_TOKEN = "__PROJECT_ROOT__"
CONFIG_NAMES = ("r1_o6.yml", "r1_o6_left.yml", "r1_o6_right.yml")


def materialize_configs(
    project_root: Path = PROJECT_ROOT,
    output_dir: Path | None = None,
) -> list[Path]:
    project_root = project_root.expanduser().resolve()
    template_dir = project_root / "configs"
    output_dir = (
        project_root / ".runtime" / "configs"
        if output_dir is None
        else output_dir.expanduser().resolve()
    )

    urdf_path = project_root / "R1-o6.urdf"
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in CONFIG_NAMES:
        source = template_dir / name
        template = source.read_text(encoding="utf-8")
        if CONFIG_TOKEN not in template:
            raise ValueError(f"missing {CONFIG_TOKEN} in {source}")
        rendered = template.replace(CONFIG_TOKEN, str(project_root))
        destination = output_dir / name
        destination.write_text(rendered, encoding="utf-8")
        written.append(destination)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Generated config directory (default: .runtime/configs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in materialize_configs(output_dir=args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
