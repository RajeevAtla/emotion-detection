"""Convenience entrypoint for running the project's pytest suite."""

from __future__ import annotations

import argparse
import sys

import pytest


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run repository tests")
    parser.add_argument(
        "--path",
        default="tests",
        help="Target path or test expression to pass to pytest (default: tests)",
    )
    parser.add_argument(
        "--cov",
        action="store_true",
        help="Collect coverage for the src/ directory.",
    )
    parser.add_argument(
        "pytest-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to pytest.",
    )
    return parser.parse_args()


def main() -> int:
    """Invoke pytest with the configured arguments."""
    args = parse_args()
    target = args.path
    pytest_args: list[str] = [target]
    if args.cov:
        pytest_args.extend(["--cov=src", "--cov-report=term"])
    if args.__dict__.get("pytest-args"):
        pytest_args.extend(args.__dict__["pytest-args"])
    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(main())
