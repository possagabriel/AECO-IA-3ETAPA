"""Testes de fumaça dos pontos de entrada executáveis."""

import subprocess
import sys
from pathlib import Path

import pytest

from iris_nn._version import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, "main.py", "--version"],
        [sys.executable, "-m", "iris_nn", "--version"],
    ],
)
def test_entrypoints_print_version(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0
    assert __version__ in completed.stdout
