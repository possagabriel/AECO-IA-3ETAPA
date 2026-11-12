"""Testes da interface de linha de comando."""

import importlib
import json
import runpy
import sys
from pathlib import Path

import pytest

from iris_nn._version import __version__
from iris_nn.cli import build_parser, main
from iris_nn.persistence import load_model


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.epochs == 1000
    assert args.learning_rate == 0.1
    assert args.hidden_size == 8
    assert args.test_size == 0.2
    assert args.seed == 42
    assert args.log_every == 100
    assert args.log_level == "INFO"
    assert args.report is None
    assert args.history is None
    assert args.save_model is None


def test_main_writes_all_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = tmp_path / "output" / "report"
    history = tmp_path / "output" / "loss"
    model = tmp_path / "output" / "model"

    exit_code = main(
        [
            "--epochs",
            "50",
            "--log-every",
            "50",
            "--log-level",
            "WARNING",
            "--report",
            str(report),
            "--history",
            str(history),
            "--save-model",
            str(model),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Acurácia" in captured.out
    assert "Relatório JSON:" in captured.out
    assert report.with_suffix(".json").exists()
    assert history.with_suffix(".csv").exists()
    assert model.with_suffix(".npz").exists()

    payload = json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["config"]["epochs"] == 50
    assert load_model(model).n_inputs == 4


def test_main_module_can_be_imported() -> None:
    module = importlib.import_module("iris_nn.__main__")

    assert callable(module.main)


def test_module_entrypoint_runs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["iris-nn", "--epochs", "5", "--log-level", "WARNING"],
    )
    sys.modules.pop("iris_nn.__main__", None)

    with pytest.raises(SystemExit) as error:
        runpy.run_module("iris_nn", run_name="__main__", alter_sys=True)

    assert error.value.code == 0
    assert "Acurácia" in capsys.readouterr().out


def test_main_without_artifacts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "--epochs",
            "5",
            "--learning-rate",
            "0.05",
            "--test-size",
            "0.25",
            "--log-level",
            "WARNING",
        ]
    )

    assert exit_code == 0
    assert "Acurácia" in capsys.readouterr().out


def test_main_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--version"])

    assert error.value.code == 0
    assert __version__ in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [
        ["--epochs", "abc"],
        ["--epochs", "0"],
        ["--learning-rate", "abc"],
        ["--learning-rate", "0"],
        ["--test-size", "abc"],
        ["--test-size", "2"],
        ["--log-every", "0"],
        ["--hidden-size", "-1"],
    ],
)
def test_main_rejects_invalid_numeric_arguments(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(arguments)

    assert error.value.code == 2


def test_main_reports_invalid_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def invalid_config(**kwargs: object) -> None:
        raise ValueError("configuração de teste")

    monkeypatch.setattr("iris_nn.cli.ExperimentConfig", invalid_config)

    exit_code = main(["--log-level", "WARNING"])

    assert exit_code == 2
    assert "Configuração inválida: configuração de teste" in capsys.readouterr().out


def test_main_returns_error_when_artifact_cannot_be_written(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("não é um diretório", encoding="utf-8")

    exit_code = main(
        [
            "--epochs",
            "10",
            "--log-every",
            "10",
            "--log-level",
            "WARNING",
            "--report",
            str(blocker / "report.json"),
        ]
    )

    assert exit_code == 1
    assert "Falha ao gravar artefatos" in capsys.readouterr().out
