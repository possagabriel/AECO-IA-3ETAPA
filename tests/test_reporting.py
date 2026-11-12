"""Testes de geração de relatórios."""

import csv
import json
from pathlib import Path

import pytest

from iris_nn._version import __version__
from iris_nn.experiment import ExperimentResult
from iris_nn.reporting import (
    build_report,
    format_console_report,
    write_history_csv,
    write_json_report,
)


def test_build_report_contains_experiment_data(
    experiment_result: ExperimentResult,
) -> None:
    report = build_report(experiment_result)

    assert report["metadata"]["package_version"] == __version__
    assert report["config"]["epochs"] == experiment_result.config.epochs
    assert report["dataset"]["n_train"] == 120
    assert report["dataset"]["n_test"] == 30
    assert report["dataset"]["n_features"] == 4
    assert report["training"]["n_parameters"] == 67
    assert report["training"]["initial_loss"] == experiment_result.initial_loss
    assert report["training"]["final_loss"] == experiment_result.final_loss
    assert len(report["metrics"]["train"]["per_class"]) == 3
    assert len(report["metrics"]["test"]["confusion_matrix"]) == 3
    assert report["metrics"]["test"]["n_samples"] == 30


def test_write_json_report_creates_valid_file(
    tmp_path: Path,
    experiment_result: ExperimentResult,
) -> None:
    path = write_json_report(experiment_result, tmp_path / "nested" / "report")

    assert path == tmp_path / "nested" / "report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["config"]["seed"] == experiment_result.config.seed
    assert payload["metrics"]["test"]["accuracy"] == pytest.approx(
        experiment_result.test_metrics.accuracy
    )


def test_write_history_csv_creates_one_row_per_epoch(
    tmp_path: Path,
    experiment_result: ExperimentResult,
) -> None:
    path = write_history_csv(experiment_result, tmp_path / "loss")

    assert path == tmp_path / "loss.csv"
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))

    assert rows[0] == ["epoch", "loss"]
    assert len(rows) == len(experiment_result.history) + 1
    assert rows[1][0] == "1"
    assert rows[-1][0] == str(len(experiment_result.history))


def test_format_console_report_includes_key_sections(
    experiment_result: ExperimentResult,
) -> None:
    text = format_console_report(experiment_result)

    assert "Classificação de flores Iris" in text
    assert "4 -> 8 (ReLU) -> 3 (Softmax)" in text
    assert "Perda final" in text
    assert "Acurácia" in text
    assert "Matriz de confusão" in text
    for class_name in experiment_result.dataset.class_names:
        assert class_name in text
