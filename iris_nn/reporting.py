"""Relatórios de experimentos em texto, JSON e CSV."""

import csv
import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from iris_nn._version import __version__
from iris_nn.experiment import ExperimentResult
from iris_nn.metrics import ClassificationMetrics


def _with_suffix(path: str | Path, suffix: str) -> Path:
    resolved = Path(path)
    if resolved.suffix.lower() != suffix:
        resolved = resolved.with_suffix(suffix)
    return resolved


def metrics_to_dict(
    metrics: ClassificationMetrics,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    """Serializa métricas para uma estrutura compatível com JSON."""
    per_class = [
        {
            "class": class_name,
            "precision": float(metrics.per_class_precision[index]),
            "recall": float(metrics.per_class_recall[index]),
            "f1": float(metrics.per_class_f1[index]),
            "support": int(metrics.support[index]),
        }
        for index, class_name in enumerate(class_names)
    ]

    return {
        "accuracy": metrics.accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "n_samples": metrics.n_samples,
        "per_class": per_class,
        "confusion_matrix": metrics.confusion_matrix.tolist(),
    }


def build_report(result: ExperimentResult) -> dict[str, Any]:
    """Monta o relatório completo do experimento."""
    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "package_version": __version__,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
        "config": asdict(result.config),
        "dataset": {
            "n_train": result.dataset.n_train,
            "n_test": result.dataset.n_test,
            "n_features": result.dataset.n_features,
            "n_classes": result.dataset.n_classes,
            "feature_names": list(result.dataset.feature_names),
            "class_names": list(result.dataset.class_names),
        },
        "training": {
            "epochs": result.config.epochs,
            "n_parameters": result.n_parameters,
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
            "elapsed_seconds": result.elapsed_seconds,
        },
        "metrics": {
            "train": metrics_to_dict(result.train_metrics, result.dataset.class_names),
            "test": metrics_to_dict(result.test_metrics, result.dataset.class_names),
        },
    }


def write_json_report(result: ExperimentResult, path: str | Path) -> Path:
    """Grava o relatório em JSON e retorna o caminho final."""
    destination = _with_suffix(path, ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as file:
        json.dump(build_report(result), file, indent=2, ensure_ascii=False)
        file.write("\n")

    return destination


def write_history_csv(result: ExperimentResult, path: str | Path) -> Path:
    """Grava o histórico de perda por época em CSV."""
    destination = _with_suffix(path, ".csv")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss"])
        for epoch, loss in enumerate(result.history, start=1):
            writer.writerow([epoch, f"{loss:.10f}"])

    return destination


def format_console_report(result: ExperimentResult) -> str:
    """Formata um resumo legível do experimento para o terminal."""
    model = result.model
    config = result.config
    train = result.train_metrics
    test = result.test_metrics
    class_names = result.dataset.class_names
    width = 72

    lines = [
        "=" * width,
        "Classificação de flores Iris com rede neural em NumPy".center(width),
        "=" * width,
        (
            f"Treino: {result.dataset.n_train} amostras | "
            f"Teste: {result.dataset.n_test} amostras"
        ),
        (
            f"Arquitetura: {model.n_inputs} -> {model.n_hidden} (ReLU) -> "
            f"{model.n_outputs} (Softmax)"
        ),
        (
            f"Épocas: {config.epochs} | "
            f"Taxa de aprendizado: {config.learning_rate:g} | "
            f"Seed: {config.seed}"
        ),
        f"Parâmetros treináveis: {result.n_parameters}",
        "-" * width,
        f"Perda inicial:   {result.initial_loss:.6f}",
        f"Perda final:     {result.final_loss:.6f}",
        f"Tempo de treino: {result.elapsed_seconds:.4f} s",
        "-" * width,
        "Métricas",
        f"  {'':<16}{'Treino':>12}{'Teste':>12}",
        f"  {'Acurácia':<16}{train.accuracy:>11.2%}{test.accuracy:>12.2%}",
        (
            f"  {'Precisão macro':<16}"
            f"{train.macro_precision:>11.2%}{test.macro_precision:>12.2%}"
        ),
        (
            f"  {'Recall macro':<16}"
            f"{train.macro_recall:>11.2%}{test.macro_recall:>12.2%}"
        ),
        f"  {'F1 macro':<16}{train.macro_f1:>11.2%}{test.macro_f1:>12.2%}",
        "",
        "Matriz de confusão de teste (linhas = real, colunas = previsto)",
    ]

    cell_width = max(12, max(len(name) for name in class_names) + 2)
    lines.append(" " * 16 + "".join(f"{name:>{cell_width}}" for name in class_names))
    for class_name, row in zip(
        class_names,
        result.test_metrics.confusion_matrix,
        strict=True,
    ):
        cells = "".join(f"{int(value):>{cell_width}}" for value in row)
        lines.append(f"{class_name:<16}{cells}")

    lines.append("=" * width)
    return "\n".join(lines)
