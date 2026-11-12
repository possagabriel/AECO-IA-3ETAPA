"""Testes das métricas de classificação."""

import numpy as np
import pytest

from iris_nn.metrics import (
    classification_metrics,
    confusion_matrix,
    labels_from_targets,
)


def test_labels_from_targets_accepts_one_hot_and_integer_labels() -> None:
    one_hot = np.eye(3)[[0, 2, 1]]
    labels = np.array([0, 2, 1])

    np.testing.assert_array_equal(labels_from_targets(one_hot), labels)
    np.testing.assert_array_equal(labels_from_targets(labels), labels)
    np.testing.assert_array_equal(labels_from_targets(labels.astype(float)), labels)


def test_labels_from_targets_respects_n_classes() -> None:
    np.testing.assert_array_equal(
        labels_from_targets(np.eye(3)[[2, 0]], n_classes=3),
        np.array([2, 0]),
    )


@pytest.mark.parametrize(
    "targets",
    [
        np.zeros((2, 2, 2)),
        np.array([0.0, 1.5]),
        np.array([0, 3]),
        np.array(["a", "b"]),
    ],
)
def test_labels_from_targets_rejects_invalid_values(targets: np.ndarray) -> None:
    with pytest.raises(ValueError):
        labels_from_targets(targets, n_classes=3)


def test_labels_from_targets_rejects_wrong_one_hot_width() -> None:
    with pytest.raises(ValueError, match="colunas"):
        labels_from_targets(np.eye(4), n_classes=3)


def test_labels_from_targets_rejects_empty_one_hot_columns() -> None:
    with pytest.raises(ValueError, match="coluna"):
        labels_from_targets(np.empty((2, 0)), n_classes=3)


def test_labels_from_targets_rejects_non_positive_n_classes() -> None:
    with pytest.raises(ValueError, match="classes"):
        labels_from_targets(np.array([0]), n_classes=0)


def test_confusion_matrix_counts_predictions() -> None:
    y_true = np.array([0, 1, 2, 2, 0])
    y_pred = np.array([0, 2, 2, 1, 0])

    matrix = confusion_matrix(y_true, y_pred, n_classes=3)

    expected = np.array(
        [
            [2, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
        ]
    )
    np.testing.assert_array_equal(matrix, expected)


def test_classification_metrics_computes_macro_averages() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 2, 1])

    metrics = classification_metrics(y_true, y_pred, n_classes=3)

    assert metrics.accuracy == pytest.approx(4.0 / 6.0)
    assert metrics.macro_precision == pytest.approx(2.0 / 3.0)
    assert metrics.macro_recall == pytest.approx(2.0 / 3.0)
    assert metrics.macro_f1 == pytest.approx(2.0 / 3.0)
    assert metrics.n_samples == 6
    np.testing.assert_allclose(metrics.per_class_precision, [1.0, 0.5, 0.5])
    np.testing.assert_allclose(metrics.per_class_recall, [1.0, 0.5, 0.5])
    np.testing.assert_allclose(metrics.per_class_f1, [1.0, 0.5, 0.5])
    np.testing.assert_array_equal(metrics.support, [2, 2, 2])


def test_classification_metrics_for_perfect_predictions() -> None:
    targets = np.eye(3)

    metrics = classification_metrics(targets, targets, n_classes=3)

    assert metrics.accuracy == 1.0
    assert metrics.macro_precision == 1.0
    assert metrics.macro_recall == 1.0
    assert metrics.macro_f1 == 1.0
    np.testing.assert_array_equal(metrics.support, [1, 1, 1])


def test_metrics_handle_classes_without_predictions() -> None:
    metrics = classification_metrics(
        np.array([0, 1]),
        np.array([0, 0]),
        n_classes=3,
    )

    assert np.all(np.isfinite(metrics.per_class_precision))
    assert metrics.per_class_precision[2] == 0.0
    assert metrics.per_class_recall[2] == 0.0


def test_confusion_matrix_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="mesma quantidade"):
        confusion_matrix(np.array([0, 1]), np.array([0]), n_classes=2)


def test_classification_metrics_rejects_invalid_n_classes() -> None:
    with pytest.raises(ValueError, match="classes"):
        classification_metrics(np.array([0]), np.array([0]), n_classes=0)
