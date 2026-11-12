"""Métricas de classificação implementadas com NumPy."""

from dataclasses import dataclass

import numpy as np

from iris_nn._typing import AnyArray, FloatArray, IntArray


@dataclass(frozen=True, eq=False)
class ClassificationMetrics:
    """Métricas macro e por classe para um problema de classificação."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_precision: FloatArray
    per_class_recall: FloatArray
    per_class_f1: FloatArray
    support: IntArray
    confusion_matrix: IntArray

    @property
    def n_samples(self) -> int:
        """Quantidade total de amostras avaliadas."""
        return int(self.support.sum())


def labels_from_targets(targets: AnyArray, n_classes: int | None = None) -> IntArray:
    """Converte alvos one-hot ou rótulos inteiros em um vetor de classes.

    Args:
        targets: Matriz one-hot ``(n_amostras, n_classes)`` ou vetor de rótulos.
        n_classes: Quantidade esperada de classes. Quando informada, valida o
            formato e o intervalo dos rótulos.

    Returns:
        Vetor unidimensional de rótulos inteiros.

    Raises:
        ValueError: Se o formato, o tipo ou o intervalo dos alvos for inválido.
    """
    array = np.asarray(targets)

    if array.ndim == 2:
        if array.shape[1] < 1:
            raise ValueError("Os alvos one-hot devem ter ao menos uma coluna.")
        if n_classes is not None and array.shape[1] != n_classes:
            raise ValueError(
                f"Os alvos one-hot devem ter {n_classes} colunas, "
                f"mas possuem {array.shape[1]}."
            )
        labels = np.argmax(array, axis=1).astype(np.int64)
    elif array.ndim == 1:
        if array.dtype.kind == "f":
            if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
                raise ValueError("Os rótulos devem ser inteiros.")
        elif array.dtype.kind not in "iub":
            raise ValueError("Os rótulos devem ser inteiros.")
        labels = array.astype(np.int64)
    else:
        raise ValueError("Os alvos devem ser um vetor ou uma matriz one-hot.")

    if n_classes is not None:
        if n_classes < 1:
            raise ValueError("A quantidade de classes deve ser positiva.")
        if labels.size > 0 and (
            int(np.min(labels)) < 0 or int(np.max(labels)) >= n_classes
        ):
            raise ValueError("Os rótulos devem estar no intervalo [0, n_classes).")

    return labels


def confusion_matrix(
    y_true: AnyArray,
    y_pred: AnyArray,
    n_classes: int,
) -> IntArray:
    """Calcula a matriz de confusão com linhas reais e colunas previstas."""
    true_labels = labels_from_targets(y_true, n_classes=n_classes)
    predicted_labels = labels_from_targets(y_pred, n_classes=n_classes)

    if true_labels.shape != predicted_labels.shape:
        raise ValueError("y_true e y_pred devem ter a mesma quantidade de amostras.")

    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (true_labels, predicted_labels), 1)
    return matrix


def classification_metrics(
    y_true: AnyArray,
    y_pred: AnyArray,
    n_classes: int,
) -> ClassificationMetrics:
    """Calcula acurácia, precisão, recall e F1 com média macro.

    Classes sem amostras ou sem previsões recebem métricas iguais a zero.
    """
    if n_classes < 1:
        raise ValueError("A quantidade de classes deve ser positiva.")

    matrix = confusion_matrix(y_true, y_pred, n_classes)
    support = matrix.sum(axis=1).astype(np.int64)
    predicted = matrix.sum(axis=0)
    true_positives = np.diag(matrix).astype(np.float64)

    per_class_precision = np.divide(
        true_positives,
        predicted,
        out=np.zeros(n_classes, dtype=np.float64),
        where=predicted > 0,
    )
    per_class_recall = np.divide(
        true_positives,
        support,
        out=np.zeros(n_classes, dtype=np.float64),
        where=support > 0,
    )

    denominator = per_class_precision + per_class_recall
    per_class_f1 = np.divide(
        2.0 * per_class_precision * per_class_recall,
        denominator,
        out=np.zeros(n_classes, dtype=np.float64),
        where=denominator > 0,
    )

    total = int(matrix.sum())
    accuracy = float(true_positives.sum() / total) if total > 0 else 0.0

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_precision=float(np.mean(per_class_precision)),
        macro_recall=float(np.mean(per_class_recall)),
        macro_f1=float(np.mean(per_class_f1)),
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        support=support,
        confusion_matrix=matrix,
    )
