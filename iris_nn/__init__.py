"""Rede neural do zero com NumPy para classificação do dataset Iris."""

from iris_nn._version import __version__
from iris_nn.data import IrisDataset, load_prepared_iris
from iris_nn.metrics import ClassificationMetrics, classification_metrics
from iris_nn.network import NeuralNetwork

__all__ = [
    "ClassificationMetrics",
    "IrisDataset",
    "NeuralNetwork",
    "__version__",
    "classification_metrics",
    "load_prepared_iris",
]
