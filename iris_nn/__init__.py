"""Rede neural do zero com NumPy para classificação do dataset Iris."""

from iris_nn.data import IrisDataset, load_prepared_iris
from iris_nn.network import NeuralNetwork

__all__ = ["IrisDataset", "NeuralNetwork", "load_prepared_iris"]
