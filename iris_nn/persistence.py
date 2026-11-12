"""Persistência de modelos em arquivos NumPy ``.npz``."""

from pathlib import Path
from typing import Any

import numpy as np

from iris_nn._version import __version__
from iris_nn.network import NeuralNetwork

_FORMAT_VERSION = 1


def _with_suffix(path: str | Path, suffix: str) -> Path:
    resolved = Path(path)
    if resolved.suffix.lower() != suffix:
        resolved = resolved.with_suffix(suffix)
    return resolved


def save_model(path: str | Path, model: NeuralNetwork) -> Path:
    """Salva os parâmetros e metadados do modelo e retorna o caminho final."""
    destination = _with_suffix(path, ".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": np.array(_FORMAT_VERSION, dtype=np.int64),
        "package_version": np.array(__version__),
        "n_inputs": np.array(model.n_inputs, dtype=np.int64),
        "n_hidden": np.array(model.n_hidden, dtype=np.int64),
        "n_outputs": np.array(model.n_outputs, dtype=np.int64),
        "learning_rate": np.array(model.learning_rate, dtype=np.float64),
        **model.state_dict(),
    }
    np.savez(destination, **payload)
    return destination


def load_model(path: str | Path) -> NeuralNetwork:
    """Carrega um modelo salvo por :func:`save_model`."""
    source = _with_suffix(path, ".npz")
    if not source.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {source}")

    with np.load(source, allow_pickle=False) as archive:
        format_version = int(np.asarray(archive["format_version"]).item())
        if format_version != _FORMAT_VERSION:
            raise ValueError(
                f"Versão de formato incompatível: {format_version} "
                f"(esperado: {_FORMAT_VERSION})."
            )

        model = NeuralNetwork(
            n_inputs=int(np.asarray(archive["n_inputs"]).item()),
            n_hidden=int(np.asarray(archive["n_hidden"]).item()),
            n_outputs=int(np.asarray(archive["n_outputs"]).item()),
            learning_rate=float(np.asarray(archive["learning_rate"]).item()),
            seed=0,
        )
        state = {name: np.asarray(archive[name]) for name in ("W1", "b1", "W2", "b2")}
        model.load_state_dict(state)

    return model
