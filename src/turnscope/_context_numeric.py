"""Optional dense numerical fitting; frozen model inference does not import NumPy."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

try:
    # Late module loading keeps the base package dependency-free and avoids
    # parsing a newer NumPy release's Python-version-specific type stubs.
    np = importlib.import_module("numpy")
except ImportError as error:  # pragma: no cover - exercised by isolated base-wheel smoke
    raise ImportError(
        "Expected-context fitting requires pip install 'turnscope[context]'"
    ) from error


@dataclass(frozen=True)
class NumericContextFit:
    basis: tuple[tuple[float, ...], ...]
    coefficients: tuple[tuple[float, ...], ...]
    source_mean: tuple[float, ...]
    context_mean: tuple[float, ...]
    singular_values: tuple[float, ...]
    training_mse: float
    mean_baseline_mse: float


def fit_numeric_context(
    source_rows: list[list[float]],
    context_rows: list[list[float]],
    context_pair_indices: list[int],
    components: int,
    regularization: float,
) -> NumericContextFit:
    """Fit a context SVD and centered multi-output ridge map in binary64."""
    x = np.asarray(source_rows, dtype=np.float64)
    y = np.asarray(context_rows, dtype=np.float64)
    try:
        _, singular_values, vt = np.linalg.svd(y, full_matrices=False)
        tolerance = max(y.shape) * np.finfo(np.float64).eps * singular_values[0]
        rank = int(np.count_nonzero(singular_values > tolerance))
        if rank == 0:
            raise ValueError("training context matrix has numerical rank zero")
        dimensions = min(components, rank)
        basis = vt[:dimensions].T.copy()
        # Fix the arbitrary sign using the first maximal-magnitude loading.
        for dimension in range(dimensions):
            pivot = int(np.argmax(np.abs(basis[:, dimension])))
            if basis[pivot, dimension] < 0:
                basis[:, dimension] *= -1
        z = (y @ basis)[context_pair_indices]
        source_mean = x.mean(axis=0)
        context_mean = z.mean(axis=0)
        xc, zc = x - source_mean, z - context_mean
        gram = xc.T @ xc
        gram.flat[:: gram.shape[0] + 1] += regularization
        coefficients = np.linalg.solve(gram, xc.T @ zc)
        residual = xc @ coefficients - zc
        training_mse = float(np.mean(residual * residual))
        baseline_mse = float(np.mean(zc * zc))
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "expected-context numerical fit failed; increase regularization"
        ) from error
    if not all(
        bool(np.all(np.isfinite(value)))
        for value in (basis, coefficients, source_mean, context_mean, singular_values)
    ) or not np.isfinite(training_mse + baseline_mse):
        raise ValueError("expected-context numerical fit produced non-finite parameters")
    return NumericContextFit(
        tuple(tuple(float(value) for value in row) for row in basis),
        tuple(tuple(float(value) for value in row) for row in coefficients),
        tuple(float(value) for value in source_mean),
        tuple(float(value) for value in context_mean),
        tuple(float(value) for value in singular_values[:dimensions]),
        training_mse,
        baseline_mse,
    )
