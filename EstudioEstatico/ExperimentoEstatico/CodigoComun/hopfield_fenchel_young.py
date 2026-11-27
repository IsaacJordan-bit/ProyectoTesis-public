"""
hopfield_fenchel_young.py

Implementación NumPy de la especialización probabilística/sparse de
Hopfield-Fenchel-Young (HFY) descrita por Santos et al. (2025),
"Hopfield-Fenchel-Young Networks: A Unified Framework for Associative
Memory Retrieval" (arXiv:2411.08590v2).

ALCANCE MATEMÁTICO
==================

Este módulo implementa la familia de las secciones 3-4 del paper con:

    Psi(q) = 1/2 ||q||^2,

por lo que la post-transformación y_hat_Psi es la identidad. Con la
convención del paper, donde las memorias son filas de X (N x D), la
actualización sparse-HFY es:

    q^(t+1) = X.T @ y_hat_Omega(beta * X @ q^t).          (Eq. 21)

El pipeline de la tesis almacena las memorias como COLUMNAS:

    patterns_matrix.shape == (D, N)

por lo que la misma actualización se escribe aquí como:

    scores  = patterns_matrix.T @ q
    logits  = beta * scores
    weights = y_hat_Omega(logits)
    q_next  = patterns_matrix @ weights

No se aplica:

    - división por sqrt(D),
    - normalización L2 de patrones o queries,
    - reescalamiento adicional de logits,
    - post-transformación adicional.

Esto es deliberado: el Experimento 2 trabaja con embeddings RAW y beta
es el eje experimental dinámico.

SCORING IMPLEMENTADO
====================

    "softmax"
        Shannon / 1-entmax.

    "entmax"
        Tsallis alpha-entmax, alpha >= 1.
        Eq. (17) del paper.

    "sparsemax"
        2-entmax / sparsemax.

    "normmax"
        gamma-normmax, gamma > 1.
        Eq. (18) y Algoritmo 4 del Apéndice A.

CONTRATO CON exp2_pipeline_integrated.py
========================================

Ruta individual:

    memory = HopfieldFenchelYoungNetwork(...)
    memory.train(patterns_matrix)                         # (D, N)
    recalled = memory.recall(
        query=query,                                      # (D,)
        beta=beta,
        iterations=iterations,
    )                                                     # (D,)

Ruta vectorizada:

    recalled = hfy_recall_matrix(
        patterns_matrix=patterns_matrix,                  # (D, N)
        query_matrix=query_matrix,                        # (D, Q)
        beta=beta,
        iterations=iterations,
        ...,
    )                                                     # (Q, D)

La forma de salida (Q, D) de hfy_recall_matrix() es exactamente la
esperada por run_classification_process().

SEPARACIÓN DE RESPONSABILIDADES
===============================

Este módulo NO importa AssociativeMemorySpec ni Exp2Config y NO define
factories dinámicas de clases. La declaración de una
AssociativeMemorySpec debe realizarse en una capa externa (por ejemplo,
un módulo de specs o el notebook), manteniendo desacoplada la
implementación matemática del pipeline experimental.

Los parámetros:

    scoring, alpha, gamma

pueden formar parte de network_params cuando sean científicamente
relevantes para la variante declarada.

Los parámetros:

    bisection_iterations, bisection_tolerance, store_diagnostics

son detalles numéricos/operacionales de implementación y no deberían
utilizarse para definir la identidad científica de la memoria.

NOTA SOBRE GARANTÍAS DE RECUPERACIÓN EXACTA
===========================================

El paper demuestra garantías de recuperación exacta bajo condiciones
específicas de margen, separación y, en algunos resultados, norma de
los patrones. Implementar entmax/normmax NO implica por sí solo que
esas condiciones se satisfagan con embeddings RAW del Experimento 2.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np


__all__ = [
    "HopfieldFenchelYoungNetwork",
    "softmax_fy",
    "sparsemax_fy",
    "entmax_bisect_fy",
    "normmax_bisect_fy",
    "hfy_recall_matrix",
    "run_hfy_self_checks",
]


ArrayLike = Union[np.ndarray, Sequence[float]]

_VALID_SCORING = frozenset(
    {
        "softmax",
        "entmax",
        "sparsemax",
        "normmax",
    }
)

_SPECIAL_CASE_ATOL = 1e-12
_SIMPLEX_ZERO_ATOL = 1e-15
_SIMPLEX_SUM_ATOL = 1e-10


# =============================================================================
# 1. Validación y utilidades numéricas
# =============================================================================


def _validate_real(
    value: Any,
    *,
    name: str,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} debe ser numérico, no booleano.")

    if not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} debe ser numérico.")

    number = float(value)

    if not np.isfinite(number):
        raise ValueError(f"{name} debe ser finito.")

    return number


def _validate_positive_integer(
    value: Any,
    *,
    name: str,
) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
    ):
        raise TypeError(f"{name} debe ser un entero.")

    integer = int(value)

    if integer < 1:
        raise ValueError(f"{name} debe ser >= 1.")

    return integer


def _validate_tolerance(
    value: Any,
    *,
    name: str,
) -> float:
    tolerance = _validate_real(
        value,
        name=name,
    )

    if tolerance <= 0.0:
        raise ValueError(f"{name} debe ser > 0.")

    return tolerance


def _normalize_scoring(scoring: Any) -> str:
    if not isinstance(scoring, str):
        raise TypeError("scoring debe ser una cadena.")

    normalized = scoring.strip().lower()

    if normalized not in _VALID_SCORING:
        raise ValueError(
            "scoring debe ser uno de "
            f"{sorted(_VALID_SCORING)}."
        )

    return normalized


def _validate_alpha(alpha: Any) -> float:
    alpha_value = _validate_real(
        alpha,
        name="alpha",
    )

    if alpha_value < 1.0:
        raise ValueError("alpha debe ser >= 1.")

    return alpha_value


def _validate_gamma(gamma: Any) -> float:
    gamma_value = _validate_real(
        gamma,
        name="gamma",
    )

    if gamma_value <= 1.0:
        raise ValueError("gamma debe ser > 1.")

    return gamma_value


def _validate_beta(
    beta: Any,
    *,
    allow_zero: bool,
) -> float:
    beta_value = _validate_real(
        beta,
        name="beta",
    )

    if allow_zero:
        if beta_value < 0.0:
            raise ValueError("beta debe ser >= 0.")
    else:
        if beta_value <= 0.0:
            raise ValueError("beta debe ser > 0.")

    return beta_value


def _vector(
    value: ArrayLike,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(
            value,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} debe contener valores numéricos."
        ) from exc

    if array.ndim != 1:
        raise ValueError(f"{name} debe ser 1D.")

    if array.size == 0:
        raise ValueError(f"{name} no puede estar vacío.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contiene NaN o Inf.")

    return np.asarray(
        array,
        dtype=np.float64,
    )


def _matrix(
    value: Any,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(
            value,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} debe contener valores numéricos."
        ) from exc

    if array.ndim != 2:
        raise ValueError(f"{name} debe ser una matriz 2D.")

    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(
            f"{name} no puede tener dimensiones vacías."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contiene NaN o Inf.")

    return np.asarray(
        array,
        dtype=np.float64,
    )


def _simplex_cleanup_columns(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Limpieza numérica para una matriz de distribuciones simplex.

    Convención:
        filas    -> memorias/categorías del scoring (N)
        columnas -> distribuciones independientes (Q)
    """
    p = np.array(
        probabilities,
        dtype=np.float64,
        copy=True,
    )

    if p.ndim != 2:
        raise ValueError(
            "probabilities debe ser una matriz 2D."
        )

    if not np.all(np.isfinite(p)):
        raise FloatingPointError(
            "La transformación Fenchel-Young produjo NaN o Inf."
        )

    p[np.abs(p) < _SIMPLEX_ZERO_ATOL] = 0.0
    p = np.maximum(p, 0.0)

    totals = np.sum(
        p,
        axis=0,
        keepdims=True,
    )

    if (
        not np.all(np.isfinite(totals))
        or np.any(totals <= 0.0)
    ):
        raise FloatingPointError(
            "Distribución Fenchel-Young degenerada."
        )

    p = p / totals

    if not np.allclose(
        np.sum(p, axis=0),
        1.0,
        rtol=0.0,
        atol=_SIMPLEX_SUM_ATOL,
    ):
        raise FloatingPointError(
            "La transformación Fenchel-Young no quedó en el simplex."
        )

    return p


def _simplex_cleanup(
    probabilities: np.ndarray,
) -> np.ndarray:
    p = _vector(
        probabilities,
        "probabilities",
    )

    return _simplex_cleanup_columns(
        p[:, np.newaxis]
    )[:, 0]


# =============================================================================
# 2. Kernels batch de separación Fenchel-Young
# =============================================================================


def _softmax_columns(
    theta: np.ndarray,
) -> np.ndarray:
    z = _matrix(
        theta,
        "theta",
    )

    shifted = (
        z
        - np.max(
            z,
            axis=0,
            keepdims=True,
        )
    )

    exp_values = np.exp(
        shifted
    )

    totals = np.sum(
        exp_values,
        axis=0,
        keepdims=True,
    )

    if (
        not np.all(np.isfinite(totals))
        or np.any(totals <= 0.0)
    ):
        raise FloatingPointError(
            "Softmax produjo una normalización inválida."
        )

    return exp_values / totals


def _sparsemax_columns(
    theta: np.ndarray,
) -> np.ndarray:
    z = _matrix(
        theta,
        "theta",
    )

    n_rows, n_columns = z.shape

    z_sorted = np.sort(
        z,
        axis=0,
    )[::-1, :]

    z_cumsum = np.cumsum(
        z_sorted,
        axis=0,
    )

    k = np.arange(
        1,
        n_rows + 1,
        dtype=np.float64,
    )[:, np.newaxis]

    support = (
        1.0
        + k * z_sorted
        > z_cumsum
    )

    support_size = np.sum(
        support,
        axis=0,
    ).astype(int)

    if np.any(support_size < 1):
        raise FloatingPointError(
            "No se pudo determinar el soporte sparsemax."
        )

    column_indices = np.arange(
        n_columns,
        dtype=int,
    )

    tau = (
        z_cumsum[
            support_size - 1,
            column_indices,
        ]
        - 1.0
    ) / support_size.astype(np.float64)

    probabilities = np.maximum(
        z
        - tau[np.newaxis, :],
        0.0,
    )

    return _simplex_cleanup_columns(
        probabilities
    )


def _entmax_bisect_columns(
    theta: np.ndarray,
    *,
    alpha: float,
    n_iter: int,
    tol: float,
) -> np.ndarray:
    z = _matrix(
        theta,
        "theta",
    )

    alpha_value = _validate_alpha(
        alpha
    )

    n_iter_value = _validate_positive_integer(
        n_iter,
        name="n_iter",
    )

    tol_value = _validate_tolerance(
        tol,
        name="tol",
    )

    if np.isclose(
        alpha_value,
        1.0,
        rtol=0.0,
        atol=_SPECIAL_CASE_ATOL,
    ):
        return _softmax_columns(
            z
        )

    if np.isclose(
        alpha_value,
        2.0,
        rtol=0.0,
        atol=_SPECIAL_CASE_ATOL,
    ):
        return _sparsemax_columns(
            z
        )

    # KKT para la Tsallis alpha-negentropy del paper:
    #
    #     p_i
    #       = [(alpha - 1)(theta_i - tau)]_+^(1/(alpha - 1))
    #
    # Para estabilidad definimos:
    #
    #     scaled_theta = (alpha - 1) * theta
    #     scaled_tau   = (alpha - 1) * tau
    #
    # y resolvemos:
    #
    #     p_i = [scaled_theta_i - scaled_tau]_+^power.
    #
    # El umbral escalado queda acotado por:
    #
    #     theta_max_scaled - 1
    #         <= scaled_tau
    #         <= theta_max_scaled - N^(1-alpha).
    #
    # Esta parametrización evita la inestabilidad numérica que
    # aparece cerca de alpha=1 cuando se bisecta directamente tau.

    alpha_minus_one = (
        alpha_value
        - 1.0
    )

    power = (
        1.0
        / alpha_minus_one
    )

    scaled_theta = (
        alpha_minus_one
        * z
    )

    n_rows, n_columns = (
        scaled_theta.shape
    )

    theta_max_scaled = np.max(
        scaled_theta,
        axis=0,
    )

    tau_lo = (
        theta_max_scaled
        - 1.0
    )

    tau_hi = (
        theta_max_scaled
        - float(
            n_rows ** (
                1.0
                - alpha_value
            )
        )
    )

    tau = 0.5 * (
        tau_lo
        + tau_hi
    )

    active = np.ones(
        n_columns,
        dtype=bool,
    )

    for _ in range(
        n_iter_value
    ):
        midpoint = 0.5 * (
            tau_lo
            + tau_hi
        )

        with np.errstate(
            over="raise",
            invalid="raise",
        ):
            try:
                positive = np.maximum(
                    scaled_theta
                    - midpoint[
                        np.newaxis,
                        :,
                    ],
                    0.0,
                )

                mass = np.sum(
                    positive ** power,
                    axis=0,
                )

            except FloatingPointError as exc:
                raise FloatingPointError(
                    "Overflow/operación inválida durante la "
                    "bisección alpha-entmax."
                ) from exc

        tau[
            active
        ] = midpoint[
            active
        ]

        converged = (
            np.abs(
                mass
                - 1.0
            )
            <= tol_value
        )

        newly_converged = (
            active
            & converged
        )

        active[
            newly_converged
        ] = False

        if not np.any(
            active
        ):
            break

        # La masa decrece monótonamente al aumentar tau.
        lower_update = (
            active
            & (
                mass
                >= 1.0
            )
        )

        upper_update = (
            active
            & (
                mass
                < 1.0
            )
        )

        tau_lo[
            lower_update
        ] = midpoint[
            lower_update
        ]

        tau_hi[
            upper_update
        ] = midpoint[
            upper_update
        ]

    with np.errstate(
        over="raise",
        invalid="raise",
    ):
        try:
            probabilities = np.maximum(
                scaled_theta
                - tau[
                    np.newaxis,
                    :,
                ],
                0.0,
            ) ** power

        except FloatingPointError as exc:
            raise FloatingPointError(
                "Overflow/operación inválida al reconstruir "
                "alpha-entmax."
            ) from exc

    return _simplex_cleanup_columns(
        probabilities
    )


def _normmax_bisect_columns(
    theta: np.ndarray,
    *,
    gamma: float,
    n_iter: int,
    tol: float,
) -> np.ndarray:
    z = _matrix(
        theta,
        "theta",
    )

    gamma_value = _validate_gamma(
        gamma
    )

    n_iter_value = _validate_positive_integer(
        n_iter,
        name="n_iter",
    )

    tol_value = _validate_tolerance(
        tol,
        name="tol",
    )

    n_rows, n_columns = z.shape

    theta_max = np.max(
        z,
        axis=0,
    )

    # Ecuación (33) / Algoritmo 4 del paper.
    mu_min = (
        theta_max
        - 1.0
    )

    mu_max = (
        theta_max
        - float(
            n_rows ** (
                1.0
                - gamma_value
            )
        )
    )

    z_power = (
        gamma_value
        / (
            gamma_value
            - 1.0
        )
    )

    p_power = (
        1.0
        / (
            gamma_value
            - 1.0
        )
    )

    mu = 0.5 * (
        mu_min
        + mu_max
    )

    active = np.ones(
        n_columns,
        dtype=bool,
    )

    for _ in range(
        n_iter_value
    ):
        midpoint = 0.5 * (
            mu_min
            + mu_max
        )

        with np.errstate(
            over="raise",
            invalid="raise",
        ):
            try:
                positive = np.maximum(
                    z
                    - midpoint[
                        np.newaxis,
                        :,
                    ],
                    0.0,
                )

                mass = np.sum(
                    positive ** z_power,
                    axis=0,
                )

            except FloatingPointError as exc:
                raise FloatingPointError(
                    "Overflow/operación inválida durante la "
                    "bisección gamma-normmax."
                ) from exc

        mu[
            active
        ] = midpoint[
            active
        ]

        converged = (
            np.abs(
                mass
                - 1.0
            )
            <= tol_value
        )

        newly_converged = (
            active
            & converged
        )

        active[
            newly_converged
        ] = False

        if not np.any(
            active
        ):
            break

        # Algoritmo 4:
        # if Z < 1: mu_max <- mu
        # else:     mu_min <- mu
        lower_update = (
            active
            & (
                mass
                >= 1.0
            )
        )

        upper_update = (
            active
            & (
                mass
                < 1.0
            )
        )

        mu_min[
            lower_update
        ] = midpoint[
            lower_update
        ]

        mu_max[
            upper_update
        ] = midpoint[
            upper_update
        ]

    with np.errstate(
        over="raise",
        invalid="raise",
    ):
        try:
            numerator = np.maximum(
                z
                - mu[
                    np.newaxis,
                    :,
                ],
                0.0,
            ) ** p_power

        except FloatingPointError as exc:
            raise FloatingPointError(
                "Overflow/operación inválida al reconstruir "
                "gamma-normmax."
            ) from exc

    return _simplex_cleanup_columns(
        numerator
    )


# =============================================================================
# 3. Transformaciones públicas 1D
# =============================================================================


def softmax_fy(
    theta: ArrayLike,
) -> np.ndarray:
    """
    Softmax / 1-entmax inducido por negentropía de Shannon.
    """
    z = _vector(
        theta,
        "theta",
    )

    return _softmax_columns(
        z[:, np.newaxis]
    )[:, 0]


def sparsemax_fy(
    theta: ArrayLike,
) -> np.ndarray:
    """
    Sparsemax / 2-entmax, proyección euclidiana al simplex.
    """
    z = _vector(
        theta,
        "theta",
    )

    return _sparsemax_columns(
        z[:, np.newaxis]
    )[:, 0]


def entmax_bisect_fy(
    theta: ArrayLike,
    alpha: float = 1.5,
    n_iter: int = 100,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    alpha-entmax para la Tsallis alpha-negentropy de la Eq. (17).

    La regularización utilizada por el paper es:

        Omega_T_alpha(p)
            = (-1 + sum_i p_i^alpha) / [alpha(alpha - 1)]
              + I_Delta(p),            alpha != 1

    con los casos especiales:

        alpha = 1 -> softmax
        alpha = 2 -> sparsemax

    Para los demás alpha se resuelve el umbral mediante bisección.
    """
    z = _vector(
        theta,
        "theta",
    )

    return _entmax_bisect_columns(
        z[:, np.newaxis],
        alpha=alpha,
        n_iter=n_iter,
        tol=tol,
    )[:, 0]


def normmax_bisect_fy(
    theta: ArrayLike,
    gamma: float = 2.0,
    n_iter: int = 100,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    gamma-normmax según Eq. (18) y Algoritmo 4 del Apéndice A.

        Omega_N_gamma(p)
            = -1 + ||p||_gamma + I_Delta(p)

    con gamma > 1.
    """
    z = _vector(
        theta,
        "theta",
    )

    return _normmax_bisect_columns(
        z[:, np.newaxis],
        gamma=gamma,
        n_iter=n_iter,
        tol=tol,
    )[:, 0]


# =============================================================================
# 4. Selección de transformación y energía
# =============================================================================


def _transform_vector(
    logits: np.ndarray,
    *,
    scoring: str,
    alpha: float,
    gamma: float,
    bisection_iterations: int,
    bisection_tolerance: float,
) -> np.ndarray:
    scoring_value = _normalize_scoring(
        scoring
    )

    if scoring_value == "softmax":
        return softmax_fy(
            logits
        )

    if scoring_value == "sparsemax":
        return sparsemax_fy(
            logits
        )

    if scoring_value == "entmax":
        return entmax_bisect_fy(
            logits,
            alpha=alpha,
            n_iter=bisection_iterations,
            tol=bisection_tolerance,
        )

    if scoring_value == "normmax":
        return normmax_bisect_fy(
            logits,
            gamma=gamma,
            n_iter=bisection_iterations,
            tol=bisection_tolerance,
        )

    raise RuntimeError(
        "scoring interno inválido."
    )


def _transform_columns(
    logits: np.ndarray,
    *,
    scoring: str,
    alpha: float,
    gamma: float,
    bisection_iterations: int,
    bisection_tolerance: float,
) -> np.ndarray:
    scoring_value = _normalize_scoring(
        scoring
    )

    if scoring_value == "softmax":
        return _softmax_columns(
            logits
        )

    if scoring_value == "sparsemax":
        return _sparsemax_columns(
            logits
        )

    if scoring_value == "entmax":
        return _entmax_bisect_columns(
            logits,
            alpha=alpha,
            n_iter=bisection_iterations,
            tol=bisection_tolerance,
        )

    if scoring_value == "normmax":
        return _normmax_bisect_columns(
            logits,
            gamma=gamma,
            n_iter=bisection_iterations,
            tol=bisection_tolerance,
        )

    raise RuntimeError(
        "scoring interno inválido."
    )


def _omega_value(
    probabilities: np.ndarray,
    *,
    scoring: str,
    alpha: float,
    gamma: float,
) -> float:
    """
    Evalúa Omega(p) dentro del simplex para el scoring seleccionado.

    Se utiliza únicamente para diagnóstico de energía.
    """
    p = _vector(
        probabilities,
        "probabilities",
    )

    if np.any(
        p
        < -_SIMPLEX_SUM_ATOL
    ):
        raise ValueError(
            "probabilities contiene valores negativos."
        )

    if not np.isclose(
        np.sum(p),
        1.0,
        rtol=0.0,
        atol=_SIMPLEX_SUM_ATOL,
    ):
        raise ValueError(
            "probabilities no suma 1."
        )

    scoring_value = _normalize_scoring(
        scoring
    )

    if scoring_value == "softmax":
        nonzero = (
            p
            > 0.0
        )

        return float(
            np.sum(
                p[nonzero]
                * np.log(
                    p[nonzero]
                )
            )
        )

    if scoring_value in {
        "entmax",
        "sparsemax",
    }:
        alpha_value = (
            2.0
            if scoring_value == "sparsemax"
            else _validate_alpha(alpha)
        )

        if np.isclose(
            alpha_value,
            1.0,
            rtol=0.0,
            atol=_SPECIAL_CASE_ATOL,
        ):
            nonzero = (
                p
                > 0.0
            )

            return float(
                np.sum(
                    p[nonzero]
                    * np.log(
                        p[nonzero]
                    )
                )
            )

        return float(
            (
                -1.0
                + np.sum(
                    p ** alpha_value
                )
            )
            / (
                alpha_value
                * (
                    alpha_value
                    - 1.0
                )
            )
        )

    if scoring_value == "normmax":
        gamma_value = _validate_gamma(
            gamma
        )

        return float(
            -1.0
            + np.linalg.norm(
                p,
                ord=gamma_value,
            )
        )

    raise RuntimeError(
        "scoring interno inválido."
    )


# =============================================================================
# 5. Implementación individual
# =============================================================================


class HopfieldFenchelYoungNetwork:
    """
    Red HFY probabilística/sparse con Psi(q)=1/2||q||^2.

    Contrato individual compatible con AssociativeMemorySpec:

        train(patterns_matrix: (D, N))
        recall(query: (D,), beta, iterations) -> (D,)

    scoring:
        "softmax"   -> Shannon / 1-entmax
        "entmax"    -> Tsallis alpha-entmax
        "sparsemax" -> 2-entmax
        "normmax"   -> gamma-normmax

    store_diagnostics=False es el valor recomendado para las corridas
    masivas del Experimento 2. Cuando es False, recall() NO calcula
    energía antes/después, evitando trabajo redundante que no modifica
    el estado recuperado.
    """

    VALID_SCORING = _VALID_SCORING

    def __init__(
        self,
        scoring: str = "entmax",
        alpha: float = 1.5,
        gamma: float = 2.0,
        bisection_iterations: int = 100,
        bisection_tolerance: float = 1e-12,
        store_diagnostics: bool = False,
    ) -> None:
        self.scoring = _normalize_scoring(
            scoring
        )

        self.alpha = _validate_alpha(
            alpha
        )

        self.gamma = _validate_gamma(
            gamma
        )

        self.bisection_iterations = (
            _validate_positive_integer(
                bisection_iterations,
                name="bisection_iterations",
            )
        )

        self.bisection_tolerance = (
            _validate_tolerance(
                bisection_tolerance,
                name="bisection_tolerance",
            )
        )

        if not isinstance(
            store_diagnostics,
            (bool, np.bool_),
        ):
            raise TypeError(
                "store_diagnostics debe ser booleano."
            )

        self.store_diagnostics = bool(
            store_diagnostics
        )

        self.X: Optional[
            np.ndarray
        ] = None

        self.XT: Optional[
            np.ndarray
        ] = None

        self._dim: Optional[
            int
        ] = None

        self._n_patterns: Optional[
            int
        ] = None

        self._clear_diagnostics()

    def _clear_diagnostics(
        self,
    ) -> None:
        self._last_weights: Optional[
            np.ndarray
        ] = None

        self._last_scores: Optional[
            np.ndarray
        ] = None

        self._last_logits: Optional[
            np.ndarray
        ] = None

        self._last_energy_before: Optional[
            float
        ] = None

        self._last_energy_after: Optional[
            float
        ] = None

    def train(
        self,
        patterns: np.ndarray,
    ) -> None:
        """
        Almacena memorias como columnas.

            patterns.shape == (embedding_dim, n_patterns)
        """
        matrix = _matrix(
            patterns,
            "patterns",
        )

        stored = np.array(
            matrix,
            dtype=np.float64,
            copy=True,
            order="C",
        )

        # Evita mutaciones accidentales del conjunto de memorias a
        # través de los atributos públicos X / XT.
        stored.setflags(
            write=False
        )

        self.X = stored
        self.XT = self.X.T
        self._dim = int(
            self.X.shape[0]
        )
        self._n_patterns = int(
            self.X.shape[1]
        )

        self._clear_diagnostics()

    def _require_trained(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        int,
    ]:
        if (
            self.X is None
            or self.XT is None
            or self._dim is None
        ):
            raise RuntimeError(
                "Llama a train() antes de recall() o energy()."
            )

        return (
            self.X,
            self.XT,
            self._dim,
        )

    def _transform(
        self,
        logits: np.ndarray,
    ) -> np.ndarray:
        return _transform_vector(
            logits,
            scoring=self.scoring,
            alpha=self.alpha,
            gamma=self.gamma,
            bisection_iterations=(
                self.bisection_iterations
            ),
            bisection_tolerance=(
                self.bisection_tolerance
            ),
        )

    def energy(
        self,
        query: ArrayLike,
        beta: float,
    ) -> float:
        """
        Energía HFY hasta una constante aditiva:

            E(q)
                = -(1/beta) Omega*(beta * X.T @ q)
                  + 1/2 ||q||^2

        para la convención de memorias por columnas.

        beta debe ser > 0 porque esta expresión contiene 1/beta.
        """
        (
            _patterns,
            patterns_t,
            dimension,
        ) = self._require_trained()

        beta_value = _validate_beta(
            beta,
            allow_zero=False,
        )

        q = _vector(
            query,
            "query",
        )

        if q.size != dimension:
            raise ValueError(
                "query tiene dimensionalidad incompatible: "
                f"{q.size}; esperada={dimension}."
            )

        scores = (
            patterns_t
            @ q
        )

        theta = (
            beta_value
            * scores
        )

        probabilities = self._transform(
            theta
        )

        omega_star = float(
            np.dot(
                theta,
                probabilities,
            )
            - _omega_value(
                probabilities,
                scoring=self.scoring,
                alpha=self.alpha,
                gamma=self.gamma,
            )
        )

        return float(
            -(
                omega_star
                / beta_value
            )
            + 0.5
            * np.dot(
                q,
                q,
            )
        )

    def recall(
        self,
        query: ArrayLike,
        beta: float,
        iterations: int = 1,
    ) -> np.ndarray:
        """
        Ejecuta la actualización sparse-HFY (Eq. 21) adaptada a
        memorias por columnas:

            q_next
                = X @ y_hat_Omega(
                    beta * (X.T @ q)
                  )

        No se aplica ningún escalamiento adicional.

        beta=0 se admite como límite computacional del eje de ganancia:
        el scoring recibe logits nulos. La función energy(), en cambio,
        exige beta>0 por contener 1/beta.
        """
        (
            patterns,
            patterns_t,
            dimension,
        ) = self._require_trained()

        beta_value = _validate_beta(
            beta,
            allow_zero=True,
        )

        iterations_value = (
            _validate_positive_integer(
                iterations,
                name="iterations",
            )
        )

        q = np.array(
            _vector(
                query,
                "query",
            ),
            dtype=np.float64,
            copy=True,
        )

        if q.size != dimension:
            raise ValueError(
                "query tiene dimensionalidad incompatible: "
                f"{q.size}; esperada={dimension}."
            )

        for _ in range(
            iterations_value
        ):
            # -----------------------------------------------------
            # Diagnóstico opcional: no forma parte de la dinámica.
            # -----------------------------------------------------

            if (
                self.store_diagnostics
                and beta_value > 0.0
            ):
                energy_before = self.energy(
                    q,
                    beta_value,
                )
            else:
                energy_before = None

            scores = (
                patterns_t
                @ q
            )

            logits = (
                beta_value
                * scores
            )

            weights = self._transform(
                logits
            )

            q_next = (
                patterns
                @ weights
            )

            q_next = np.asarray(
                q_next,
                dtype=np.float64,
            )

            if q_next.shape != (
                dimension,
            ):
                raise RuntimeError(
                    "El estado HFY recuperado tiene forma "
                    f"inválida: {q_next.shape}; "
                    f"esperada={(dimension,)}."
                )

            if not np.all(
                np.isfinite(
                    q_next
                )
            ):
                raise FloatingPointError(
                    "El estado HFY recuperado contiene NaN o Inf."
                )

            if self.store_diagnostics:
                if beta_value > 0.0:
                    energy_after = self.energy(
                        q_next,
                        beta_value,
                    )
                else:
                    energy_after = None

                self._last_scores = np.array(
                    scores,
                    copy=True,
                )

                self._last_logits = np.array(
                    logits,
                    copy=True,
                )

                self._last_weights = np.array(
                    weights,
                    copy=True,
                )

                self._last_energy_before = (
                    None
                    if energy_before is None
                    else float(
                        energy_before
                    )
                )

                self._last_energy_after = (
                    None
                    if energy_after is None
                    else float(
                        energy_after
                    )
                )

            q = q_next

        return np.array(
            q,
            dtype=np.float64,
            copy=True,
        )

    def last_weights(
        self,
    ) -> Optional[np.ndarray]:
        return (
            None
            if self._last_weights is None
            else np.array(
                self._last_weights,
                copy=True,
            )
        )

    def last_diagnostics(
        self,
    ) -> Dict[str, Any]:
        return {
            "scoring":
                self.scoring,

            "alpha":
                float(
                    self.alpha
                ),

            "gamma":
                float(
                    self.gamma
                ),

            "store_diagnostics":
                bool(
                    self.store_diagnostics
                ),

            "scores_raw": (
                None
                if self._last_scores is None
                else np.array(
                    self._last_scores,
                    copy=True,
                )
            ),

            "logits": (
                None
                if self._last_logits is None
                else np.array(
                    self._last_logits,
                    copy=True,
                )
            ),

            "weights": (
                None
                if self._last_weights is None
                else np.array(
                    self._last_weights,
                    copy=True,
                )
            ),

            "support_size": (
                None
                if self._last_weights is None
                else int(
                    np.count_nonzero(
                        self._last_weights
                        > 0.0
                    )
                )
            ),

            "energy_before":
                self._last_energy_before,

            "energy_after":
                self._last_energy_after,
        }


# =============================================================================
# 6. Adaptador batch/vectorizado para AssociativeMemorySpec
# =============================================================================


def hfy_recall_matrix(
    *,
    patterns_matrix: np.ndarray,
    query_matrix: np.ndarray,
    beta: float,
    iterations: int,
    scoring: str = "entmax",
    alpha: float = 1.5,
    gamma: float = 2.0,
    bisection_iterations: int = 100,
    bisection_tolerance: float = 1e-12,
) -> np.ndarray:
    """
    Recall HFY batch/vectorizado compatible con el contrato de
    AssociativeMemorySpec de exp2_pipeline_integrated.py.

    Entrada
    -------
    patterns_matrix:
        shape == (embedding_dim, n_patterns)

    query_matrix:
        shape == (embedding_dim, n_queries)

    Salida
    ------
    np.ndarray:
        shape == (n_queries, embedding_dim)

    Dinámica
    --------

        state_(t+1)
            = patterns_matrix
              @ y_hat_Omega(
                    beta
                    * patterns_matrix.T
                    @ state_t
                )

    y_hat_Omega se aplica independientemente a cada columna/query.

    No se normalizan los embeddings ni se añade ninguna escala a beta.
    """
    patterns = _matrix(
        patterns_matrix,
        "patterns_matrix",
    )

    queries = _matrix(
        query_matrix,
        "query_matrix",
    )

    if (
        patterns.shape[0]
        != queries.shape[0]
    ):
        raise ValueError(
            "patterns_matrix y query_matrix deben compartir "
            "embedding_dim. "
            f"Obtenido: {patterns.shape[0]} y {queries.shape[0]}."
        )

    beta_value = _validate_beta(
        beta,
        allow_zero=True,
    )

    iterations_value = (
        _validate_positive_integer(
            iterations,
            name="iterations",
        )
    )

    scoring_value = _normalize_scoring(
        scoring
    )

    alpha_value = _validate_alpha(
        alpha
    )

    gamma_value = _validate_gamma(
        gamma
    )

    bisection_iterations_value = (
        _validate_positive_integer(
            bisection_iterations,
            name="bisection_iterations",
        )
    )

    bisection_tolerance_value = (
        _validate_tolerance(
            bisection_tolerance,
            name="bisection_tolerance",
        )
    )

    # Copia para que iterations>1 nunca modifique query_matrix del
    # caller ni dependa de su layout de memoria.
    state = np.array(
        queries,
        dtype=np.float64,
        copy=True,
        order="C",
    )

    patterns_t = patterns.T

    for _ in range(
        iterations_value
    ):
        scores = (
            patterns_t
            @ state
        )

        logits = (
            beta_value
            * scores
        )

        weights = _transform_columns(
            logits,
            scoring=scoring_value,
            alpha=alpha_value,
            gamma=gamma_value,
            bisection_iterations=(
                bisection_iterations_value
            ),
            bisection_tolerance=(
                bisection_tolerance_value
            ),
        )

        state = (
            patterns
            @ weights
        )

        state = np.asarray(
            state,
            dtype=np.float64,
        )

        if state.shape != queries.shape:
            raise RuntimeError(
                "La ruta HFY vectorizada produjo una forma "
                f"inválida: {state.shape}; "
                f"esperada={queries.shape}."
            )

        if not np.all(
            np.isfinite(
                state
            )
        ):
            raise FloatingPointError(
                "El recall HFY vectorizado produjo NaN o Inf."
            )

    # El pipeline cerrado espera observaciones por filas.
    recalled = np.asarray(
        state.T,
        dtype=np.float64,
    )

    expected_shape = (
        queries.shape[1],
        patterns.shape[0],
    )

    if recalled.shape != expected_shape:
        raise RuntimeError(
            "La salida HFY vectorizada tiene forma inesperada: "
            f"{recalled.shape}; esperada={expected_shape}."
        )

    return np.array(
        recalled,
        dtype=np.float64,
        copy=True,
    )


# =============================================================================
# 7. Self-checks reproducibles
# =============================================================================


def run_hfy_self_checks(
    seed: int = 123,
) -> Dict[str, Any]:
    """
    Ejecuta checks internos de identidad matemática y equivalencia
    individual/batch.

    No sustituye la regresión experimental del notebook, pero detecta
    errores locales en la implementación HFY.
    """
    seed_value = _validate_positive_integer(
        seed,
        name="seed",
    )

    rng = np.random.default_rng(
        seed_value
    )

    theta = rng.normal(
        size=17
    )

    alpha1 = entmax_bisect_fy(
        theta,
        alpha=1.0,
    )

    softmax = softmax_fy(
        theta
    )

    alpha2 = entmax_bisect_fy(
        theta,
        alpha=2.0,
    )

    sparsemax = sparsemax_fy(
        theta
    )

    alpha15 = entmax_bisect_fy(
        theta,
        alpha=1.5,
    )

    normmax2 = normmax_bisect_fy(
        theta,
        gamma=2.0,
    )

    translation = 13.75

    translation_checks = {
        "softmax_translation_invariant": bool(
            np.allclose(
                softmax_fy(theta),
                softmax_fy(
                    theta
                    + translation
                ),
                rtol=1e-12,
                atol=1e-12,
            )
        ),

        "entmax_translation_invariant": bool(
            np.allclose(
                entmax_bisect_fy(
                    theta,
                    alpha=1.5,
                ),
                entmax_bisect_fy(
                    theta
                    + translation,
                    alpha=1.5,
                ),
                rtol=1e-10,
                atol=1e-11,
            )
        ),

        "sparsemax_translation_invariant": bool(
            np.allclose(
                sparsemax_fy(theta),
                sparsemax_fy(
                    theta
                    + translation
                ),
                rtol=1e-12,
                atol=1e-12,
            )
        ),

        "normmax_translation_invariant": bool(
            np.allclose(
                normmax_bisect_fy(
                    theta,
                    gamma=2.0,
                ),
                normmax_bisect_fy(
                    theta
                    + translation,
                    gamma=2.0,
                ),
                rtol=1e-10,
                atol=1e-11,
            )
        ),
    }

    transform_checks = {
        "alpha1_equals_softmax": bool(
            np.allclose(
                alpha1,
                softmax,
                rtol=1e-12,
                atol=1e-12,
            )
        ),

        "alpha2_equals_sparsemax": bool(
            np.allclose(
                alpha2,
                sparsemax,
                rtol=1e-12,
                atol=1e-12,
            )
        ),

        "alpha15_simplex": bool(
            np.all(
                alpha15
                >= 0.0
            )
            and np.isclose(
                np.sum(alpha15),
                1.0,
                rtol=0.0,
                atol=1e-10,
            )
        ),

        "normmax2_simplex": bool(
            np.all(
                normmax2
                >= 0.0
            )
            and np.isclose(
                np.sum(normmax2),
                1.0,
                rtol=0.0,
                atol=1e-10,
            )
        ),

        "alpha15_support_size": int(
            np.count_nonzero(
                alpha15
                > 0.0
            )
        ),

        "normmax2_support_size": int(
            np.count_nonzero(
                normmax2
                > 0.0
            )
        ),
    }

    # ------------------------------------------------------------
    # Equivalencia individual <-> batch
    # ------------------------------------------------------------

    dimension = 9
    n_patterns = 13
    n_queries = 7

    patterns = rng.normal(
        size=(
            dimension,
            n_patterns,
        )
    )

    queries = rng.normal(
        size=(
            dimension,
            n_queries,
        )
    )

    variants = {
        "softmax": {
            "scoring": "softmax",
        },
        "entmax_a15": {
            "scoring": "entmax",
            "alpha": 1.5,
        },
        "sparsemax": {
            "scoring": "sparsemax",
        },
        "normmax_g2": {
            "scoring": "normmax",
            "gamma": 2.0,
        },
    }

    recall_equivalence: Dict[
        str,
        Any,
    ] = {}

    for (
        variant_name,
        variant_kwargs,
    ) in variants.items():
        constructor_kwargs = {
            "scoring": variant_kwargs[
                "scoring"
            ],
            "alpha": float(
                variant_kwargs.get(
                    "alpha",
                    1.5,
                )
            ),
            "gamma": float(
                variant_kwargs.get(
                    "gamma",
                    2.0,
                )
            ),
            "bisection_iterations": 100,
            "bisection_tolerance": 1e-12,
            "store_diagnostics": False,
        }

        memory = HopfieldFenchelYoungNetwork(
            **constructor_kwargs
        )

        memory.train(
            patterns
        )

        max_difference = 0.0
        equivalent = True

        for beta_value in (
            0.0,
            0.01,
            0.25,
            0.495,
        ):
            for iterations_value in (
                1,
                2,
            ):
                batch = hfy_recall_matrix(
                    patterns_matrix=patterns,
                    query_matrix=queries,
                    beta=beta_value,
                    iterations=iterations_value,
                    scoring=(
                        constructor_kwargs[
                            "scoring"
                        ]
                    ),
                    alpha=(
                        constructor_kwargs[
                            "alpha"
                        ]
                    ),
                    gamma=(
                        constructor_kwargs[
                            "gamma"
                        ]
                    ),
                    bisection_iterations=(
                        constructor_kwargs[
                            "bisection_iterations"
                        ]
                    ),
                    bisection_tolerance=(
                        constructor_kwargs[
                            "bisection_tolerance"
                        ]
                    ),
                )

                individual = np.vstack(
                    [
                        memory.recall(
                            query=(
                                queries[
                                    :,
                                    query_index,
                                ]
                            ),
                            beta=beta_value,
                            iterations=(
                                iterations_value
                            ),
                        )
                        for query_index
                        in range(
                            n_queries
                        )
                    ]
                )

                difference = float(
                    np.max(
                        np.abs(
                            batch
                            - individual
                        )
                    )
                )

                max_difference = max(
                    max_difference,
                    difference,
                )

                equivalent = bool(
                    equivalent
                    and np.allclose(
                        batch,
                        individual,
                        rtol=1e-10,
                        atol=1e-11,
                    )
                )

        recall_equivalence[
            variant_name
        ] = {
            "equivalent":
                equivalent,

            "max_abs_difference":
                max_difference,
        }

    # beta=0 debe producir el promedio uniforme de las memorias
    # para todos los generalized-negentropy scorings implementados.
    expected_beta_zero = np.mean(
        patterns,
        axis=1,
    )

    beta_zero_checks: Dict[
        str,
        bool,
    ] = {}

    for (
        variant_name,
        variant_kwargs,
    ) in variants.items():
        beta_zero_output = hfy_recall_matrix(
            patterns_matrix=patterns,
            query_matrix=queries,
            beta=0.0,
            iterations=1,
            scoring=variant_kwargs[
                "scoring"
            ],
            alpha=float(
                variant_kwargs.get(
                    "alpha",
                    1.5,
                )
            ),
            gamma=float(
                variant_kwargs.get(
                    "gamma",
                    2.0,
                )
            ),
        )

        beta_zero_checks[
            variant_name
        ] = bool(
            np.allclose(
                beta_zero_output,
                np.tile(
                    expected_beta_zero,
                    (
                        n_queries,
                        1,
                    ),
                ),
                rtol=1e-10,
                atol=1e-11,
            )
        )

    all_recall_equivalent = all(
        bool(
            value[
                "equivalent"
            ]
        )
        for value
        in recall_equivalence.values()
    )

    all_beta_zero_correct = all(
        beta_zero_checks.values()
    )

    all_translation_invariant = all(
        translation_checks.values()
    )

    all_transform_checks = bool(
        transform_checks[
            "alpha1_equals_softmax"
        ]
        and transform_checks[
            "alpha2_equals_sparsemax"
        ]
        and transform_checks[
            "alpha15_simplex"
        ]
        and transform_checks[
            "normmax2_simplex"
        ]
    )

    return {
        "transform_checks":
            transform_checks,

        "translation_checks":
            translation_checks,

        "recall_equivalence":
            recall_equivalence,

        "beta_zero_checks":
            beta_zero_checks,

        "all_transform_checks":
            all_transform_checks,

        "all_translation_invariant":
            bool(
                all_translation_invariant
            ),

        "all_recall_equivalent":
            bool(
                all_recall_equivalent
            ),

        "all_beta_zero_correct":
            bool(
                all_beta_zero_correct
            ),

        "all_passed": bool(
            all_transform_checks
            and all_translation_invariant
            and all_recall_equivalent
            and all_beta_zero_correct
        ),
    }


if __name__ == "__main__":
    checks = run_hfy_self_checks()
    print(checks)
