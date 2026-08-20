"""Source-clustered statistics for the QAD distortion-repair probe."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from scipy import stats

from quantization_distortion_probe.statistics import CLASS_NAMES, stage_for_layer


METHODS = ("qad", "ste")


def confusion_metrics(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    diagonal = np.diag(matrix)
    union = matrix.sum(0) + matrix.sum(1) - diagonal
    iou = np.divide(
        diagonal,
        union,
        out=np.full_like(diagonal, np.nan),
        where=union != 0,
    )
    return {
        "miou": float(np.nanmean(iou)),
        "pixel_accuracy": float(diagonal.sum() / matrix.sum()),
        "per_class_iou": {
            name: float(value) for name, value in zip(CLASS_NAMES, iou)
        },
    }


def percentile_ci(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return [float("nan"), float("nan")]
    low, high = np.percentile(finite, [2.5, 97.5])
    return [float(low), float(high)]


def _bootstrap_mean(
    values: np.ndarray, rng: np.random.Generator, replicates: int
) -> np.ndarray:
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    return values[indices].mean(1)


def _exact_sign_p(delta: np.ndarray) -> float:
    nonzero = delta[delta != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    k = min(int((nonzero > 0).sum()), int((nonzero < 0).sum()))
    tail = sum(math.comb(n, index) for index in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def paired_summary(
    qad: np.ndarray,
    ste: np.ndarray,
    *,
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, Any]:
    qad = np.asarray(qad, dtype=np.float64)
    ste = np.asarray(ste, dtype=np.float64)
    delta = qad - ste
    indices = rng.integers(0, len(delta), size=(replicates, len(delta)))
    boot_delta = delta[indices].mean(1)
    boot_qad = qad[indices].mean(1)
    boot_ste = ste[indices].mean(1)
    relative_reduction = np.divide(
        boot_ste - boot_qad,
        boot_ste,
        out=np.full_like(boot_ste, np.nan),
        where=boot_ste != 0,
    )
    standard_deviation = delta.std(ddof=1) if len(delta) > 1 else 0.0
    return {
        "qad_mean": float(qad.mean()),
        "ste_mean": float(ste.mean()),
        "qad_minus_ste": float(delta.mean()),
        "qad_minus_ste_cluster_bootstrap_95ci": percentile_ci(boot_delta),
        "relative_reduction_fraction": float(
            (ste.mean() - qad.mean()) / ste.mean()
        ) if ste.mean() != 0 else float("nan"),
        "relative_reduction_cluster_bootstrap_95ci": percentile_ci(
            relative_reduction
        ),
        "paired_effect_dz": float(delta.mean() / standard_deviation)
        if standard_deviation > 0
        else float("inf"),
        "qad_lower_source_count": int((delta < 0).sum()),
        "qad_higher_source_count": int((delta > 0).sum()),
        "equal_source_count": int((delta == 0).sum()),
        "source_count": int(len(delta)),
        "two_sided_exact_sign_p": _exact_sign_p(delta),
    }


def _qif_rate(row: dict[str, Any], numerator: str) -> float:
    return float(row[numerator]) / float(row["count"])


def _source_qif_metrics(
    rows: Iterable[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], dict[str, float]],
    dict[tuple[str, str, str], dict[str, float]],
]:
    source_layers: dict[tuple[str, str, str], dict[str, float]] = {}
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    grouped_raw: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metrics = {
            "disagreement": _qif_rate(row, "disagreement_count"),
            "mae": _qif_rate(row, "abs_error_sum"),
            "signed_error": _qif_rate(row, "signed_error_sum"),
            "zero_delta": (
                float(row["variant_zero_count"])
                - float(row["reference_zero_count"])
            )
            / float(row["count"]),
            "saturation_delta": (
                float(row["variant_saturation_count"])
                - float(row["reference_saturation_count"])
            )
            / float(row["count"]),
        }
        metrics["abs_signed_error"] = abs(metrics["signed_error"])
        metrics["abs_zero_delta"] = abs(metrics["zero_delta"])
        metrics["abs_saturation_delta"] = abs(metrics["saturation_delta"])
        key = (str(row["method"]), str(row["source_id"]), str(row["layer"]))
        source_layers[key] = metrics
        grouped[key[:2]].append(metrics)
        grouped_raw[key[:2]].append(row)
    source_metrics = {
        key: {
            metric: float(np.mean([row[metric] for row in values]))
            for metric in values[0]
        }
        for key, values in grouped.items()
    }
    for key, values in grouped_raw.items():
        count = sum(float(row["count"]) for row in values)
        source_metrics[key]["micro_disagreement"] = sum(
            float(row["disagreement_count"]) for row in values
        ) / count
        source_metrics[key]["micro_mae"] = sum(
            float(row["abs_error_sum"]) for row in values
        ) / count
    return source_metrics, source_layers


def _source_feature_metrics(
    rows: Iterable[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], dict[str, float]],
    dict[tuple[str, str, int], dict[str, float]],
]:
    source_features: dict[tuple[str, str, int], dict[str, float]] = {}
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        teacher_sq = float(row["reference_sq_sum"])
        student_sq = float(row["variant_sq_sum"])
        dot = float(row["dot_sum"])
        metrics = {
            "mse": float(row["sq_error_sum"]) / float(row["count"]),
            "normalized_mse": float(row["sq_error_sum"]) / max(teacher_sq, 1e-12),
            "cosine_distance": 1.0
            - dot / max(math.sqrt(teacher_sq * student_sq), 1e-12),
            "mae": float(row["abs_error_sum"]) / float(row["count"]),
        }
        key = (
            str(row["method"]),
            str(row["source_id"]),
            int(row["feature_index"]),
        )
        source_features[key] = metrics
        grouped[key[:2]].append(metrics)
    source_metrics = {
        key: {
            metric: float(np.mean([row[metric] for row in values]))
            for metric in values[0]
        }
        for key, values in grouped.items()
    }
    return source_metrics, source_features


def _spearman_summary(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, Any]:
    observed = stats.spearmanr(x, y)
    indices = rng.integers(0, len(x), size=(replicates, len(x)))
    boot = np.empty(replicates, dtype=np.float64)
    for index, sampled in enumerate(indices):
        boot[index] = stats.spearmanr(x[sampled], y[sampled]).statistic
    return {
        "coefficient": float(observed.statistic),
        "p_value": float(observed.pvalue),
        "cluster_bootstrap_95ci": percentile_ci(boot),
        "source_count": int(len(x)),
    }


def summarize_repair(
    *,
    qif_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    confusion_rows: list[dict[str, Any]],
    flip_rows: list[dict[str, Any]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    rng = np.random.default_rng(bootstrap_seed)
    source_qif, source_layers = _source_qif_metrics(qif_rows)
    source_feature, source_features = _source_feature_metrics(feature_rows)
    sources = sorted({key[1] for key in source_qif})
    layers = sorted({key[2] for key in source_layers})
    feature_indices = sorted({key[2] for key in source_features})

    representation = {}
    for metric in (
        "disagreement",
        "mae",
        "signed_error",
        "zero_delta",
        "abs_signed_error",
        "abs_zero_delta",
        "abs_saturation_delta",
        "micro_disagreement",
        "micro_mae",
    ):
        representation[metric] = paired_summary(
            np.asarray([source_qif[("qad", source)][metric] for source in sources]),
            np.asarray([source_qif[("ste", source)][metric] for source in sources]),
            rng=rng,
            replicates=bootstrap_replicates,
        )

    features = {}
    for metric in ("normalized_mse", "cosine_distance", "mse", "mae"):
        features[metric] = paired_summary(
            np.asarray([source_feature[("qad", source)][metric] for source in sources]),
            np.asarray([source_feature[("ste", source)][metric] for source in sources]),
            rng=rng,
            replicates=bootstrap_replicates,
        )

    feature_detail = []
    for feature_index in feature_indices:
        row: dict[str, Any] = {"feature_index": feature_index}
        for metric in ("normalized_mse", "cosine_distance", "mse", "mae"):
            comparison = paired_summary(
                np.asarray(
                    [
                        source_features[("qad", source, feature_index)][metric]
                        for source in sources
                    ]
                ),
                np.asarray(
                    [
                        source_features[("ste", source, feature_index)][metric]
                        for source in sources
                    ]
                ),
                rng=rng,
                replicates=bootstrap_replicates,
            )
            row[metric] = comparison
        feature_detail.append(row)

    features_excluding_six = {}
    retained_features = [index for index in feature_indices if index != 6]
    for metric in ("normalized_mse", "cosine_distance", "mse", "mae"):
        features_excluding_six[metric] = paired_summary(
            np.asarray(
                [
                    np.mean(
                        [
                            source_features[("qad", source, index)][metric]
                            for index in retained_features
                        ]
                    )
                    for source in sources
                ]
            ),
            np.asarray(
                [
                    np.mean(
                        [
                            source_features[("ste", source, index)][metric]
                            for index in retained_features
                        ]
                    )
                    for source in sources
                ]
            ),
            rng=rng,
            replicates=bootstrap_replicates,
        )

    stage_layers: dict[str, list[str]] = defaultdict(list)
    for layer in layers:
        stage_layers[stage_for_layer(layer)].append(layer)
    stage_detail = []
    for stage, member_layers in sorted(stage_layers.items()):
        row = {"stage": stage, "layer_count": len(member_layers)}
        for metric in ("disagreement", "mae", "signed_error", "zero_delta"):
            method_values: dict[str, np.ndarray] = {}
            for method in METHODS:
                method_values[method] = np.asarray(
                    [
                        np.mean(
                            [
                                source_layers[(method, source, layer)][metric]
                                for layer in member_layers
                            ]
                        )
                        for source in sources
                    ]
                )
            row[metric] = paired_summary(
                method_values["qad"],
                method_values["ste"],
                rng=rng,
                replicates=bootstrap_replicates,
            )
        stage_detail.append(row)

    layer_detail = []
    for layer in layers:
        row = {"layer": layer, "stage": stage_for_layer(layer)}
        for metric in ("disagreement", "mae", "signed_error", "zero_delta"):
            row[metric] = paired_summary(
                np.asarray(
                    [source_layers[("qad", source, layer)][metric] for source in sources]
                ),
                np.asarray(
                    [source_layers[("ste", source, layer)][metric] for source in sources]
                ),
                rng=rng,
                replicates=bootstrap_replicates,
            )
        layer_detail.append(row)

    confusion = {
        (str(row["method"]), str(row["source_id"])): np.asarray(
            row["matrix"], dtype=np.int64
        )
        for row in confusion_rows
    }
    task = {}
    per_source_miou: dict[tuple[str, str], float] = {}
    for method in ("fp", *METHODS):
        total = sum(
            (confusion[(method, source)] for source in sources),
            np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64),
        )
        task[method] = confusion_metrics(total)
        for source in sources:
            per_source_miou[(method, source)] = confusion_metrics(
                confusion[(method, source)]
            )["miou"]

    task["qad_vs_ste_source_miou"] = paired_summary(
        np.asarray([per_source_miou[("qad", source)] for source in sources]),
        np.asarray([per_source_miou[("ste", source)] for source in sources]),
        rng=rng,
        replicates=bootstrap_replicates,
    )

    flip_lookup = {
        (str(row["method"]), str(row["source_id"])): (
            int(row["flip_count"]), int(row["valid_count"])
        )
        for row in flip_rows
    }
    for method in METHODS:
        flip = sum(flip_lookup[(method, source)][0] for source in sources)
        valid = sum(flip_lookup[(method, source)][1] for source in sources)
        task[method]["prediction_flip_rate_vs_fp"] = float(flip / valid)

    qif_repair = np.asarray(
        [
            source_qif[("ste", source)]["disagreement"]
            - source_qif[("qad", source)]["disagreement"]
            for source in sources
        ]
    )
    feature_repair = np.asarray(
        [
            source_feature[("ste", source)]["normalized_mse"]
            - source_feature[("qad", source)]["normalized_mse"]
            for source in sources
        ]
    )
    task_gain = np.asarray(
        [
            per_source_miou[("qad", source)]
            - per_source_miou[("ste", source)]
            for source in sources
        ]
    )
    associations = {
        "qif_repair_vs_source_miou_gain": _spearman_summary(
            qif_repair,
            task_gain,
            rng=rng,
            replicates=bootstrap_replicates,
        ),
        "feature_repair_vs_source_miou_gain": _spearman_summary(
            feature_repair,
            task_gain,
            rng=rng,
            replicates=bootstrap_replicates,
        ),
    }

    qif_layer_reductions = [
        row["disagreement"]["qad_minus_ste"] for row in layer_detail
    ]
    summary = {
        "scope": {
            "source_count": len(sources),
            "active_qif_layer_count": len(layers),
            "feature_count": len(feature_indices),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
        },
        "representation": representation,
        "distillation_features": features,
        "distillation_features_excluding_feature6": features_excluding_six,
        "feature6_share_of_raw_mse_reduction": float(
            (
                feature_detail[-1]["mse"]["ste_mean"]
                - feature_detail[-1]["mse"]["qad_mean"]
            )
            / sum(
                row["mse"]["ste_mean"] - row["mse"]["qad_mean"]
                for row in feature_detail
            )
        ),
        "task": task,
        "associations": associations,
        "layer_direction": {
            "qad_lower_disagreement_layer_count": int(
                np.sum(np.asarray(qif_layer_reductions) < 0)
            ),
            "qad_higher_disagreement_layer_count": int(
                np.sum(np.asarray(qif_layer_reductions) > 0)
            ),
            "equal_layer_count": int(np.sum(np.asarray(qif_layer_reductions) == 0)),
            "layer_count": len(layers),
        },
        "feature_detail": feature_detail,
        "stage_detail": stage_detail,
    }

    source_detail = []
    for source in sources:
        source_detail.append(
            {
                "source_id": source,
                "qad_qif_disagreement": source_qif[("qad", source)]["disagreement"],
                "ste_qif_disagreement": source_qif[("ste", source)]["disagreement"],
                "qif_repair": source_qif[("ste", source)]["disagreement"]
                - source_qif[("qad", source)]["disagreement"],
                "qad_feature_normalized_mse": source_feature[("qad", source)][
                    "normalized_mse"
                ],
                "ste_feature_normalized_mse": source_feature[("ste", source)][
                    "normalized_mse"
                ],
                "feature_repair": source_feature[("ste", source)]["normalized_mse"]
                - source_feature[("qad", source)]["normalized_mse"],
                "fp_miou": per_source_miou[("fp", source)],
                "qad_miou": per_source_miou[("qad", source)],
                "ste_miou": per_source_miou[("ste", source)],
                "qad_minus_ste_miou": per_source_miou[("qad", source)]
                - per_source_miou[("ste", source)],
            }
        )
    detail = {
        "sources": source_detail,
        "layers": layer_detail,
        "features": feature_detail,
        "stages": stage_detail,
    }
    return summary, detail


def summarize_self_distortion(
    *,
    qif_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    confusion_rows: list[dict[str, Any]],
    flip_rows: list[dict[str, Any]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Compare residual W4A4 sensitivity at each student's own master weights."""

    rng = np.random.default_rng(bootstrap_seed)
    source_qif, _ = _source_qif_metrics(qif_rows)
    source_feature, source_features = _source_feature_metrics(feature_rows)
    sources = sorted({key[1] for key in source_qif})
    feature_indices = sorted({key[2] for key in source_features})
    representation = {}
    for metric in (
        "disagreement",
        "mae",
        "abs_signed_error",
        "abs_zero_delta",
        "micro_disagreement",
        "micro_mae",
    ):
        representation[metric] = paired_summary(
            np.asarray([source_qif[("qad", source)][metric] for source in sources]),
            np.asarray([source_qif[("ste", source)][metric] for source in sources]),
            rng=rng,
            replicates=bootstrap_replicates,
        )
    features = {}
    for metric in ("normalized_mse", "cosine_distance", "mse", "mae"):
        features[metric] = paired_summary(
            np.asarray([source_feature[("qad", source)][metric] for source in sources]),
            np.asarray([source_feature[("ste", source)][metric] for source in sources]),
            rng=rng,
            replicates=bootstrap_replicates,
        )
    feature_detail = []
    for feature_index in feature_indices:
        feature_detail.append(
            {
                "feature_index": feature_index,
                "normalized_mse": paired_summary(
                    np.asarray(
                        [
                            source_features[("qad", source, feature_index)][
                                "normalized_mse"
                            ]
                            for source in sources
                        ]
                    ),
                    np.asarray(
                        [
                            source_features[("ste", source, feature_index)][
                                "normalized_mse"
                            ]
                            for source in sources
                        ]
                    ),
                    rng=rng,
                    replicates=bootstrap_replicates,
                ),
            }
        )

    confusion = {
        (str(row["method"]), str(row["source_id"])): np.asarray(
            row["matrix"], dtype=np.int64
        )
        for row in confusion_rows
    }
    task: dict[str, Any] = {}
    source_miou: dict[tuple[str, str], float] = {}
    for state in ("qad_fp", "qad_w4a4", "ste_fp", "ste_w4a4"):
        total = sum(
            (confusion[(state, source)] for source in sources),
            np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64),
        )
        task[state] = confusion_metrics(total)
        for source in sources:
            source_miou[(state, source)] = confusion_metrics(
                confusion[(state, source)]
            )["miou"]
    for method in METHODS:
        delta = np.asarray(
            [
                source_miou[(f"{method}_w4a4", source)]
                - source_miou[(f"{method}_fp", source)]
                for source in sources
            ]
        )
        boot = _bootstrap_mean(delta, rng, bootstrap_replicates)
        task[f"{method}_w4a4_minus_fp_source_miou"] = {
            "source_balanced_mean": float(delta.mean()),
            "cluster_bootstrap_95ci": percentile_ci(boot),
            "positive_source_count": int((delta > 0).sum()),
            "negative_source_count": int((delta < 0).sum()),
            "source_count": len(sources),
        }
    flips = {
        (str(row["method"]), str(row["source_id"])): (
            int(row["flip_count"]), int(row["valid_count"])
        )
        for row in flip_rows
    }
    for method in METHODS:
        flip = sum(flips[(method, source)][0] for source in sources)
        valid = sum(flips[(method, source)][1] for source in sources)
        task[f"{method}_w4a4"]["prediction_flip_rate_vs_own_fp"] = float(
            flip / valid
        )
    return {
        "scope": {
            "source_count": len(sources),
            "active_qif_layer_count": len({str(row["layer"]) for row in qif_rows}),
            "feature_count": len(feature_indices),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
        },
        "representation": representation,
        "features": features,
        "feature_detail": feature_detail,
        "task": task,
    }
