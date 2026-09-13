from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.paired_statistics import (
    BOOTSTRAP_RESAMPLES,
    PairedStatisticsError,
    holm_bonferroni,
    paired_delta_bootstrap,
)


STRATA = {"CI": 28, "VI-1": 18, "VI-2": 32, "VI-3": 10, "VI-4": 62}


def _outcomes() -> tuple[tuple[bool, ...], tuple[bool, ...], tuple[str, ...]]:
    current = tuple([True] * 100 + [False] * 50)
    reference = tuple([True] * 80 + [False] * 20 + [True] * 10 + [False] * 40)
    strata = tuple(label for label, count in STRATA.items() for _ in range(count))
    return current, reference, strata


def test_exact_deterministic_paired_bootstrap_results() -> None:
    current, reference, strata = _outcomes()
    stratified = paired_delta_bootstrap(
        current,
        reference,
        strata=strata,
        expected_strata=STRATA,
        expected_n=150,
    )
    repeated = paired_delta_bootstrap(
        current,
        reference,
        strata=strata,
        expected_strata=STRATA,
        expected_n=150,
    )
    unstratified = paired_delta_bootstrap(current, reference, expected_n=150)

    assert repeated == stratified
    assert stratified["resamples"] == BOOTSTRAP_RESAMPLES
    assert stratified["observed_delta_percentage_points"] == 6.666667
    assert stratified["low_percentage_points"] == 0.666667
    assert stratified["high_percentage_points"] == 12.666667
    assert unstratified["low_percentage_points"] == -0.666667
    assert unstratified["high_percentage_points"] == 14.0


def test_degenerate_paired_bootstrap_is_exactly_zero() -> None:
    values = tuple(index % 2 == 0 for index in range(150))
    result = paired_delta_bootstrap(values, values, expected_n=150)
    assert result["observed_delta_percentage_points"] == 0.0
    assert result["low_percentage_points"] == 0.0
    assert result["high_percentage_points"] == 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda current, reference, strata: (current[:-1], reference, strata), "equal nonzero"),
        (
            lambda current, reference, strata: ((1, *current[1:]), reference, strata),
            "strict booleans",
        ),
        (
            lambda current, reference, strata: (current, reference, strata[:-1]),
            "every paired case",
        ),
    ],
)
def test_bootstrap_rejects_malformed_case_vectors(mutation, message: str) -> None:
    current, reference, strata = _outcomes()
    bad_current, bad_reference, bad_strata = mutation(current, reference, strata)
    with pytest.raises(PairedStatisticsError, match=message):
        paired_delta_bootstrap(
            bad_current,
            bad_reference,
            strata=bad_strata,
            expected_strata=STRATA,
            expected_n=150,
        )


def test_bootstrap_rejects_changed_protocol_parameters_and_quotas() -> None:
    current, reference, strata = _outcomes()
    bad_quotas = copy.deepcopy(STRATA)
    bad_quotas["CI"] -= 1
    with pytest.raises(PairedStatisticsError, match="frozen stratum quotas"):
        paired_delta_bootstrap(
            current,
            reference,
            strata=strata,
            expected_strata=bad_quotas,
            expected_n=150,
        )
    with pytest.raises(PairedStatisticsError, match="exactly 10000"):
        paired_delta_bootstrap(current, reference, expected_n=150, resamples=9999)
    with pytest.raises(PairedStatisticsError, match="frozen protocol"):
        paired_delta_bootstrap(current, reference, expected_n=150, seed=42)
    with pytest.raises(PairedStatisticsError, match="frozen sample size"):
        paired_delta_bootstrap(current, reference, expected_n=149)


def test_holm_adjustment_requires_and_reports_one_complete_family() -> None:
    result = holm_bonferroni(
        {"model-c": 0.04, "model-a": 0.01, "model-b": 0.03},
        family_label="yao_predeclared_model_comparisons",
        declared_family_size=3,
    )
    assert result == {
        "method": "holm_bonferroni",
        "family_label": "yao_predeclared_model_comparisons",
        "alpha": 0.05,
        "family_size": 3,
        "complete_family_asserted": True,
        "adjusted_p_values": {
            "model-a": 0.03,
            "model-b": 0.06,
            "model-c": 0.06,
        },
        "reject_null": {
            "model-a": True,
            "model-b": False,
            "model-c": False,
        },
    }
    with pytest.raises(PairedStatisticsError, match="complete family"):
        holm_bonferroni({}, family_label="yao", declared_family_size=1)
    with pytest.raises(PairedStatisticsError, match="declared complete family size"):
        holm_bonferroni(
            {"model-a": 0.01},
            family_label="yao",
            declared_family_size=2,
        )
    with pytest.raises(PairedStatisticsError, match="finite probability"):
        holm_bonferroni(
            {"model-a": float("nan")},
            family_label="yao",
            declared_family_size=1,
        )
