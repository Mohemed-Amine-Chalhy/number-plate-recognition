from __future__ import annotations

import pytest

from number_plate_recognition.evaluation import (
    EvaluationSample,
    character_similarity,
    evaluate_samples,
)


def test_character_similarity() -> None:
    assert character_similarity("123A45", "123A45") == 1.0
    assert character_similarity("", "") == 1.0
    assert character_similarity("ABC", "AXC") == pytest.approx(2 / 3)
    assert character_similarity("ABC", "") == 0.0


def test_evaluate_samples_counts_duplicates_and_errors() -> None:
    metrics = evaluate_samples(
        [
            EvaluationSample("one", ("123A", "123A"), ("123A", "999B")),
            EvaluationSample("two", ("777H",), ("777H",)),
        ]
    )

    assert metrics.sample_count == 2
    assert metrics.expected_plate_count == 3
    assert metrics.predicted_plate_count == 3
    assert metrics.exact_matches == 2
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.exact_sample_rate == 0.5
    assert 0.0 < metrics.mean_character_similarity < 1.0


def test_empty_evaluation_is_well_defined() -> None:
    metrics = evaluate_samples([])

    assert metrics.sample_count == 0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.exact_sample_rate == 0.0


def test_character_matching_is_globally_optimal_and_order_independent() -> None:
    first = evaluate_samples([EvaluationSample("one", ("ABC", "ABD"), ("ABD", "AXC"))])
    reversed_predictions = evaluate_samples(
        [EvaluationSample("one", ("ABC", "ABD"), ("AXC", "ABD"))]
    )

    assert first.mean_character_similarity == pytest.approx(5 / 6)
    assert reversed_predictions.mean_character_similarity == pytest.approx(5 / 6)


def test_character_metric_penalizes_surplus_predictions() -> None:
    metrics = evaluate_samples([EvaluationSample("one", ("ABC",), ("ABC", "EXTRA"))])

    assert metrics.mean_character_similarity == pytest.approx(0.5)
