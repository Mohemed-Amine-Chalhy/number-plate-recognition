"""Deterministic end-to-end recognition metrics for regression evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def character_similarity(expected: str, predicted: str) -> float:
    """Return normalized Levenshtein similarity in the inclusive range [0, 1]."""

    longest = max(len(expected), len(predicted))
    if longest == 0:
        return 1.0
    return 1.0 - (_edit_distance(expected, predicted) / longest)


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    sample_id: str
    expected: tuple[str, ...]
    predicted: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    sample_count: int
    expected_plate_count: int
    predicted_plate_count: int
    exact_matches: int
    precision: float
    recall: float
    f1: float
    exact_sample_rate: float
    mean_character_similarity: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "sample_count": self.sample_count,
            "expected_plate_count": self.expected_plate_count,
            "predicted_plate_count": self.predicted_plate_count,
            "exact_matches": self.exact_matches,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "exact_sample_rate": self.exact_sample_rate,
            "mean_character_similarity": self.mean_character_similarity,
        }


def _best_character_similarities(
    expected: tuple[str, ...], predicted: tuple[str, ...]
) -> list[float]:
    """Return globally optimal one-to-one scores, including surplus penalties."""

    size = max(len(expected), len(predicted))
    if size == 0:
        return []
    similarities = [
        [
            character_similarity(expected[row], predicted[column])
            if row < len(expected) and column < len(predicted)
            else 0.0
            for column in range(size)
        ]
        for row in range(size)
    ]

    # The Hungarian algorithm minimizes cost. A cost of 1 - similarity gives
    # the globally maximal similarity assignment in O(n^3) time.
    costs = [[1.0 - score for score in row] for row in similarities]
    row_potential = [0.0] * (size + 1)
    column_potential = [0.0] * (size + 1)
    matched_row = [0] * (size + 1)
    predecessor = [0] * (size + 1)

    for row in range(1, size + 1):
        matched_row[0] = row
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                reduced_cost = (
                    costs[current_row - 1][candidate_column - 1]
                    - row_potential[current_row]
                    - column_potential[candidate_column]
                )
                if reduced_cost < minimum[candidate_column]:
                    minimum[candidate_column] = reduced_cost
                    predecessor[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    row_potential[matched_row[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous_column = predecessor[column]
            matched_row[column] = matched_row[previous_column]
            column = previous_column
            if column == 0:
                break

    assignment = [-1] * size
    for column in range(1, size + 1):
        assignment[matched_row[column] - 1] = column - 1
    return [similarities[row][column] for row, column in enumerate(assignment)]


def evaluate_samples(samples: list[EvaluationSample]) -> EvaluationMetrics:
    """Calculate exact plate and character-level metrics for labeled samples."""

    expected_count = sum(len(sample.expected) for sample in samples)
    predicted_count = sum(len(sample.predicted) for sample in samples)
    exact_matches = 0
    exact_samples = 0
    character_scores: list[float] = []

    for sample in samples:
        expected_counter = Counter(sample.expected)
        predicted_counter = Counter(sample.predicted)
        exact_matches += sum((expected_counter & predicted_counter).values())
        exact_samples += expected_counter == predicted_counter
        character_scores.extend(_best_character_similarities(sample.expected, sample.predicted))

    precision = exact_matches / predicted_count if predicted_count else 0.0
    recall = exact_matches / expected_count if expected_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EvaluationMetrics(
        sample_count=len(samples),
        expected_plate_count=expected_count,
        predicted_plate_count=predicted_count,
        exact_matches=exact_matches,
        precision=precision,
        recall=recall,
        f1=f1,
        exact_sample_rate=exact_samples / len(samples) if samples else 0.0,
        mean_character_similarity=(
            sum(character_scores) / len(character_scores) if character_scores else 0.0
        ),
    )
