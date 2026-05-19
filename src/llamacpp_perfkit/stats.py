from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import exp, log, sqrt
from statistics import fmean


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class Sample:
    values: tuple[float, ...] = ()

    @classmethod
    def from_iterable(cls, values: Iterable[object]) -> Sample:
        return cls(tuple(float(v) for v in values if _is_number(v)))  # type: ignore[arg-type]

    def sorted(self) -> tuple[float, ...]:
        return tuple(sorted(self.values))

    def mean(self) -> float | None:
        return fmean(self.values) if self.values else None

    def median(self) -> float | None:
        if not self.values:
            return None
        ordered = sorted(self.values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2

    def stddev(self) -> float | None:
        if len(self.values) < 2:
            return None
        mean = self.mean()
        assert mean is not None
        return sqrt(sum((value - mean) ** 2 for value in self.values) / (len(self.values) - 1))

    def percentile(self, percentile: float) -> float | None:
        if not self.values:
            return None
        ordered = sorted(self.values)
        if len(ordered) == 1:
            return ordered[0]
        bounded = min(100.0, max(0.0, float(percentile)))
        rank = (bounded / 100.0) * (len(ordered) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = rank - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    def geometric_mean(self) -> float | None:
        positives = [value for value in self.values if value > 0]
        if not positives:
            return None
        return exp(sum(log(value) for value in positives) / len(positives))

    def min(self) -> float | None:
        return min(self.values) if self.values else None

    def max(self) -> float | None:
        return max(self.values) if self.values else None


@dataclass(frozen=True)
class MetricSummary:
    count: int = 0
    mean: float | None = None
    median: float | None = None
    p10: float | None = None
    stddev: float | None = None
    geometric_mean: float | None = None
    min: float | None = None
    max: float | None = None

    @classmethod
    def empty(cls) -> MetricSummary:
        return cls()


def summarize(values: Iterable[object]) -> MetricSummary:
    sample = Sample.from_iterable(values)
    return MetricSummary(
        count=len(sample.values),
        mean=sample.mean(),
        median=sample.median(),
        p10=sample.percentile(10),
        stddev=sample.stddev(),
        geometric_mean=sample.geometric_mean(),
        min=sample.min(),
        max=sample.max(),
    )
