package stats

import (
	"math"
	"sort"
)

type MetricSummary struct {
	Count         int      `json:"count"`
	Mean          *float64 `json:"mean"`
	Median        *float64 `json:"median"`
	Stddev        *float64 `json:"stddev"`
	GeometricMean *float64 `json:"geometric_mean"`
	Min           *float64 `json:"min"`
	Max           *float64 `json:"max"`
	P10           *float64 `json:"p10"`
}

func Summarize(values []float64) MetricSummary {
	clean := make([]float64, 0, len(values))
	for _, value := range values {
		if !math.IsNaN(value) && !math.IsInf(value, 0) {
			clean = append(clean, value)
		}
	}
	if len(clean) == 0 {
		return MetricSummary{}
	}

	ordered := append([]float64(nil), clean...)
	sort.Float64s(ordered)

	mean := average(clean)
	median := percentileSorted(ordered, 50)
	p10 := percentileSorted(ordered, 10)
	minVal := ordered[0]
	maxVal := ordered[len(ordered)-1]
	geo := geometricMean(clean)

	out := MetricSummary{
		Count:  len(clean),
		Mean:   &mean,
		Median: &median,
		Min:    &minVal,
		Max:    &maxVal,
		P10:    &p10,
	}
	if len(clean) > 1 {
		std := sampleStddev(clean, mean)
		out.Stddev = &std
	}
	if geo != nil {
		out.GeometricMean = geo
	}
	return out
}

func average(values []float64) float64 {
	var sum float64
	for _, value := range values {
		sum += value
	}
	return sum / float64(len(values))
}

func sampleStddev(values []float64, mean float64) float64 {
	var sum float64
	for _, value := range values {
		diff := value - mean
		sum += diff * diff
	}
	return math.Sqrt(sum / float64(len(values)-1))
}

func geometricMean(values []float64) *float64 {
	var sum float64
	var count int
	for _, value := range values {
		if value <= 0 {
			continue
		}
		sum += math.Log(value)
		count++
	}
	if count == 0 {
		return nil
	}
	out := math.Exp(sum / float64(count))
	return &out
}

func percentileSorted(ordered []float64, pct float64) float64 {
	if len(ordered) == 1 {
		return ordered[0]
	}
	bounded := math.Max(0, math.Min(100, pct))
	rank := (bounded / 100) * float64(len(ordered)-1)
	lower := int(rank)
	upper := lower + 1
	if upper >= len(ordered) {
		upper = len(ordered) - 1
	}
	fraction := rank - float64(lower)
	return ordered[lower] + (ordered[upper]-ordered[lower])*fraction
}
