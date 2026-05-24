package stats

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSummarize(t *testing.T) {
	tests := map[string]struct {
		input  []float64
		verify func(*testing.T, MetricSummary)
	}{
		"three values": {
			input: []float64{1, 2, 3},
			verify: func(t *testing.T, s MetricSummary) {
				assert.Equal(t, 3, s.Count)
				assert.InDelta(t, 2.0, *s.Mean, 0.0001)
				assert.InDelta(t, 2.0, *s.Median, 0.0001)
				assert.InDelta(t, 1.0, *s.Stddev, 0.0001)
				assert.InDelta(t, 1.0, *s.Min, 0.0001)
				assert.InDelta(t, 3.0, *s.Max, 0.0001)
			},
		},
		"even count returns average of two middle values": {
			input: []float64{4, 1, 3, 2},
			verify: func(t *testing.T, s MetricSummary) {
				assert.InDelta(t, 2.5, *s.Median, 0.0001)
			},
		},
		"odd count returns middle value": {
			input: []float64{5, 1, 3, 2, 4},
			verify: func(t *testing.T, s MetricSummary) {
				assert.InDelta(t, 3.0, *s.Median, 0.0001)
			},
		},
		"p10 of five values": {
			input: []float64{1, 2, 3, 4, 5},
			verify: func(t *testing.T, s MetricSummary) {
				assert.InDelta(t, 1.4, *s.P10, 0.0001)
			},
		},
		"p10 of single value": {
			input: []float64{42},
			verify: func(t *testing.T, s MetricSummary) {
				assert.InDelta(t, 42.0, *s.P10, 0.0001)
			},
		},
		"single value has no stddev": {
			input: []float64{5},
			verify: func(t *testing.T, s MetricSummary) {
				assert.Equal(t, 1, s.Count)
				assert.InDelta(t, 5.0, *s.Mean, 0.0001)
				assert.InDelta(t, 5.0, *s.Median, 0.0001)
				assert.InDelta(t, 5.0, *s.Min, 0.0001)
				assert.InDelta(t, 5.0, *s.Max, 0.0001)
				assert.Nil(t, s.Stddev)
			},
		},
		"all same values has zero stddev": {
			input: []float64{10, 10, 10, 10},
			verify: func(t *testing.T, s MetricSummary) {
				assert.Equal(t, 4, s.Count)
				assert.InDelta(t, 10.0, *s.Mean, 0.0001)
				assert.InDelta(t, 0.0, *s.Stddev, 0.0001)
				assert.InDelta(t, 10.0, *s.Min, 0.0001)
				assert.InDelta(t, 10.0, *s.Max, 0.0001)
			},
		},
		"geometric mean ignores non positive values": {
			input: []float64{1, 0, -4, 9},
			verify: func(t *testing.T, s MetricSummary) {
				assert.InDelta(t, 3.0, *s.GeometricMean, 0.0001)
			},
		},
		"all non positive gives nil geometric mean": {
			input: []float64{0, -1, -2},
			verify: func(t *testing.T, s MetricSummary) {
				assert.Nil(t, s.GeometricMean)
			},
		},
		"NaN values are filtered out": {
			input: []float64{1, math.NaN(), 3},
			verify: func(t *testing.T, s MetricSummary) {
				assert.Equal(t, 2, s.Count)
				assert.InDelta(t, 2.0, *s.Mean, 0.0001)
			},
		},
		"Inf values are filtered out": {
			input: []float64{1, math.Inf(1), 3},
			verify: func(t *testing.T, s MetricSummary) {
				assert.Equal(t, 2, s.Count)
				assert.InDelta(t, 2.0, *s.Mean, 0.0001)
			},
		},
		"nil input returns empty summary": {
			input: nil,
			verify: func(t *testing.T, s MetricSummary) {
				assert.Equal(t, 0, s.Count)
				assert.Nil(t, s.Mean)
				assert.Nil(t, s.Median)
				assert.Nil(t, s.Stddev)
				assert.Nil(t, s.GeometricMean)
				assert.Nil(t, s.Min)
				assert.Nil(t, s.Max)
				assert.Nil(t, s.P10)
			},
		},
		"empty slice returns empty summary": {
			input: []float64{},
			verify: func(t *testing.T, s MetricSummary) {
				assert.Equal(t, 0, s.Count)
				assert.Nil(t, s.Mean)
			},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			s := Summarize(tc.input)
			tc.verify(t, s)
		})
	}
}
