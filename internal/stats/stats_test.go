package stats

import (
	"math"
	"testing"
)

func TestSummarize(t *testing.T) {
	s := Summarize([]float64{1, 2, 3})
	if *s.Mean != 2 {
		t.Fatalf("mean = %v", *s.Mean)
	}
	if *s.Median != 2 {
		t.Fatalf("median = %v", *s.Median)
	}
	if math.Abs(*s.Stddev-1) > 0.0001 {
		t.Fatalf("stddev = %v", *s.Stddev)
	}
}

func TestMedianEvenAndP10(t *testing.T) {
	s := Summarize([]float64{4, 1, 3, 2})
	if *s.Median != 2.5 {
		t.Fatalf("median even = %v", *s.Median)
	}
	s = Summarize([]float64{1, 2, 3, 4, 5})
	if math.Abs(*s.P10-1.4) > 0.0001 {
		t.Fatalf("p10 = %v", *s.P10)
	}
}

func TestGeometricMeanIgnoresNonPositive(t *testing.T) {
	s := Summarize([]float64{1, 0, -4, 9})
	if math.Abs(*s.GeometricMean-3) > 0.0001 {
		t.Fatalf("geomean = %v", *s.GeometricMean)
	}
}

func TestEmptyInput(t *testing.T) {
	s := Summarize(nil)
	if s.Count != 0 || s.Mean != nil || s.Median != nil || s.Stddev != nil || s.GeometricMean != nil || s.Min != nil || s.Max != nil || s.P10 != nil {
		t.Fatalf("empty summary has data: %+v", s)
	}
}
