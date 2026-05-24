package report

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
)

func TestAggregate(t *testing.T) {
	tests := map[string]struct {
		rows   []domain.LoadedRun
		verify func(*testing.T, []AggregatedServerConfigReport)
	}{
		"groups by server config ignoring prompt profile": {
			rows: []domain.LoadedRun{
				run("a", "code", 2048, 100, 50, 4096, domain.StatusSuccess),
				run("b", "qa", 2048, 90, 45, 4096, domain.StatusSuccess),
			},
			verify: func(t *testing.T, reports []AggregatedServerConfigReport) {
				require.Len(t, reports, 1)
				assert.Equal(t, 2, reports[0].TotalRuns)
				assert.Equal(t, 2, reports[0].SuccessCount)
				assert.ElementsMatch(t, []string{"code", "qa"}, reports[0].ProfilesSeen)
			},
		},
		"evidence and vram are calculated": {
			rows: []domain.LoadedRun{
				run("a", "code", 2048, 100, 50, 1024, domain.StatusSuccess),
				run("b", "code", 2048, 0, 0, 512, domain.StatusTimeout),
			},
			verify: func(t *testing.T, reports []AggregatedServerConfigReport) {
				r := reports[0]
				assert.Equal(t, "timeout (1/2)", r.Evidence)
				require.NotNil(t, r.FreeVRAMMiB.Min)
				assert.InDelta(t, 512.0, *r.FreeVRAMMiB.Min, 0.0001)
				require.NotNil(t, r.FreeVRAMMiB.Mean)
				assert.InDelta(t, 768.0, *r.FreeVRAMMiB.Mean, 0.0001)
			},
		},
		"mixed statuses are counted correctly": {
			rows: []domain.LoadedRun{
				run("a", "code", 2048, 100, 50, 4096, domain.StatusSuccess),
				run("b", "code", 2048, 0, 0, 2048, domain.StatusOOM),
				run("c", "code", 2048, 0, 0, 2048, domain.StatusFailed),
			},
			verify: func(t *testing.T, reports []AggregatedServerConfigReport) {
				require.Len(t, reports, 1)
				assert.Equal(t, 3, reports[0].TotalRuns)
				assert.Equal(t, 1, reports[0].SuccessCount)
				assert.Equal(t, 1, reports[0].OOMCount)
			},
		},
		"empty input returns no reports": {
			rows: nil,
			verify: func(t *testing.T, reports []AggregatedServerConfigReport) {
				assert.Empty(t, reports)
			},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			reports, context := Aggregate(tc.rows, 1.5)
			assert.NotNil(t, context)
			tc.verify(t, reports)
		})
	}
}

func TestCompareRejectsDifferentProfileSets(t *testing.T) {
	reports, _ := Aggregate([]domain.LoadedRun{
		run("base", "code", 2048, 100, 50, 4096, domain.StatusSuccess),
		run("candidate", "qa", 4096, 110, 55, 4096, domain.StatusSuccess),
	}, 1.5)
	err := EnforcePromptProfileComparability(reports)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "prompt profile sets differ")
}

func TestSortReports(t *testing.T) {
	makeReports := func() []AggregatedServerConfigReport {
		reports, _ := Aggregate([]domain.LoadedRun{
			run("a", "code", 2048, 100, 50, 4096, domain.StatusSuccess),
			run("b", "code", 4096, 100, 50, 4096, domain.StatusSuccess),
		}, 1.5)
		return reports
	}

	tests := map[string]struct {
		sortKey string
		verify  func(*testing.T, []AggregatedServerConfigReport)
	}{
		"generation_tok_s preserves stable order for equal scores": {
			sortKey: "generation_tok_s",
			verify: func(t *testing.T, reports []AggregatedServerConfigReport) {
				assert.Equal(t, 2048, reports[0].Key.ContextSize)
				assert.Equal(t, 4096, reports[1].Key.ContextSize)
			},
		},
		"balanced sorts without error": {
			sortKey: "balanced",
			verify: func(t *testing.T, reports []AggregatedServerConfigReport) {
				require.Len(t, reports, 2)
			},
		},
		"latest groups by same key": {
			sortKey: "latest",
			verify: func(t *testing.T, reports []AggregatedServerConfigReport) {
				require.Len(t, reports, 2)
			},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			reports := makeReports()
			SortReports(reports, tc.sortKey)
			tc.verify(t, reports)
		})
	}
}

func TestKeyFromServerConfig(t *testing.T) {
	cfg := domain.ServerConfig{
		Model: "model", ContextSize: 4096, KVType: "q8_0",
		NCPUMOE: new(18), MTP: new(true),
		SpecType: "draft-mtp", BatchSize: new(512), UBatchSize: new(256),
	}
	key := KeyFromServerConfig(cfg)
	assert.Equal(t, "model", key.Model)
	assert.Equal(t, 4096, key.ContextSize)
	assert.Equal(t, "q8_0", key.KVType)
	assert.Equal(t, 18, *key.NCPUMOE)
	assert.True(t, *key.MTP)
	assert.Equal(t, "draft-mtp", key.SpecType)
}

func TestObservationFromRun(t *testing.T) {
	lr := run("a", "code", 2048, 78.0, 812.0, 5520, domain.StatusSuccess)
	obs := ObservationFromRun(lr)
	assert.Equal(t, "code", obs.PromptProfile)
	assert.Equal(t, domain.StatusSuccess, obs.Status)
	assert.Len(t, obs.GenerationValues, 1)
	assert.InDelta(t, 78.0, obs.GenerationValues[0], 0.0001)
	assert.Len(t, obs.PromptValues, 1)
	assert.InDelta(t, 812.0, obs.PromptValues[0], 0.0001)
}

func TestRenderTable(t *testing.T) {
	tests := map[string]struct {
		columns []string
		rows    []map[string]string
		limit   int
		verify  func(*testing.T, string)
	}{
		"no rows returns empty message": {
			columns: []string{"col1", "col2"},
			rows:    nil,
			verify:  func(t *testing.T, s string) { assert.Equal(t, "No rows.", s) },
		},
		"renders header and rows": {
			columns: []string{"name", "value"},
			rows:    []map[string]string{{"name": "a", "value": "1"}, {"name": "b", "value": "2"}},
			verify: func(t *testing.T, s string) {
				assert.Contains(t, s, "name")
				assert.Contains(t, s, "a")
				assert.Contains(t, s, "b")
			},
		},
		"limit truncates output": {
			columns: []string{"x"},
			rows:    []map[string]string{{"x": "1"}, {"x": "2"}, {"x": "3"}},
			limit:   2,
			verify: func(t *testing.T, s string) {
				assert.Contains(t, s, "1")
				assert.Contains(t, s, "2")
				assert.NotContains(t, s, "3")
			},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			tc.verify(t, renderTable(tc.columns, tc.rows, tc.limit))
		})
	}
}

func TestStatusWeight(t *testing.T) {
	tests := map[string]struct {
		status domain.RunStatus
		want   float64
	}{
		"success has weight 1":       {status: domain.StatusSuccess, want: 1},
		"timeout has weight 0":       {status: domain.StatusTimeout, want: 0},
		"oom has weight 0":           {status: domain.StatusOOM, want: 0},
		"failed has weight 0":        {status: domain.StatusFailed, want: 0},
		"unsupported has weight 0.4": {status: domain.StatusUnsupported, want: 0.4},
		"unknown has weight 0.4":     {status: domain.StatusUnknown, want: 0.4},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			assert.InDelta(t, tc.want, statusWeight(tc.status), 0.0001)
		})
	}
}

// --- helpers ---

func run(id, profile string, ctx int, gen, prompt, vram float64, status domain.RunStatus) domain.LoadedRun {
	cfg := domain.ServerConfig{
		Model: "model:A", ContextSize: ctx, KVType: "q8_0",
		NCPUMOE: new(0), BatchSize: new(128), UBatchSize: new(32),
	}
	return domain.LoadedRun{
		Summary: domain.RunSummary{
			RunID: id, CreatedAt: "2026-01-01T00:00:00Z", Model: "model:A", PromptProfile: profile,
			ServerConfig: cfg, Status: domain.NewRunStatusInfo(status, ""), DurationSec: 10,
		},
		SystemMetrics: []domain.SystemMetricSample{{Time: "t", VRAMFree: &vram, VRAMUsed: new(1024.0)}},
		LlamaMetrics:  []domain.LlamaCppMetricSample{{Time: "t", GenerationTokS: &gen, PromptEvalTokS: &prompt, TotalTimeSeconds: new(10.0)}},
		SystemSummary: domain.SystemSummary{MinVRAMFreeMiB: &vram, MeanVRAMFreeMiB: &vram, PeakVRAMMiB: new(1024.0)},
		LlamaSummary:  domain.LlamaSummary{GenerationTokS: &gen, PromptEvalTokS: &prompt, TotalTimeSec: new(10.0)},
	}
}
