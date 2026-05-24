package report

import (
	"testing"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
)

func TestAggregateGroupsByServerConfigIgnoringPromptProfile(t *testing.T) {
	rows := []domain.LoadedRun{
		run("a", "code", 2048, 100, 50, 4096, domain.StatusSuccess),
		run("b", "qa", 2048, 90, 45, 4096, domain.StatusSuccess),
	}
	reports, _ := Aggregate(rows, 1.5)
	if len(reports) != 1 {
		t.Fatalf("reports = %d", len(reports))
	}
	if reports[0].TotalRuns != 2 || reports[0].SuccessCount != 2 {
		t.Fatalf("bad counts: %+v", reports[0])
	}
	if got := reports[0].ProfilesSeen; len(got) != 2 || got[0] != "code" || got[1] != "qa" {
		t.Fatalf("profiles = %v", got)
	}
}

func TestCompareRejectsDifferentProfileSets(t *testing.T) {
	reports, _ := Aggregate([]domain.LoadedRun{
		run("base", "code", 2048, 100, 50, 4096, domain.StatusSuccess),
		run("candidate", "qa", 4096, 110, 55, 4096, domain.StatusSuccess),
	}, 1.5)
	if err := EnforcePromptProfileComparability(reports); err == nil {
		t.Fatal("expected profile mismatch error")
	}
}

func TestEvidenceAndVRAM(t *testing.T) {
	rows := []domain.LoadedRun{
		run("a", "code", 2048, 100, 50, 1024, domain.StatusSuccess),
		run("b", "code", 2048, 0, 0, 512, domain.StatusTimeout),
	}
	reports, _ := Aggregate(rows, 1.5)
	report := reports[0]
	if report.Evidence != "timeout (1/2)" {
		t.Fatalf("evidence = %q", report.Evidence)
	}
	if *report.FreeVRAMMiB.Min != 512 || *report.FreeVRAMMiB.Mean != 768 {
		t.Fatalf("vram summary = %+v", report.FreeVRAMMiB)
	}
}

func TestSortReportsStableForEqualScores(t *testing.T) {
	rows := []domain.LoadedRun{
		run("a", "code", 2048, 100, 50, 4096, domain.StatusSuccess),
		run("b", "code", 4096, 100, 50, 4096, domain.StatusSuccess),
	}
	reports, _ := Aggregate(rows, 1.5)
	SortReports(reports, "generation_tok_s")
	if reports[0].Key.ContextSize != 2048 || reports[1].Key.ContextSize != 4096 {
		t.Fatalf("stable order changed: %+v", reports)
	}
}

func run(id, profile string, ctx int, gen, prompt, vram float64, status domain.RunStatus) domain.LoadedRun {
	cfg := domain.ServerConfig{Model: "model:A", ContextSize: ctx, KVType: "q8_0", NCPUMOE: domain.Ptr(0), BatchSize: domain.Ptr(128), UBatchSize: domain.Ptr(32)}
	return domain.LoadedRun{
		Summary: domain.RunSummary{
			RunID: id, CreatedAt: "2026-01-01T00:00:00Z", Model: "model:A", PromptProfile: profile,
			ServerConfig: cfg, Status: domain.NewRunStatusInfo(status, ""), DurationSec: 10,
		},
		SystemMetrics: []domain.SystemMetricSample{{Time: "t", VRAMFree: &vram, VRAMUsed: domain.Ptr(1024.0)}},
		LlamaMetrics:  []domain.LlamaCppMetricSample{{Time: "t", GenerationTokS: &gen, PromptEvalTokS: &prompt, TotalTimeSeconds: domain.Ptr(10.0)}},
		SystemSummary: domain.SystemSummary{MinVRAMFreeMiB: &vram, MeanVRAMFreeMiB: &vram, PeakVRAMMiB: domain.Ptr(1024.0)},
		LlamaSummary:  domain.LlamaSummary{GenerationTokS: &gen, PromptEvalTokS: &prompt, TotalTimeSec: domain.Ptr(10.0)},
	}
}
