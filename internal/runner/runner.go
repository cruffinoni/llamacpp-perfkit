package runner

import (
	"context"
	"fmt"
	"path/filepath"
	"time"

	"github.com/cruffinoni/llamacpp-perfkit/internal/bench"
	"github.com/cruffinoni/llamacpp-perfkit/internal/config"
	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
	"github.com/cruffinoni/llamacpp-perfkit/internal/llamacpp"
	"github.com/cruffinoni/llamacpp-perfkit/internal/storage"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui"
	tuifmt "github.com/cruffinoni/llamacpp-perfkit/internal/tui/format"
)

type Options struct {
	Mode        string
	MaxRuns     *int
	RetryFailed bool
	Force       bool
	DryRun      bool
}

type Runner struct {
	Config        config.Config
	Features      llamacpp.Features
	ResultsDir    string
	RawDir        string
	MonitoringDir string
	Options       Options
}

func New(ctx context.Context, cfg config.Config, opts Options) (*Runner, error) {
	_, resultsDir, rawDir, monitoringDir, err := cfg.OutputDirs()
	if err != nil {
		return nil, err
	}
	features, ok, err := llamacpp.LoadFeatures(resultsDir)
	if err != nil {
		return nil, err
	}
	if !ok || !features.ValidForBench {
		features, err = llamacpp.DetectFeatures(ctx, cfg)
		if err != nil {
			return nil, fmt.Errorf("detect llama.cpp features: %w", err)
		}
		if err := llamacpp.WriteFeatures(features, resultsDir); err != nil {
			return nil, fmt.Errorf("write llama.cpp feature cache: %w", err)
		}
	}
	if !features.ValidForBench && !opts.DryRun {
		return nil, fmt.Errorf("cannot benchmark: %s", features.InvalidReason)
	}
	return &Runner{Config: cfg, Features: features, ResultsDir: resultsDir, RawDir: rawDir, MonitoringDir: monitoringDir, Options: opts}, nil
}

func (r *Runner) Plan() (domain.BenchmarkPlan, error) {
	rows, err := storage.LoadRuns(r.ResultsDir)
	if err != nil {
		return domain.BenchmarkPlan{}, err
	}
	plan := bench.MakePlan(r.Config, r.Features, rows, bench.PlanOptions{
		Mode:        r.Options.Mode,
		MaxRuns:     r.Options.MaxRuns,
		RetryFailed: r.Options.RetryFailed,
		Force:       r.Options.Force,
	})
	if err := bench.WritePlan(filepath.Join(r.ResultsDir, "last_plan.json"), plan); err != nil {
		return domain.BenchmarkPlan{}, err
	}
	return plan, nil
}

func (r *Runner) PrintPlan(plan domain.BenchmarkPlan) {
	fmt.Printf("Budget mode: %s\n", plan.Mode)
	fmt.Printf("Max new requests: %s\n", maxRunsLabel(plan.MaxRuns))
	fmt.Printf("Reuse existing results: %v\n", plan.ReuseExistingResults)
	fmt.Printf("Candidate combinations: %d\n", plan.CandidateCount)
	fmt.Printf("Selected plan entries: %d\n", plan.SelectedCount)
	fmt.Printf("Estimated new requests now: %d\n", plan.EstimatedRuns)
	for _, group := range bench.RunnableGroups(plan) {
		if len(group) == 0 {
			continue
		}
		job := group[0].Job
		rawPath := filepath.Join(r.RawDir, fmt.Sprintf("dry-run-server-%04d.log", group[0].ServerIndex))
		cmd := llamacpp.BuildServerCommand(r.Config, r.Features, job, 0, rawPath)
		fmt.Printf("\n[server %04d] entries=%d ctx=%d kv=%s moe=%s spec=%s batch=%s ubatch=%s\n",
			group[0].ServerIndex, len(group), job.ServerConfig.ContextSize, job.ServerConfig.KVType,
			domain.IntValue(job.ServerConfig.NCPUMOE), or(job.ServerConfig.SpecType, "none"),
			domain.IntValue(job.ServerConfig.BatchSize), domain.IntValue(job.ServerConfig.UBatchSize))
		fmt.Println(llamacpp.CommandToShell(cmd))
		for _, item := range group {
			fmt.Printf("  request plan_id=%s action=%s profile=%s hash=%s risk=%s\n", item.RunID, item.Action, item.Job.PromptProfile.Name, item.ConfigHash, item.RiskLevel)
		}
	}
}

func (r *Runner) Run(ctx context.Context) error {
	plan, err := r.Plan()
	if err != nil {
		return err
	}
	if r.Options.DryRun {
		r.PrintPlan(plan)
		return nil
	}
	initial := tui.BenchmarkTUIState{
		RunID:         fmt.Sprintf("%d", time.Now().Unix()),
		BuildInfo:     tui.BuildInfoView{CommitShort: or(r.Features.LlamaCpp.CommitShort, "unknown"), Branch: or(r.Features.LlamaCpp.Branch, "unknown"), Backend: or(r.Features.Backend, "server")},
		ModelName:     r.Config.ModelHF(),
		Progress:      tui.ProgressState{ServersTotal: len(bench.RunnableGroups(plan)), JobsTotal: plan.EstimatedRuns},
		StatusMessage: "Starting benchmark.",
	}
	return tui.Run(ctx, initial, func(runCtx context.Context, updates chan<- tui.StateUpdate) error {
		return r.runBenchmark(runCtx, updates, plan, time.Now())
	})
}

func (r *Runner) runBenchmark(ctx context.Context, updates chan<- tui.StateUpdate, plan domain.BenchmarkPlan, started time.Time) error {
	groups := bench.RunnableGroups(plan)
	jobsCompleted := 0
	adapter := newTUIAdapter(updates, started)
	for groupIndex, group := range groups {
		if err := ctx.Err(); err != nil {
			return err
		}
		if len(group) == 0 {
			continue
		}
		if err := r.runGroup(ctx, adapter, group, groupIndex+1, len(groups), &jobsCompleted); err != nil {
			return err
		}
	}
	adapter.CompleteBenchmark(len(groups))
	return nil
}

func (r *Runner) runGroup(ctx context.Context, adapter tuiAdapter, group []domain.PlannedRun, groupIndex int, totalGroups int, jobsCompleted *int) error {
	server, err := r.prepareServer(group, groupIndex)
	if err != nil {
		return err
	}
	adapter.BeginGroup(server, group, groupIndex, totalGroups)
	process, err := startServer(ctx, r.Config, llamacpp.NewClient(nil), server)
	if err != nil {
		return r.recordStartupFailure(group, server, err.Error(), jobsCompleted, adapter)
	}
	defer process.Terminate()
	executor := requestExecutor{runner: r, client: llamacpp.NewClient(nil)}
	for i, item := range group {
		if err := ctx.Err(); err != nil {
			return err
		}
		adapter.BeginPrompt(item, i)
		result, err := executor.Execute(ctx, item, server, *jobsCompleted+1)
		if err != nil {
			return err
		}
		*jobsCompleted++
		adapter.CompletePrompt(item, result, *jobsCompleted)
	}
	adapter.CompleteGroup(groupIndex, totalGroups)
	return nil
}

func (r *Runner) recordStartupFailure(group []domain.PlannedRun, server serverExecution, errText string, jobsCompleted *int, adapter tuiAdapter) error {
	for _, item := range group {
		runID := fmt.Sprintf("%s-%s", server.ID, item.Job.PromptProfile.Name)
		duration := 0.0
		summary := domain.RunSummary{
			RunID:          runID,
			BatchID:        server.ID,
			CreatedAt:      nowISO(),
			Model:          r.Config.ModelHF(),
			PromptProfile:  item.Job.PromptProfile.Name,
			ServerConfig:   item.Job.ServerConfig,
			Status:         domain.NewRunStatusInfo(domain.StatusFailed, errText),
			ConfigHash:     item.ConfigHash,
			DurationSec:    duration,
			Command:        server.CmdArgs,
			CommandShell:   llamacpp.CommandToShell(server.CmdArgs),
			LlamaCpp:       r.Features.LlamaCpp,
			Backend:        "server",
			RawLogPath:     server.RawPath,
			ServerLogPath:  server.RawPath,
			PromptFile:     item.Job.PromptFile,
			GenerationToks: r.Config.Run.GenerationTokens,
			Seed:           r.Config.Run.Seed,
		}
		if _, err := storage.WriteRunSummary(r.ResultsDir, summary); err != nil {
			return fmt.Errorf("write startup failure summary for %s: %w", runID, err)
		}
		*jobsCompleted++
		adapter.StartupFailedPrompt(item, errText, duration, *jobsCompleted)
	}
	return nil
}

func sanitizedRequest(payload map[string]any) map[string]any {
	out := map[string]any{}
	for key, value := range payload {
		if key == "prompt" || key == "messages" {
			continue
		}
		out[key] = value
	}
	return out
}

func nowISO() string {
	return time.Now().UTC().Format(time.RFC3339Nano)
}

func maxRunsLabel(maxRuns int) string {
	if maxRuns <= 0 {
		return "unlimited"
	}
	return fmt.Sprintf("%d", maxRuns)
}

func or(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}

func intValue(value *int) int {
	if value == nil {
		return 0
	}
	return *value
}

func phaseForStatus(status domain.RunStatus) string {
	if status == domain.StatusSuccess {
		return "done"
	}
	return string(status)
}

func estimateETA(done, total int, elapsed float64) float64 {
	if done <= 0 || total <= done {
		return 0
	}
	per := elapsed / float64(done)
	return per * float64(total-done)
}

func ContextLabel(tokens int) string {
	return tuifmt.FormatContextSize(tokens)
}
