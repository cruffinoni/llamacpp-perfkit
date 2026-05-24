package runner

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
	"github.com/cruffinoni/llamacpp-perfkit/internal/llamacpp"
	"github.com/cruffinoni/llamacpp-perfkit/internal/metrics"
	"github.com/cruffinoni/llamacpp-perfkit/internal/storage"
	metricsummary "github.com/cruffinoni/llamacpp-perfkit/internal/summary"
)

type requestExecutor struct {
	runner *Runner
	client llamacpp.Client
}

type requestResult struct {
	Summary         domain.RunSummary
	Loaded          domain.LoadedRun
	Status          domain.RunStatus
	Duration        float64
	CollectorErrors []error
}

func (e requestExecutor) Execute(ctx context.Context, item domain.PlannedRun, server serverExecution, ordinal int) (requestResult, error) {
	r := e.runner
	runID := fmt.Sprintf("%d-%04d-%04d", time.Now().Unix(), item.ServerIndex, ordinal)
	runDir := storage.RunDir(r.ResultsDir, runID)
	if err := os.MkdirAll(storage.MetricsDir(runDir), 0o755); err != nil {
		return requestResult{}, fmt.Errorf("create metrics directory %s: %w", storage.MetricsDir(runDir), err)
	}

	start := time.Now()
	payload, err := llamacpp.BuildRequestPayload(r.Config, r.Features, item.Job)
	if err != nil {
		return requestResult{}, fmt.Errorf("build request payload for %s: %w", item.Job.PromptFile, err)
	}
	requestForSummary := sanitizedRequest(payload)

	stopCollectors := e.startCollectors(ctx, runDir, server.BaseURL)
	reqCtx, cancelReq := context.WithTimeout(ctx, time.Duration(r.Config.Run.TimeoutSeconds)*time.Second)
	endpoint := server.BaseURL + llamacpp.EndpointPath(llamacpp.EndpointKind(r.Config))
	resp, requestErr := e.client.JSON(reqCtx, http.MethodPost, endpoint, payload)
	cancelReq()
	collectorErrors := stopCollectors()
	if err := storage.AppendSystemMetric(runDir, metrics.SampleSystem()); err != nil {
		collectorErrors = append(collectorErrors, fmt.Errorf("append final system metric: %w", err))
	}

	status, errText := classifyRequest(reqCtx, resp.StatusCode, resp.Text, requestErr)
	parsed, response := parseRequestResult(r, status, resp.Body, resp.Text, server.RawPath, time.Since(start).Seconds())
	duration := time.Since(start).Seconds()
	summary := domain.RunSummary{
		RunID:          runID,
		BatchID:        server.ID,
		CreatedAt:      nowISO(),
		Model:          r.Config.ModelHF(),
		PromptProfile:  item.Job.PromptProfile.Name,
		ServerConfig:   item.Job.ServerConfig,
		Status:         domain.NewRunStatusInfo(status, errText),
		ConfigHash:     item.ConfigHash,
		DurationSec:    duration,
		Request:        requestForSummary,
		Response:       response,
		Parsed:         parsed,
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
		return requestResult{}, fmt.Errorf("write run summary for %s: %w", runID, err)
	}
	if err := storage.AppendLlamaMetric(runDir, metricsummary.TerminalLlamaSample(summary.CreatedAt, parsed, duration)); err != nil {
		return requestResult{}, fmt.Errorf("append terminal llama metric for %s: %w", runID, err)
	}
	loaded, err := storage.ReadRun(runDir)
	if err != nil {
		return requestResult{}, fmt.Errorf("read run summary for %s: %w", runID, err)
	}
	return requestResult{
		Summary:         summary,
		Loaded:          loaded,
		Status:          status,
		Duration:        duration,
		CollectorErrors: collectorErrors,
	}, nil
}

func (e requestExecutor) startCollectors(ctx context.Context, runDir string, baseURL string) func() []error {
	interval := time.Duration(e.runner.Config.Run.MonitorIntervalSeconds * float64(time.Second))
	collectCtx, cancelCollect := context.WithCancel(ctx)
	errCh := make(chan error, 2)
	go func() {
		errCh <- metrics.SystemCollector{RunDir: runDir, Interval: interval}.Run(collectCtx)
	}()
	go func() {
		errCh <- metrics.LlamaCollector{BaseURL: baseURL, RunDir: runDir, Interval: interval}.Run(collectCtx)
	}()
	return func() []error {
		cancelCollect()
		var errs []error
		for range 2 {
			if err := <-errCh; err != nil {
				errs = append(errs, err)
			}
		}
		return errs
	}
}

func classifyRequest(ctx context.Context, statusCode int, responseText string, err error) (domain.RunStatus, string) {
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return domain.StatusTimeout, ctx.Err().Error()
		}
		return domain.StatusFailed, err.Error()
	}
	if statusCode == http.StatusOK {
		return domain.StatusSuccess, ""
	}
	return llamacpp.ClassifyHTTPError(statusCode, responseText), responseText
}

func parseRequestResult(r *Runner, status domain.RunStatus, responseBody map[string]any, responseText string, rawPath string, elapsedSeconds float64) (map[string]any, map[string]any) {
	if status == domain.StatusSuccess && llamacpp.EndpointKind(r.Config) == "chat" {
		return llamacpp.ParseChatCompletion(responseBody, elapsedSeconds, llamacpp.LogText(rawPath))
	}
	if status == domain.StatusSuccess {
		return llamacpp.ParseCompletion(responseBody)
	}
	parsed := llamacpp.ParseLlamaOutput(responseText)
	response := map[string]any{"content": responseBody["content"], "stop_type": responseBody["stop_type"], "truncated": responseBody["truncated"]}
	return parsed, response
}
