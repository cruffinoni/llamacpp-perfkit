// Package report aggregates benchmark run observations into config-grouped
// summaries and renders them as comparison tables.
package report

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
	"github.com/cruffinoni/llamacpp-perfkit/internal/stats"
	"github.com/cruffinoni/llamacpp-perfkit/internal/storage"
)

// ServerConfigKey is a normalized grouping key derived from a server configuration.
type ServerConfigKey struct {
	Model         string
	ContextSize   int
	KVType        string
	NCPUMOE       *int
	MTP           *bool
	BatchSize     *int
	UBatchSize    *int
	SpecType      string
	SpecDraftNMax *int
	SpecDraftPMin *float64
	Parallel      *int
	NGPULayers    *int
	SplitMode     string
}

// RunObservation is a single benchmark run flattened for aggregation.
type RunObservation struct {
	Run              domain.LoadedRun
	Key              ServerConfigKey
	PromptProfile    string
	Status           domain.RunStatus
	CreatedAt        string
	DurationSeconds  *float64
	GenerationValues []float64
	PromptValues     []float64
	TTFTValues       []float64
	TotalTimeValues  []float64
	FreeVRAMValues   []float64
	PeakVRAMValues   []float64
}

// AggregatedServerConfigReport groups observations that share the same server configuration key.
type AggregatedServerConfigReport struct {
	Key             ServerConfigKey
	Observations    []RunObservation
	TotalRuns       int
	SuccessCount    int
	FailureCount    int
	TimeoutCount    int
	OOMCount        int
	ProfilesSeen    []string
	Score           stats.MetricSummary
	GenerationTokS  stats.MetricSummary
	PromptTokS      stats.MetricSummary
	TTFTSeconds     stats.MetricSummary
	DurationSeconds stats.MetricSummary
	FreeVRAMMiB     stats.MetricSummary
	PeakVRAMMiB     stats.MetricSummary
	LatestCreatedAt string
	Status          string
	Evidence        string
}

// SummaryOptions controls table output formatting.
type SummaryOptions struct {
	Details bool
	Sort    string
	Limit   int
}

// --- Leaf utilities (no local calls) ---

func or(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}

func valueOr(value *float64, fallback float64) float64 {
	if value == nil {
		return fallback
	}
	return *value
}

func minFloat(values []float64) float64 {
	out := values[0]
	for _, value := range values[1:] {
		if value < out {
			out = value
		}
	}
	return out
}

func maxFloat(values []float64) *float64 {
	if len(values) == 0 {
		return nil
	}
	out := values[0]
	for _, value := range values[1:] {
		if value > out {
			out = value
		}
	}
	return &out
}

func firstFloat(values ...*float64) *float64 {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func intPtrString(value *int) string {
	if value == nil {
		return "-"
	}
	return fmt.Sprintf("%d", *value)
}

func floatPtrString(value *float64) string {
	if value == nil {
		return "-"
	}
	return fmt.Sprintf("%g", *value)
}

func padRight(value string, width int) string {
	if len(value) >= width {
		return value
	}
	return value + strings.Repeat(" ", width-len(value))
}

func ptrIntValue(value *int) any {
	if value == nil {
		return nil
	}
	return *value
}

func ptrBoolValue(value *bool) any {
	if value == nil {
		return nil
	}
	return *value
}

func ptrFloatValue(value *float64) any {
	if value == nil {
		return nil
	}
	return *value
}

func llamaValues(samples []domain.LlamaCppMetricSample, pick func(domain.LlamaCppMetricSample) *float64) []float64 {
	out := make([]float64, 0, len(samples))
	for _, sample := range samples {
		if value := pick(sample); value != nil {
			out = append(out, *value)
		}
	}
	return out
}

func systemValues(samples []domain.SystemMetricSample, pick func(domain.SystemMetricSample) *float64) []float64 {
	out := make([]float64, 0, len(samples))
	for _, sample := range samples {
		if value := pick(sample); value != nil {
			out = append(out, *value)
		}
	}
	return out
}

// --- Simple formatters ---

func formatScore(summary stats.MetricSummary) string {
	if summary.Mean == nil {
		return "-"
	}
	return fmt.Sprintf("%.3f", *summary.Mean)
}

func formatMetric(summary stats.MetricSummary) string {
	if summary.GeometricMean == nil {
		return "-"
	}
	text := fmt.Sprintf("g%.1f", *summary.GeometricMean)
	if summary.P10 != nil {
		text += fmt.Sprintf(" p10:%.1f", *summary.P10)
	}
	return text
}

func formatSeconds(summary stats.MetricSummary) string {
	if summary.GeometricMean != nil {
		return fmt.Sprintf("%.2fs", *summary.GeometricMean)
	}
	if summary.Mean != nil {
		return fmt.Sprintf("%.2fs", *summary.Mean)
	}
	return "-"
}

func formatDurationPtr(value *float64) string {
	if value == nil {
		return "-"
	}
	return fmt.Sprintf("%.2fs", *value)
}

func formatVRAM(report AggregatedServerConfigReport, details bool) string {
	if report.FreeVRAMMiB.Min == nil {
		return "-"
	}
	if details && report.FreeVRAMMiB.Mean != nil {
		return fmt.Sprintf("min%.1f mean%.1fG", *report.FreeVRAMMiB.Min/1024, *report.FreeVRAMMiB.Mean/1024)
	}
	return fmt.Sprintf("min %.1fG", *report.FreeVRAMMiB.Min/1024)
}

func statusWeight(status domain.RunStatus) float64 {
	if status == domain.StatusSuccess {
		return 1
	}
	if status == domain.StatusTimeout || status == domain.StatusOOM || status == domain.StatusFailed {
		return 0
	}
	return 0.4
}

// --- Mid-level helpers ---

func mtpSpec(mtp, spec string) string {
	if spec != "" && spec != "none" && spec != "no-mtp" {
		return mtp
	}
	return mtp
}

func configLabel(key ServerConfigKey) string {
	spec := key.SpecType
	if spec == "" {
		spec = "no-mtp"
	}
	mtp := "no-mtp"
	if key.MTP != nil && *key.MTP {
		mtp = "mtp"
	}
	return fmt.Sprintf("moe=%s %s b%s/u%s", intPtrString(key.NCPUMOE), mtpSpec(mtp, spec), intPtrString(key.BatchSize), intPtrString(key.UBatchSize))
}

func pctCell(text string, current, baseline *float64) string {
	if current == nil || baseline == nil || *baseline == 0 {
		return text
	}
	return fmt.Sprintf("%s %+0.1f%%", text, ((*current-*baseline)/(*baseline))*100)
}

func diffCell(text string, current, baseline *float64, unit string, gib bool) string {
	if current == nil || baseline == nil {
		return text
	}
	diff := *current - *baseline
	if gib {
		diff /= 1024
	}
	return fmt.Sprintf("%s %+0.2f%s", text, diff, unit)
}

func reportSortScore(report AggregatedServerConfigReport) float64 {
	score := valueOr(report.Score.Mean, -1)
	gen := valueOr(report.GenerationTokS.GeometricMean, -1)
	return score*1000000 + gen
}

func fillConfigColumns(row map[string]string, key ServerConfigKey) {
	row["model"] = or(key.Model, "-")
	row["ctx"] = fmt.Sprintf("%d", key.ContextSize)
	row["kv"] = or(key.KVType, "-")
	row["moe"] = intPtrString(key.NCPUMOE)
	row["spec"] = or(key.SpecType, "none")
	row["draft"] = intPtrString(key.SpecDraftNMax)
	row["pmin"] = floatPtrString(key.SpecDraftPMin)
	row["batch"] = intPtrString(key.BatchSize)
	row["ubatch"] = intPtrString(key.UBatchSize)
}

func keyString(key ServerConfigKey) string {
	payload := map[string]any{
		"model":            key.Model,
		"ctx":              key.ContextSize,
		"kv_type":          key.KVType,
		"n_cpu_moe":        ptrIntValue(key.NCPUMOE),
		"mtp":              ptrBoolValue(key.MTP),
		"batch_size":       ptrIntValue(key.BatchSize),
		"ubatch_size":      ptrIntValue(key.UBatchSize),
		"spec_type":        key.SpecType,
		"spec_draft_n_max": ptrIntValue(key.SpecDraftNMax),
		"spec_draft_p_min": ptrFloatValue(key.SpecDraftPMin),
		"parallel":         ptrIntValue(key.Parallel),
		"n_gpu_layers":     ptrIntValue(key.NGPULayers),
		"split_mode":       key.SplitMode,
	}
	data, _ := json.Marshal(payload)
	return string(data)
}

// --- Upper-mid helpers ---

func scoreObservation(obs RunObservation, context map[string]*float64) *float64 {
	generation := stats.Summarize(obs.GenerationValues).GeometricMean
	if generation == nil || *generation <= 0 {
		return nil
	}
	speedNorm := 0.0
	if context["fastest_speed"] != nil && *context["fastest_speed"] > 0 {
		speedNorm = *generation / *context["fastest_speed"]
	}
	vramNorm := 0.0
	if len(obs.FreeVRAMValues) > 0 && context["max_vram"] != nil && *context["max_vram"] > 0 {
		vramNorm = math.Max(minFloat(obs.FreeVRAMValues), 0) / *context["max_vram"]
	}
	score := speedNorm*0.70 + statusWeight(obs.Status)*0.20 + vramNorm*0.10
	return &score
}

func aggregateStatus(observations []RunObservation, minHeadroomGB float64) string {
	for _, status := range []domain.RunStatus{domain.StatusTimeout, domain.StatusOOM, domain.StatusFailed, domain.StatusUnsupported} {
		for _, obs := range observations {
			if obs.Status == status {
				return string(status)
			}
		}
	}
	if len(observations) == 0 {
		return "unknown"
	}
	for _, obs := range observations {
		if obs.Status != domain.StatusSuccess {
			return string(obs.Status)
		}
	}
	var free []float64
	for _, obs := range observations {
		free = append(free, obs.FreeVRAMValues...)
	}
	if len(free) > 0 && minFloat(free) < minHeadroomGB*1024 {
		return "tight"
	}
	return "stable"
}

func summaryContext(observations []RunObservation) map[string]*float64 {
	var speeds, vram []float64
	for _, obs := range observations {
		if gm := stats.Summarize(obs.GenerationValues).GeometricMean; gm != nil {
			speeds = append(speeds, *gm)
		}
		if len(obs.FreeVRAMValues) > 0 {
			vram = append(vram, minFloat(obs.FreeVRAMValues))
		}
	}
	return map[string]*float64{"fastest_speed": maxFloat(speeds), "max_vram": maxFloat(vram)}
}

// KeyFromServerConfig converts a domain server config into a grouping key.
func KeyFromServerConfig(cfg domain.ServerConfig) ServerConfigKey {
	return ServerConfigKey{
		Model:         cfg.Model,
		ContextSize:   cfg.ContextSize,
		KVType:        cfg.KVType,
		NCPUMOE:       cfg.NCPUMOE,
		MTP:           cfg.MTP,
		BatchSize:     cfg.BatchSize,
		UBatchSize:    cfg.UBatchSize,
		SpecType:      cfg.SpecType,
		SpecDraftNMax: cfg.SpecDraftNMax,
		SpecDraftPMin: cfg.SpecDraftPMin,
		Parallel:      cfg.Parallel,
		NGPULayers:    cfg.NGPULayers,
		SplitMode:     cfg.SplitMode,
	}
}

// ObservationFromRun converts a loaded run into a flat observation.
func ObservationFromRun(run domain.LoadedRun) RunObservation {
	summary := run.Summary
	duration := summary.DurationSec
	var durationPtr *float64
	if duration > 0 {
		durationPtr = &duration
	}
	generation := llamaValues(run.LlamaMetrics, func(s domain.LlamaCppMetricSample) *float64 { return s.GenerationTokS })
	if len(generation) == 0 && run.LlamaSummary.GenerationTokS != nil {
		generation = append(generation, *run.LlamaSummary.GenerationTokS)
	}
	prompt := llamaValues(run.LlamaMetrics, func(s domain.LlamaCppMetricSample) *float64 { return s.PromptEvalTokS })
	if len(prompt) == 0 && run.LlamaSummary.PromptEvalTokS != nil {
		prompt = append(prompt, *run.LlamaSummary.PromptEvalTokS)
	}
	ttft := llamaValues(run.LlamaMetrics, func(s domain.LlamaCppMetricSample) *float64 { return s.TTFTSeconds })
	totalTime := llamaValues(run.LlamaMetrics, func(s domain.LlamaCppMetricSample) *float64 { return s.TotalTimeSeconds })
	freeVRAM := systemValues(run.SystemMetrics, func(s domain.SystemMetricSample) *float64 { return s.VRAMFree })
	if len(freeVRAM) == 0 && run.SystemSummary.MinVRAMFreeMiB != nil {
		freeVRAM = append(freeVRAM, *run.SystemSummary.MinVRAMFreeMiB)
	}
	peakVRAM := systemValues(run.SystemMetrics, func(s domain.SystemMetricSample) *float64 { return s.VRAMUsed })
	if len(peakVRAM) == 0 && run.SystemSummary.PeakVRAMMiB != nil {
		peakVRAM = append(peakVRAM, *run.SystemSummary.PeakVRAMMiB)
	}
	return RunObservation{
		Run:              run,
		Key:              KeyFromServerConfig(summary.ServerConfig),
		PromptProfile:    or(summary.PromptProfile, "default"),
		Status:           summary.Status.Kind(),
		CreatedAt:        summary.CreatedAt,
		DurationSeconds:  durationPtr,
		GenerationValues: generation,
		PromptValues:     prompt,
		TTFTValues:       ttft,
		TotalTimeValues:  totalTime,
		FreeVRAMValues:   freeVRAM,
		PeakVRAMValues:   peakVRAM,
	}
}

func aggregateOne(
	key ServerConfigKey,
	observations []RunObservation,
	context map[string]*float64,
	minHeadroomGB float64,
) AggregatedServerConfigReport {
	var generation, prompt, ttft, duration, freeVRAM, peakVRAM, scores []float64
	success, timeoutCount, oomCount := 0, 0, 0
	profileSet := map[string]bool{}
	latest := ""
	for _, obs := range observations {
		if obs.Status == domain.StatusSuccess {
			success++
		}
		if obs.Status == domain.StatusTimeout {
			timeoutCount++
		}
		if obs.Status == domain.StatusOOM {
			oomCount++
		}
		profileSet[obs.PromptProfile] = true
		if obs.CreatedAt > latest {
			latest = obs.CreatedAt
		}
		generation = append(generation, obs.GenerationValues...)
		prompt = append(prompt, obs.PromptValues...)
		ttft = append(ttft, obs.TTFTValues...)
		if len(obs.TotalTimeValues) > 0 {
			duration = append(duration, obs.TotalTimeValues...)
		} else if obs.DurationSeconds != nil {
			duration = append(duration, *obs.DurationSeconds)
		}
		freeVRAM = append(freeVRAM, obs.FreeVRAMValues...)
		peakVRAM = append(peakVRAM, obs.PeakVRAMValues...)
		if score := scoreObservation(obs, context); score != nil {
			scores = append(scores, *score)
		}
	}
	profiles := make([]string, 0, len(profileSet))
	for profile := range profileSet {
		profiles = append(profiles, profile)
	}
	sort.Strings(profiles)
	status := aggregateStatus(observations, minHeadroomGB)
	return AggregatedServerConfigReport{
		Key:             key,
		Observations:    observations,
		TotalRuns:       len(observations),
		SuccessCount:    success,
		FailureCount:    len(observations) - success - timeoutCount,
		TimeoutCount:    timeoutCount,
		OOMCount:        oomCount,
		ProfilesSeen:    profiles,
		Score:           stats.Summarize(scores),
		GenerationTokS:  stats.Summarize(generation),
		PromptTokS:      stats.Summarize(prompt),
		TTFTSeconds:     stats.Summarize(ttft),
		DurationSeconds: stats.Summarize(duration),
		FreeVRAMMiB:     stats.Summarize(freeVRAM),
		PeakVRAMMiB:     stats.Summarize(peakVRAM),
		LatestCreatedAt: latest,
		Status:          status,
		Evidence:        fmt.Sprintf("%s (%d/%d)", status, success, len(observations)),
	}
}

func renderTable(columns []string, rows []map[string]string, limit int) string {
	if limit > 0 && len(rows) > limit {
		rows = rows[:limit]
	}
	if len(rows) == 0 {
		return "No rows."
	}
	widths := map[string]int{}
	for _, col := range columns {
		widths[col] = len(col)
		for _, row := range rows {
			if len(row[col]) > widths[col] {
				widths[col] = len(row[col])
			}
		}
	}
	var lines []string
	headerParts := make([]string, len(columns))
	sepParts := make([]string, len(columns))
	for i, col := range columns {
		headerParts[i] = padRight(col, widths[col])
		sepParts[i] = strings.Repeat("-", widths[col])
	}
	lines = append(lines, strings.Join(headerParts, "  "), strings.Join(sepParts, "  "))
	for _, row := range rows {
		parts := make([]string, len(columns))
		for i, col := range columns {
			parts[i] = padRight(row[col], widths[col])
		}
		lines = append(lines, strings.Join(parts, "  "))
	}
	return strings.Join(lines, "\n")
}

func rowFromReport(report AggregatedServerConfigReport, details bool) map[string]string {
	row := map[string]string{
		"score":        formatScore(report.Score),
		"config":       configLabel(report.Key),
		"gen tok/s":    formatMetric(report.GenerationTokS),
		"prompt tok/s": formatMetric(report.PromptTokS),
		"ttft":         formatSeconds(report.TTFTSeconds),
		"time":         formatSeconds(report.DurationSeconds),
		"vram":         formatVRAM(report, details),
		"status":       report.Evidence,
	}
	fillConfigColumns(row, report.Key)
	return row
}

func rowFromObservation(obs RunObservation) map[string]string {
	report := AggregatedServerConfigReport{Key: obs.Key, FreeVRAMMiB: stats.Summarize(obs.FreeVRAMValues)}
	row := map[string]string{
		"profile":      obs.PromptProfile,
		"config":       configLabel(obs.Key),
		"gen tok/s":    formatMetric(stats.Summarize(obs.GenerationValues)),
		"prompt tok/s": formatMetric(stats.Summarize(obs.PromptValues)),
		"time":         formatDurationPtr(obs.DurationSeconds),
		"vram":         formatVRAM(report, false),
		"status":       string(obs.Status),
	}
	fillConfigColumns(row, obs.Key)
	return row
}

func compareRow(
	report AggregatedServerConfigReport,
	baseline AggregatedServerConfigReport,
	details bool,
) map[string]string {
	row := rowFromReport(report, details)
	row["gen tok/s"] = pctCell(row["gen tok/s"], report.GenerationTokS.GeometricMean, baseline.GenerationTokS.GeometricMean)
	row["prompt tok/s"] = pctCell(row["prompt tok/s"], report.PromptTokS.GeometricMean, baseline.PromptTokS.GeometricMean)
	row["ttft"] = diffCell(row["ttft"], firstFloat(report.TTFTSeconds.GeometricMean, report.TTFTSeconds.Mean), firstFloat(baseline.TTFTSeconds.GeometricMean, baseline.TTFTSeconds.Mean), "s", false)
	row["time"] = diffCell(row["time"], firstFloat(report.DurationSeconds.GeometricMean, report.DurationSeconds.Mean), firstFloat(baseline.DurationSeconds.GeometricMean, baseline.DurationSeconds.Mean), "s", false)
	row["vram"] = diffCell(row["vram"], report.FreeVRAMMiB.Min, baseline.FreeVRAMMiB.Min, "G", true)
	return row
}

// --- Top-level exported functions ---

// Load reads benchmark runs from the given path.
func Load(path string) ([]domain.LoadedRun, error) {
	return storage.LoadRuns(path)
}

// Aggregate groups loaded runs by server configuration key and returns
// aggregated reports along with summary context.
func Aggregate(rows []domain.LoadedRun, minHeadroomGB float64) ([]AggregatedServerConfigReport, map[string]*float64) {
	observations := make([]RunObservation, 0, len(rows))
	for _, row := range rows {
		observations = append(observations, ObservationFromRun(row))
	}
	context := summaryContext(observations)
	byKey := map[string][]RunObservation{}
	keys := map[string]ServerConfigKey{}
	var order []string
	for _, obs := range observations {
		keyID := keyString(obs.Key)
		if _, ok := byKey[keyID]; !ok {
			order = append(order, keyID)
			keys[keyID] = obs.Key
		}
		byKey[keyID] = append(byKey[keyID], obs)
	}
	out := make([]AggregatedServerConfigReport, 0, len(order))
	for _, keyID := range order {
		out = append(out, aggregateOne(keys[keyID], byKey[keyID], context, minHeadroomGB))
	}
	return out, context
}

// SortReports sorts aggregated reports by the given sort key.
func SortReports(reports []AggregatedServerConfigReport, sortKey string) {
	sort.SliceStable(reports, func(i, j int) bool {
		a, b := reports[i], reports[j]
		switch sortKey {
		case "latest":
			return a.LatestCreatedAt > b.LatestCreatedAt
		case "generation_tok_s":
			return valueOr(a.GenerationTokS.GeometricMean, -1) > valueOr(b.GenerationTokS.GeometricMean, -1)
		case "vram_headroom":
			return valueOr(a.FreeVRAMMiB.Min, -1) > valueOr(b.FreeVRAMMiB.Min, -1)
		case "peak_vram":
			return valueOr(a.PeakVRAMMiB.Max, -1) > valueOr(b.PeakVRAMMiB.Max, -1)
		case "context_size":
			return a.Key.ContextSize > b.Key.ContextSize
		default:
			return reportSortScore(a) > reportSortScore(b)
		}
	})
}

// EnforcePromptProfileComparability ensures all reports cover the same prompt profiles.
func EnforcePromptProfileComparability(reports []AggregatedServerConfigReport) error {
	if len(reports) < 2 {
		return nil
	}
	baseline := strings.Join(reports[0].ProfilesSeen, "\x00")
	for _, report := range reports[1:] {
		if strings.Join(report.ProfilesSeen, "\x00") != baseline {
			return fmt.Errorf("cannot compare configs: prompt profile sets differ\nbaseline profiles: %v\ncandidate profiles: %v", reports[0].ProfilesSeen, report.ProfilesSeen)
		}
	}
	return nil
}

// PrintSummary writes a summary table of aggregated reports to w.
func PrintSummary(w io.Writer, rows []domain.LoadedRun, opts SummaryOptions) {
	reports, _ := Aggregate(rows, 1.5)
	SortReports(reports, or(opts.Sort, "balanced"))
	fmt.Fprintf(w, "Groups: %d from %d runs\n", len(reports), len(rows))
	columns := []string{"score", "config", "gen tok/s", "prompt tok/s", "ttft", "time", "vram", "status"}
	if opts.Details {
		columns = []string{"score", "model", "ctx", "kv", "moe", "spec", "draft", "pmin", "batch", "ubatch", "gen tok/s", "prompt tok/s", "ttft", "time", "vram", "status"}
	}
	var tableRows []map[string]string
	for _, report := range reports {
		tableRows = append(tableRows, rowFromReport(report, opts.Details))
	}
	fmt.Fprintln(w, renderTable(columns, tableRows, opts.Limit))
}

// PrintByProfile writes per-profile tables of run observations to w.
func PrintByProfile(w io.Writer, rows []domain.LoadedRun, opts SummaryOptions) {
	columns := []string{"profile", "config", "gen tok/s", "prompt tok/s", "time", "vram", "status"}
	if opts.Details {
		columns = []string{"profile", "model", "ctx", "kv", "moe", "spec", "draft", "pmin", "batch", "ubatch", "gen tok/s", "prompt tok/s", "time", "vram", "status"}
	}
	byProfile := map[string][]RunObservation{}
	var profiles []string
	for _, row := range rows {
		obs := ObservationFromRun(row)
		if _, ok := byProfile[obs.PromptProfile]; !ok {
			profiles = append(profiles, obs.PromptProfile)
		}
		byProfile[obs.PromptProfile] = append(byProfile[obs.PromptProfile], obs)
	}
	sort.Strings(profiles)
	for i, profile := range profiles {
		if i > 0 {
			fmt.Fprintln(w)
		}
		fmt.Fprintln(w, profile)
		var tableRows []map[string]string
		for _, obs := range byProfile[profile] {
			tableRows = append(tableRows, rowFromObservation(obs))
		}
		fmt.Fprintln(w, renderTable(columns, tableRows, opts.Limit))
	}
}

// PrintCompare writes a comparison table of candidate versus baseline reports to w.
func PrintCompare(
	w io.Writer,
	baselineRows []domain.LoadedRun,
	candidateRows []domain.LoadedRun,
	opts SummaryOptions,
) error {
	baselineReports, _ := Aggregate(baselineRows, 1.5)
	candidateReports, _ := Aggregate(candidateRows, 1.5)
	SortReports(baselineReports, "balanced")
	SortReports(candidateReports, "balanced")
	if len(baselineReports) == 0 {
		return fmt.Errorf("baseline has no runs")
	}
	if len(candidateReports) == 0 {
		return fmt.Errorf("candidate has no runs")
	}
	all := append([]AggregatedServerConfigReport{baselineReports[0]}, candidateReports...)
	if err := EnforcePromptProfileComparability(all); err != nil {
		return err
	}
	columns := []string{"score", "config", "gen tok/s", "prompt tok/s", "ttft", "time", "vram", "status"}
	if opts.Details {
		columns = []string{"score", "model", "ctx", "kv", "moe", "spec", "draft", "pmin", "batch", "ubatch", "gen tok/s", "prompt tok/s", "ttft", "time", "vram", "status"}
	}
	var tableRows []map[string]string
	tableRows = append(tableRows, rowFromReport(baselineReports[0], opts.Details))
	for _, candidate := range candidateReports {
		tableRows = append(tableRows, compareRow(candidate, baselineReports[0], opts.Details))
	}
	fmt.Fprintln(w, renderTable(columns, tableRows, opts.Limit))
	return nil
}
