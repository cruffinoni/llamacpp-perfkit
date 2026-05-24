// Package bench generates benchmark plans from configuration, feature detection, and previous run results.
package bench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"github.com/cruffinoni/llamacpp-perfkit/internal/config"
	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
	"github.com/cruffinoni/llamacpp-perfkit/internal/llamacpp"
)

// PlanOptions controls benchmark plan generation behavior.
type PlanOptions struct {
	RetryFailed bool
	Force       bool
}

// leftPadInt left-pads an integer with zeros to the given width.
func leftPadInt(value, width int) string {
	text := strconv.Itoa(value)
	for len(text) < width {
		text = "0" + text
	}
	return text
}

// stringPtrValue returns the string value of a pointer, or "" if nil.
func stringPtrValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

// usableKV returns the list of KV cache types that are both configured and supported.
func usableKV(cfg config.Config, features llamacpp.Features) []string {
	if len(features.KV.UsableValues) > 0 {
		return features.KV.UsableValues
	}
	if len(features.KV.SupportedValues) == 0 {
		return cfg.Matrix.KVType
	}
	supported := map[string]bool{}
	for _, value := range features.KV.SupportedValues {
		supported[value] = true
	}
	var out []string
	for _, value := range cfg.Matrix.KVType {
		if supported[value] {
			out = append(out, value)
		}
	}
	return out
}

// fileIdentity returns file metadata used for identity hashing.
func fileIdentity(path string) map[string]any {
	info, err := os.Stat(path)
	if err != nil {
		return map[string]any{"path": path, "exists": false}
	}
	out := map[string]any{"path": filepath.Clean(path), "exists": true, "size": info.Size(), "mtime_ns": info.ModTime().UnixNano()}
	if !info.IsDir() && info.Size() <= 1024*1024 {
		if data, err := os.ReadFile(path); err == nil {
			hash := sha256.Sum256(data)
			out["sha256"] = hex.EncodeToString(hash[:])
		}
	}
	return out
}

// featureSkips aggregates skip reasons from feature detection results.
func featureSkips(features llamacpp.Features) []domain.SkipReason {
	var out []domain.SkipReason
	out = append(out, features.KV.Skipped...)
	out = append(out, features.Spec.Skipped...)
	for _, skip := range features.ExtraArgs.Skipped {
		out = append(out, skip)
	}
	return out
}

// WritePlan serializes a benchmark plan to a JSON file at the given path.
func WritePlan(path string, plan domain.BenchmarkPlan) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create plan directory %s: %w", filepath.Dir(path), err)
	}
	data, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal benchmark plan: %w", err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		return fmt.Errorf("write benchmark plan %s: %w", path, err)
	}
	return nil
}

// ServerGroupKey returns a deterministic string key that groups benchmark jobs sharing the same server configuration.
func ServerGroupKey(job domain.BenchmarkJob) string {
	payload := map[string]any{
		"model":            job.ServerConfig.Model,
		"ctx":              job.ServerConfig.ContextSize,
		"kv_type":          job.ServerConfig.KVType,
		"n_cpu_moe":        job.ServerConfig.NCPUMOE,
		"mtp":              job.ServerConfig.MTP,
		"batch_size":       job.ServerConfig.BatchSize,
		"ubatch_size":      job.ServerConfig.UBatchSize,
		"spec_type":        job.ServerConfig.SpecType,
		"spec_draft_n_max": job.ServerConfig.SpecDraftNMax,
		"spec_draft_p_min": job.ServerConfig.SpecDraftPMin,
		"parallel":         job.ServerConfig.Parallel,
		"n_gpu_layers":     job.ServerConfig.NGPULayers,
		"split_mode":       job.ServerConfig.SplitMode,
	}
	data, _ := json.Marshal(payload)
	return string(data)
}

// actionForJob determines what action to take for a benchmark job based on existing results and options.
func actionForJob(
	job domain.BenchmarkJob,
	rows map[string]domain.LoadedRun,
	opts PlanOptions,
) (domain.PlanAction, string) {
	if opts.Force {
		return domain.ActionRun, ""
	}
	existing, ok := rows[job.ConfigHash]
	if !ok {
		return domain.ActionRun, ""
	}
	if existing.Summary.Status.Kind() == domain.StatusSuccess {
		return domain.ActionReuse, "successful result already exists: " + existing.Summary.RunID
	}
	if !opts.RetryFailed {
		return domain.ActionSkip, "previous result is " + string(existing.Summary.Status.Kind()) + "; pass --retry-failed to rerun"
	}
	return domain.ActionRun, ""
}

// riskLevel assesses the risk of OOM for a benchmark job based on previous run results.
func riskLevel(job domain.BenchmarkJob, rows []domain.LoadedRun, minHeadroomGB float64) string {
	for _, row := range rows {
		cfg := row.Summary.ServerConfig
		if cfg.KVType != job.ServerConfig.KVType {
			continue
		}
		if row.Summary.Status.Kind() == domain.StatusOOM && job.ServerConfig.ContextSize >= cfg.ContextSize {
			return "high"
		}
		if row.Summary.Status.Kind() == domain.StatusSuccess && row.SystemSummary.MinVRAMFreeMiB != nil && *row.SystemSummary.MinVRAMFreeMiB < minHeadroomGB*1024 && job.ServerConfig.ContextSize >= cfg.ContextSize {
			return "high"
		}
	}
	return "low"
}

// latestRowsByHash groups loaded runs by config hash, keeping only the most recent for each hash.
func latestRowsByHash(rows []domain.LoadedRun) map[string]domain.LoadedRun {
	out := map[string]domain.LoadedRun{}
	for _, row := range rows {
		hash := row.Summary.ConfigHash
		if hash == "" {
			continue
		}
		current, ok := out[hash]
		if !ok || row.Summary.CreatedAt > current.Summary.CreatedAt {
			out[hash] = row
		}
	}
	return out
}

// planRunID generates a formatted plan run identifier with zero-padded index.
func planRunID(index int) string {
	return "plan-" + leftPadInt(index, 4)
}

// ConfigHash computes a short hash that uniquely identifies a benchmark job configuration.
func ConfigHash(cfg config.Config, features llamacpp.Features, job domain.BenchmarkJob) string {
	payload := map[string]any{
		"backend":           "server",
		"llama_cpp_commit":  features.LlamaCpp.Commit,
		"model_hf":          cfg.ModelHF(),
		"server_config":     job.ServerConfig,
		"generation_tokens": cfg.Run.GenerationTokens,
		"endpoint":          cfg.Run.Endpoint,
		"seed":              cfg.Run.Seed,
		"prompt_profile":    job.PromptProfile.Name,
		"prompt":            fileIdentity(job.PromptFile),
		"binary":            fileIdentity(cfg.LlamaServerPath()),
		"server_args":       features.ExtraArgs.ServerUsable,
		"request_args":      features.ExtraArgs.Request,
		"cache_prompt":      cfg.Run.CachePrompt,
	}
	data, _ := json.Marshal(payload)
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])[:16]
}

// uniqueJobs deduplicates benchmark jobs by config hash or server group key.
func uniqueJobs(jobs []domain.BenchmarkJob) []domain.BenchmarkJob {
	seen := map[string]bool{}
	out := make([]domain.BenchmarkJob, 0, len(jobs))
	for _, job := range jobs {
		key := job.ConfigHash
		if key == "" {
			key = ServerGroupKey(job) + "|" + job.PromptProfile.Name
		}
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, job)
	}
	return out
}

// assignServerIndexes assigns sequential server group indices to planned runs.
func assignServerIndexes(items []domain.PlannedRun) {
	indexes := map[string]int{}
	next := 1
	for i := range items {
		key := ServerGroupKey(items[i].Job)
		if _, ok := indexes[key]; !ok {
			indexes[key] = next
			next++
		}
		items[i].ServerIndex = indexes[key]
	}
}

// RunnableGroups groups planned runs by server configuration, returning only those with ActionRun.
func RunnableGroups(plan domain.BenchmarkPlan) [][]domain.PlannedRun {
	byKey := map[string][]domain.PlannedRun{}
	var order []string
	for _, item := range plan.Planned {
		if item.Action != domain.ActionRun {
			continue
		}
		key := ServerGroupKey(item.Job)
		if _, ok := byKey[key]; !ok {
			order = append(order, key)
		}
		byKey[key] = append(byKey[key], item)
	}
	out := make([][]domain.PlannedRun, 0, len(order))
	for _, key := range order {
		out = append(out, byKey[key])
	}
	return out
}

// candidateJobs generates all candidate benchmark jobs from the configuration matrix and feature detection.
func candidateJobs(cfg config.Config, features llamacpp.Features) ([]domain.BenchmarkJob, []domain.SkipReason) {
	var skipped []domain.SkipReason
	kvValues := usableKV(cfg, features)
	if len(kvValues) == 0 {
		skipped = append(skipped, domain.SkipReason{Dimension: "kv_type", Reason: "no configured KV values are supported"})
	}
	specValues := cfg.Matrix.SpecType
	if len(specValues) == 0 {
		specValues = []*string{nil}
	}
	nCPUValues := cfg.Matrix.NCPUMOE
	if features.Flags.LlamaServer.NCPUMOE == "" {
		skipped = append(skipped, domain.SkipReason{Dimension: "n_cpu_moe", Reason: "local llama-server lacks --n-cpu-moe"})
		nCPUValues = []*int{nil}
	}
	extraArgs := config.NormalizeExtraArgs(cfg.Llama.ExtraArgs)
	parallel := config.ExtraArgInt(extraArgs, "--parallel", "-np")
	nGPULayers := config.ExtraArgInt(extraArgs, "--n-gpu-layers", "-ngl")
	splitMode := config.ExtraArgString(extraArgs, "--split-mode")
	profiles := cfg.PromptProfiles()

	var jobs []domain.BenchmarkJob
	for _, profile := range profiles {
		for _, nCPU := range nCPUValues {
			for _, ctx := range cfg.Matrix.ContextSize {
				for _, kv := range kvValues {
					for _, batch := range cfg.Matrix.BatchSize {
						for _, ubatch := range cfg.Matrix.UBatchSize {
							for _, spec := range specValues {
								draftValues := []*int{nil}
								pValues := []*float64{nil}
								if spec != nil {
									if features.Flags.LlamaServer.SpecDraftNMax != "" && len(cfg.Matrix.SpecDraftNMax) > 0 {
										draftValues = cfg.Matrix.SpecDraftNMax
									}
									if features.Flags.LlamaServer.SpecDraftPMin != "" && len(cfg.Matrix.SpecDraftPMin) > 0 {
										pValues = cfg.Matrix.SpecDraftPMin
									}
								}
								for _, draft := range draftValues {
									for _, pMin := range pValues {
										mtp := spec != nil
										jobs = append(jobs, domain.BenchmarkJob{
											PromptProfile: profile,
											PromptFile:    profile.File,
											ServerConfig: domain.ServerConfig{
												Model:           cfg.ModelHF(),
												ContextSize:     ctx,
												KVType:          kv,
												NCPUMOE:         nCPU,
												MTP:             &mtp,
												SpecType:        stringPtrValue(spec),
												SpecDraftNMax:   draft,
												SpecDraftPMin:   pMin,
												BatchSize:       &batch,
												UBatchSize:      &ubatch,
												Parallel:        parallel,
												NGPULayers:      nGPULayers,
												SplitMode:       splitMode,
												Host:            cfg.Llama.Server.Host,
												StartupTimeout:  cfg.Llama.Server.StartupTimeoutSeconds,
												ShutdownTimeout: cfg.Llama.Server.ShutdownTimeoutSeconds,
											},
											GenerationTok: cfg.Run.GenerationTokens,
											Seed:          cfg.Run.Seed,
											CachePrompt:   cfg.Run.CachePrompt,
											Endpoint:      cfg.Run.Endpoint,
										})
									}
								}
							}
						}
					}
				}
			}
		}
	}
	return jobs, append(skipped, featureSkips(features)...)
}

// MakePlan expands the config matrix, deduplicates jobs, and determines actions for each job.
func MakePlan(
	cfg config.Config,
	features llamacpp.Features,
	rows []domain.LoadedRun,
	opts PlanOptions,
) domain.BenchmarkPlan {
	candidates, skipped := candidateJobs(cfg, features)
	for i := range candidates {
		candidates[i].ConfigHash = ConfigHash(cfg, features, candidates[i])
	}
	selected := uniqueJobs(candidates)

	byHash := latestRowsByHash(rows)
	planned := make([]domain.PlannedRun, 0, len(selected))
	estimated := 0
	for i, job := range selected {
		action, reason := actionForJob(job, byHash, opts)
		if action == domain.ActionRun {
			estimated++
		}
		planned = append(planned, domain.PlannedRun{
			RunID:      planRunID(i + 1),
			ConfigHash: job.ConfigHash,
			Action:     action,
			ActionNote: reason,
			RiskLevel:  riskLevel(job, rows, cfg.Run.MinVRAMHeadroomGB),
			Job:        job,
		})
	}
	assignServerIndexes(planned)
	return domain.BenchmarkPlan{
		ReuseExistingResults: cfg.Budget.ReuseExistingResults,
		CandidateCount:       len(candidates),
		SelectedCount:        len(selected),
		EstimatedRuns:        estimated,
		Skipped:              skipped,
		Planned:              planned,
	}
}
