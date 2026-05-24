package llamacpp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/cruffinoni/llamacpp-perfkit/internal/config"
	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
)

var searchTerms = []string{
	"n-cpu-moe", "cache-type", "cache-type-k", "cache-type-v", "kv", "turbo",
	"spec", "draft", "mtp", "ngram", "hfd", "ctx", "n-ctx", "seed", "predict", "tokens",
}

var toolkitOwnedFlags = map[string]bool{
	"-hf": true, "--hf-repo": true, "-m": true, "--model": true,
	"--ctx-size": true, "--n-ctx": true, "--context-size": true,
	"--predict": true, "--n-predict": true, "--file": true, "--prompt": true,
	"--seed": true, "--n-cpu-moe": true, "--cache-type": true,
	"--cache-type-k": true, "--cache-type-v": true, "--spec-type": true,
	"--spec-draft-n-max": true, "--spec-draft-p-min": true,
	"--log-file": true, "--no-warmup": true, "--host": true, "--port": true,
	"--no-webui": true, "--batch-size": true, "--ubatch-size": true,
}

var requestArgMap = map[string]string{
	"--temp": "temperature", "--temperature": "temperature",
	"--top-p": "top_p", "--top-k": "top_k",
	"--presence-penalty": "presence_penalty", "--frequency-penalty": "frequency_penalty",
	"--min-p": "min_p", "--typical": "typical_p", "--typical-p": "typical_p",
	"--repeat-penalty": "repeat_penalty", "--repeat-last-n": "repeat_last_n",
	"--seed": "seed",
}

// Features holds the complete set of detected llama.cpp features for a binary directory.
type Features struct {
	CreatedAt     string                 `json:"created_at"`
	ProjectRoot   string                 `json:"project_root,omitempty"`
	BinDir        string                 `json:"bin_dir"`
	LlamaCpp      domain.BuildInfo       `json:"llama_cpp"`
	Backend       string                 `json:"backend"`
	Binaries      map[string]BinaryInfo  `json:"binaries"`
	Flags         FeatureFlags           `json:"flags"`
	KV            ValuesFeature          `json:"kv"`
	Spec          ValuesFeature          `json:"spec"`
	ExtraArgs     ExtraArgsFeature       `json:"extra_args"`
	HelpExcerpt   map[string]string      `json:"help_excerpt,omitempty"`
	ValidForBench bool                   `json:"valid_for_bench"`
	InvalidReason string                 `json:"invalid_reason,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// BinaryInfo describes a single llama.cpp binary and its capabilities.
type BinaryInfo struct {
	Path           string `json:"path"`
	Exists         bool   `json:"exists"`
	Usable         bool   `json:"usable"`
	HelpReturnCode int    `json:"help_returncode"`
	HelpError      string `json:"help_error,omitempty"`
}

// FeatureFlags collects the flag availability for llama-bench and llama-server.
type FeatureFlags struct {
	LlamaBench  map[string]string `json:"llama_bench"`
	LlamaServer ServerFlags       `json:"llama_server"`
}

// ServerFlags tracks which server-side flags are available in the local binary.
type ServerFlags struct {
	HF            string `json:"hf,omitempty"`
	Context       string `json:"context,omitempty"`
	Generation    string `json:"generation,omitempty"`
	Seed          string `json:"seed,omitempty"`
	NCPUMOE       string `json:"n_cpu_moe,omitempty"`
	CacheTypeK    string `json:"cache_type_k,omitempty"`
	CacheTypeV    string `json:"cache_type_v,omitempty"`
	SpecType      string `json:"spec_type,omitempty"`
	SpecDraftNMax string `json:"spec_draft_n_max,omitempty"`
	SpecDraftPMin string `json:"spec_draft_p_min,omitempty"`
	NoWebUI       string `json:"no_webui,omitempty"`
	Host          string `json:"host,omitempty"`
	Port          string `json:"port,omitempty"`
	Metrics       string `json:"metrics,omitempty"`
	Slots         string `json:"slots,omitempty"`
	BatchSize     string `json:"batch_size,omitempty"`
	UBatchSize    string `json:"ubatch_size,omitempty"`
}

// ValuesFeature captures supported, requested, and usable values for a feature dimension.
type ValuesFeature struct {
	SupportedValues []string            `json:"supported_values,omitempty"`
	RequestedValues []string            `json:"requested_values,omitempty"`
	UsableValues    []string            `json:"usable_values,omitempty"`
	Skipped         []domain.SkipReason `json:"skipped,omitempty"`
}

// ExtraArgsFeature captures extra argument handling results.
type ExtraArgsFeature struct {
	Requested    []config.ExtraArg   `json:"requested,omitempty"`
	Usable       []config.ExtraArg   `json:"usable,omitempty"`
	ServerUsable []config.ExtraArg   `json:"server_usable,omitempty"`
	Request      map[string]any      `json:"request"`
	Skipped      []domain.SkipReason `json:"skipped,omitempty"`
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func or(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func nowISO() string {
	return time.Now().UTC().Format(time.RFC3339Nano)
}

func hasFlag(help, flag string) bool {
	re := regexp.MustCompile(`(^|[\s,])` + regexp.QuoteMeta(flag) + `([\s,]|$)`)
	return re.FindStringIndex(help) != nil
}

func nearestExistingPath(path string) string {
	for path != "" {
		if _, err := os.Stat(path); err == nil {
			return path
		}
		parent := filepath.Dir(path)
		if parent == path {
			return path
		}
		path = parent
	}
	return "."
}

func stringPointers(values []*string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value != nil {
			out = append(out, *value)
		}
	}
	return out
}

func interestingHelp(help string) string {
	var out []string
	for _, line := range strings.Split(help, "\n") {
		lower := strings.ToLower(line)
		for _, term := range searchTerms {
			if strings.Contains(lower, strings.ToLower(term)) {
				out = append(out, line)
				break
			}
		}
	}
	return strings.Join(out, "\n")
}

func runCapture(parent context.Context, args []string, timeout time.Duration) (int, string, string) {
	ctx, cancel := context.WithTimeout(parent, timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return 124, stdout.String(), "timeout"
		}
		if exitErr, ok := err.(*exec.ExitError); ok {
			return exitErr.ExitCode(), stdout.String(), stderr.String()
		}
		return 127, stdout.String(), err.Error()
	}
	return 0, stdout.String(), stderr.String()
}

func valueFeature(requested, supported []string, reason string) ValuesFeature {
	usable := make([]string, 0, len(requested))
	var skipped []domain.SkipReason
	supportedSet := map[string]bool{}
	for _, value := range supported {
		supportedSet[value] = true
	}
	for _, value := range requested {
		if value == "" {
			continue
		}
		if len(supportedSet) == 0 || supportedSet[value] {
			usable = append(usable, value)
		} else {
			skipped = append(skipped, domain.SkipReason{Value: value, Reason: reason})
		}
	}
	return ValuesFeature{SupportedValues: supported, RequestedValues: requested, UsableValues: usable, Skipped: skipped}
}

func firstFlag(help string, flags ...string) string {
	for _, flag := range flags {
		if hasFlag(help, flag) {
			return flag
		}
	}
	return ""
}

func allowedValues(help, flag string) []string {
	lines := strings.Split(help, "\n")
	set := map[string]bool{}
	for i, line := range lines {
		if !strings.Contains(line, flag) {
			continue
		}
		block := strings.Join(lines[i:min(i+5, len(lines))], "\n")
		match := regexp.MustCompile(`(?i)allowed values:\s*([^\n]+)`).FindStringSubmatch(block)
		if len(match) == 2 {
			for _, value := range strings.Split(match[1], ",") {
				value = strings.Trim(value, " ,")
				if value != "" && !strings.HasPrefix(value, "<") {
					set[value] = true
				}
			}
		}
		if flag == "--spec-type" {
			inline := regexp.MustCompile(regexp.QuoteMeta(flag) + `\s+([A-Za-z0-9_,\-]+)`).FindStringSubmatch(line)
			if len(inline) == 2 {
				for _, value := range strings.Split(inline[1], ",") {
					value = strings.Trim(value, " ,")
					if value != "" && !strings.HasPrefix(value, "<") {
						set[value] = true
					}
				}
			}
		}
	}
	out := make([]string, 0, len(set))
	for value := range set {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func resolveExtraArgs(serverHelp string, requested []config.ExtraArg) ExtraArgsFeature {
	request := map[string]any{}
	var usable []config.ExtraArg
	var skipped []domain.SkipReason
	for _, item := range requested {
		flag := strings.TrimSpace(item.Flag)
		if flag == "" {
			continue
		}
		if toolkitOwnedFlags[flag] {
			skipped = append(skipped, domain.SkipReason{Flag: flag, Value: item.Value, Reason: "flag is controlled by the benchmark toolkit"})
			continue
		}
		if requestKey, ok := requestArgMap[flag]; ok {
			request[requestKey] = item.Value
			continue
		}
		if !strings.HasPrefix(flag, "-") {
			skipped = append(skipped, domain.SkipReason{Flag: flag, Value: item.Value, Reason: "extra arg key must be a llama.cpp flag"})
			continue
		}
		if !hasFlag(serverHelp, flag) {
			skipped = append(skipped, domain.SkipReason{Flag: flag, Value: item.Value, Reason: "local llama-server --help does not list this flag"})
			continue
		}
		usable = append(usable, item)
	}
	return ExtraArgsFeature{Requested: requested, Usable: usable, ServerUsable: usable, Request: request, Skipped: skipped}
}

// GitMetadata probes a directory for Git repository information.
func GitMetadata(ctx context.Context, path string) domain.BuildInfo {
	probe := nearestExistingPath(path)
	rc, stdout, stderr := runCapture(ctx, []string{"git", "-C", probe, "rev-parse", "--show-toplevel"}, 10*time.Second)
	if rc != 0 {
		return domain.BuildInfo{Error: strings.TrimSpace(or(stderr, stdout, "git rev-parse --show-toplevel failed"))}
	}
	repo := strings.TrimSpace(stdout)
	rc, stdout, stderr = runCapture(ctx, []string{"git", "-C", repo, "rev-parse", "HEAD"}, 10*time.Second)
	if rc != 0 {
		return domain.BuildInfo{Repo: repo, Error: strings.TrimSpace(or(stderr, stdout, "git rev-parse HEAD failed"))}
	}
	commit := strings.TrimSpace(stdout)
	rc, stdout, _ = runCapture(ctx, []string{"git", "-C", repo, "rev-parse", "--short", "HEAD"}, 10*time.Second)
	commitShort := strings.TrimSpace(stdout)
	if rc != 0 {
		commitShort = ""
	}
	rc, stdout, _ = runCapture(ctx, []string{"git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"}, 10*time.Second)
	branch := strings.TrimSpace(stdout)
	if rc != 0 || branch == "HEAD" {
		branch = ""
	}
	return domain.BuildInfo{Repo: repo, Commit: commit, CommitShort: commitShort, Branch: branch}
}

// FreeTCPPort finds a free TCP port on the given host.
func FreeTCPPort(host string) (int, error) {
	listener, err := net.Listen("tcp", net.JoinHostPort(host, "0"))
	if err != nil {
		return 0, err
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port, nil
}

// ServerExtraArgs returns the extra arguments usable by the server.
func ServerExtraArgs(features Features) []config.ExtraArg {
	return features.ExtraArgs.ServerUsable
}

// RequestArgs returns the request-level extra arguments.
func RequestArgs(features Features) map[string]any {
	return features.ExtraArgs.Request
}

// LoadFeatures reads a previously saved feature report from disk.
func LoadFeatures(resultsDir string) (Features, bool, error) {
	path := filepath.Join(resultsDir, "features.json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Features{}, false, nil
		}
		return Features{}, false, fmt.Errorf("read feature cache %s: %w", path, err)
	}
	var features Features
	if err := json.Unmarshal(data, &features); err != nil {
		return Features{}, false, fmt.Errorf("parse feature cache %s: %w", path, err)
	}
	return features, true, nil
}

// WriteFeatures writes the feature report as JSON and a human-readable text file.
func WriteFeatures(features Features, resultsDir string) error {
	if err := os.MkdirAll(resultsDir, 0o755); err != nil {
		return fmt.Errorf("create feature output directory %s: %w", resultsDir, err)
	}
	data, err := json.MarshalIndent(features, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal feature cache: %w", err)
	}
	jsonPath := filepath.Join(resultsDir, "features.json")
	if err := os.WriteFile(jsonPath, append(data, '\n'), 0o644); err != nil {
		return fmt.Errorf("write feature cache %s: %w", jsonPath, err)
	}
	var lines []string
	lines = append(lines,
		"created_at: "+features.CreatedAt,
		"bin_dir: "+features.BinDir,
		"llama_cpp_commit: "+or(features.LlamaCpp.CommitShort, "unknown"),
		"backend: "+features.Backend,
		fmt.Sprintf("valid_for_bench: %v", features.ValidForBench),
	)
	if features.InvalidReason != "" {
		lines = append(lines, "invalid_reason: "+features.InvalidReason)
	}
	lines = append(lines, "", "Binaries:")
	for _, name := range []string{"llama-bench", "llama-server"} {
		item := features.Binaries[name]
		lines = append(lines, fmt.Sprintf("  %s: usable=%v path=%s", name, item.Usable, item.Path))
	}
	lines = append(lines, "", "llama-server flags:")
	serverData, _ := json.MarshalIndent(features.Flags.LlamaServer, "  ", "  ")
	lines = append(lines, strings.Split(string(serverData), "\n")...)
	lines = append(lines, "", "KV:", "  supported_values: "+strings.Join(features.KV.SupportedValues, ", "), "  usable_values: "+strings.Join(features.KV.UsableValues, ", "))
	lines = append(lines, "", "Spec:", "  supported_values: "+strings.Join(features.Spec.SupportedValues, ", "), "  usable_values: "+strings.Join(features.Spec.UsableValues, ", "))
	textPath := filepath.Join(resultsDir, "features.txt")
	if err := os.WriteFile(textPath, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		return fmt.Errorf("write feature report %s: %w", textPath, err)
	}
	return nil
}

// DetectFeatures probes llama.cpp binaries and returns a complete feature report.
func DetectFeatures(ctx context.Context, cfg config.Config) (Features, error) {
	binDir := cfg.LlamaBinDir()
	binaries := map[string]BinaryInfo{}
	helpTexts := map[string]string{}
	for _, name := range []string{"llama-bench", "llama-server"} {
		path := filepath.Join(binDir, name)
		info, err := os.Stat(path)
		exists := err == nil && !info.IsDir() && info.Mode()&0o111 != 0
		rc, stdout, stderr := 127, "", "missing"
		if exists {
			rc, stdout, stderr = runCapture(ctx, []string{path, "--help"}, 20*time.Second)
		}
		binaries[name] = BinaryInfo{Path: path, Exists: exists, Usable: exists && rc == 0, HelpReturnCode: rc, HelpError: strings.TrimSpace(stderr)}
		helpTexts[name] = stdout + "\n" + stderr
	}

	benchHelp := helpTexts["llama-bench"]
	serverHelp := helpTexts["llama-server"]
	serverFlags := ServerFlags{
		HF:            firstFlag(serverHelp, "-hf", "--hf-repo"),
		Context:       firstFlag(serverHelp, "--ctx-size", "--n-ctx", "--context-size"),
		Generation:    firstFlag(serverHelp, "--n-predict", "--predict"),
		Seed:          firstFlag(serverHelp, "--seed"),
		NCPUMOE:       firstFlag(serverHelp, "--n-cpu-moe"),
		CacheTypeK:    firstFlag(serverHelp, "--cache-type-k"),
		CacheTypeV:    firstFlag(serverHelp, "--cache-type-v"),
		SpecType:      firstFlag(serverHelp, "--spec-type"),
		SpecDraftNMax: firstFlag(serverHelp, "--spec-draft-n-max"),
		SpecDraftPMin: firstFlag(serverHelp, "--spec-draft-p-min"),
		NoWebUI:       firstFlag(serverHelp, "--no-webui"),
		Host:          firstFlag(serverHelp, "--host"),
		Port:          firstFlag(serverHelp, "--port"),
		Metrics:       firstFlag(serverHelp, "--metrics"),
		Slots:         firstFlag(serverHelp, "--slots"),
		BatchSize:     firstFlag(serverHelp, "--batch-size"),
		UBatchSize:    firstFlag(serverHelp, "--ubatch-size"),
	}

	requestedExtra := config.NormalizeExtraArgs(cfg.Llama.ExtraArgs)
	extraFeature := resolveExtraArgs(serverHelp, requestedExtra)
	kvValues := allowedValues(serverHelp, "--cache-type-k")
	if len(kvValues) == 0 {
		kvValues = allowedValues(benchHelp, "--cache-type-k")
	}
	specValues := allowedValues(serverHelp, "--spec-type")
	kvFeature := valueFeature(cfg.Matrix.KVType, kvValues, "not listed in local cache-type allowed values")
	specRequested := stringPointers(cfg.Matrix.SpecType)
	specFeature := valueFeature(specRequested, specValues, "not listed in local --spec-type allowed values")
	valid := binaries["llama-server"].Usable && serverFlags.HF != "" && serverFlags.Context != "" && serverFlags.Port != ""

	wd, _ := os.Getwd()
	features := Features{
		CreatedAt:   nowISO(),
		ProjectRoot: wd,
		BinDir:      binDir,
		LlamaCpp:    GitMetadata(ctx, binDir),
		Backend:     "server",
		Binaries:    binaries,
		Flags: FeatureFlags{
			LlamaBench: map[string]string{
				"hf":           firstFlag(benchHelp, "-hf", "--hf-repo"),
				"n_prompt":     firstFlag(benchHelp, "--n-prompt"),
				"n_gen":        firstFlag(benchHelp, "--n-gen"),
				"n_cpu_moe":    firstFlag(benchHelp, "--n-cpu-moe"),
				"cache_type_k": firstFlag(benchHelp, "--cache-type-k"),
				"cache_type_v": firstFlag(benchHelp, "--cache-type-v"),
				"spec_type":    firstFlag(benchHelp, "--spec-type"),
			},
			LlamaServer: serverFlags,
		},
		KV:            kvFeature,
		Spec:          specFeature,
		ExtraArgs:     extraFeature,
		HelpExcerpt:   map[string]string{"llama-bench": interestingHelp(benchHelp), "llama-server": interestingHelp(serverHelp)},
		ValidForBench: valid,
	}
	if !valid {
		features.InvalidReason = "llama-server is missing or lacks -hf/context/port flags"
	}
	return features, nil
}
