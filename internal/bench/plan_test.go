package bench

import (
	"testing"

	"github.com/cruffinoni/llamacpp-perfkit/internal/config"
	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
	"github.com/cruffinoni/llamacpp-perfkit/internal/llamacpp"
)

func TestServerGroupKeyIgnoresPromptProfile(t *testing.T) {
	a := domain.BenchmarkJob{PromptProfile: domain.PromptProfile{Name: "code"}, ServerConfig: domain.ServerConfig{ContextSize: 2048, KVType: "q8_0", NCPUMOE: new(4), BatchSize: new(128), UBatchSize: new(32)}}
	b := a
	b.PromptProfile = domain.PromptProfile{Name: "qa"}
	if ServerGroupKey(a) != ServerGroupKey(b) {
		t.Fatal("prompt profile should not be part of server group key")
	}
}

func TestMakePlanSelectsAllCandidates(t *testing.T) {
	cfg := config.Defaults()
	cfg.Models.HF = "model:A"
	cfg.Matrix.KVType = []string{"q8_0"}
	cfg.Prompt.Profiles = []config.ProfileRef{{Name: "code", File: "prompts/default.txt"}}
	features := llamacpp.Features{
		Flags: llamacpp.FeatureFlags{LlamaServer: llamacpp.ServerFlags{NCPUMOE: "--n-cpu-moe", SpecDraftNMax: "--spec-draft-n-max", SpecDraftPMin: "--spec-draft-p-min"}},
		KV:    llamacpp.ValuesFeature{UsableValues: []string{"q8_0"}},
	}
	plan := MakePlan(cfg, features, nil, PlanOptions{})
	if plan.CandidateCount == 0 || plan.SelectedCount != plan.CandidateCount {
		t.Fatalf("bad plan: %+v", plan)
	}
}
