package views

import "testing"

func TestUpsertPromptPreservesOrder(t *testing.T) {
	state := BenchmarkTUIState{}
	state.UpsertPrompt(PromptJobView{Profile: "qa", Status: "pending", Phase: "-"})
	state.UpsertPrompt(PromptJobView{Profile: "code", Status: "pending", Phase: "-"})
	state.UpsertPrompt(PromptJobView{Profile: "qa", Status: "running", Phase: "generating"})
	if got := []string{state.PromptJobs[0].Profile, state.PromptJobs[1].Profile}; got[0] != "qa" || got[1] != "code" {
		t.Fatalf("order = %v", got)
	}
	if state.PromptJobs[0].Status != "running" {
		t.Fatalf("status = %s", state.PromptJobs[0].Status)
	}
}
