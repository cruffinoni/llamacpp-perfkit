package cli

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/components"
)

func TestDevTUIOnceCommand(t *testing.T) {
	t.Setenv("LLAMACPP_PERFKIT_TUI_ONCE", "1")
	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"dev", "tui"})
	if err := cmd.ExecuteContext(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "dev tui simulation") {
		t.Fatalf("unexpected output: %q", out.String())
	}
	_ = os.Getenv("LLAMACPP_PERFKIT_TUI_ONCE")
}

func TestDevTUIWithBarStyle(t *testing.T) {
	t.Setenv("LLAMACPP_PERFKIT_TUI_ONCE", "1")
	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"dev", "tui", "--style", "block"})
	if err := cmd.ExecuteContext(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestDevProgressBarDefaultFlags(t *testing.T) {
	cmd := devCommand()
	progressBar, _, err := cmd.Find([]string{"progress-bar"})
	if err != nil {
		t.Fatal(err)
	}
	if err := progressBar.ParseFlags(nil); err != nil {
		t.Fatal(err)
	}
	if got := progressBar.Flags().Lookup("step").Value.String(); got != "1" {
		t.Fatalf("default step = %q, want 1", got)
	}
	if got := progressBar.Flags().Lookup("interval").Value.String(); got != "300ms" {
		t.Fatalf("default interval = %q, want 300ms", got)
	}
	if got := progressBar.Flags().Lookup("max-steps").Value.String(); got != "100" {
		t.Fatalf("default max-steps = %q, want 100", got)
	}
}

func TestDevProgressBarCustomFlags(t *testing.T) {
	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{
		"dev", "progress-bar",
		"--style", "block",
		"--step", "25",
		"--interval", "1ns",
		"--max-steps", "50",
	})
	if err := cmd.ExecuteContext(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "50/50") {
		t.Fatalf("preview should reach 50/50, got: %q", out.String())
	}
}

func TestDevProgressBarRejectsInvalidOptions(t *testing.T) {
	tests := map[string][]string{
		"zero step":          {"--step", "0"},
		"negative step":      {"--step", "-1"},
		"zero interval":      {"--interval", "0s"},
		"negative interval":  {"--interval", "-1ms"},
		"zero max steps":     {"--max-steps", "0"},
		"negative max steps": {"--max-steps", "-1"},
	}

	for name, args := range tests {
		t.Run(name, func(t *testing.T) {
			cmd := NewRootCommand()
			var out bytes.Buffer
			cmd.SetOut(&out)
			cmd.SetArgs(append([]string{"dev", "progress-bar"}, args...))
			if err := cmd.ExecuteContext(context.Background()); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestDevProgressBarWithAllStyles(t *testing.T) {
	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{
		"dev", "progress-bar",
		"--style", "all",
		"--interval", "1ns",
		"--max-steps", "1",
	})
	if err := cmd.ExecuteContext(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "block") {
		t.Error("all preview should contain block style")
	}
	if !strings.Contains(out.String(), "braille") {
		t.Error("all preview should contain braille style")
	}
}

func TestDevTUIInvalidBarStyle(t *testing.T) {
	t.Setenv("LLAMACPP_PERFKIT_TUI_ONCE", "1")
	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"dev", "tui", "--style", "invalid"})
	err := cmd.ExecuteContext(context.Background())
	if err == nil {
		t.Fatal("expected error for invalid bar style")
	}
}

func TestProgressPreviewModelReachesMax(t *testing.T) {
	model := newProgressPreviewModel(progressPreviewOptions{
		Styles:   []components.ProgressBarStyle{components.ProgressBarStyleBlock},
		Step:     25,
		Interval: time.Nanosecond,
		MaxSteps: 100,
	})

	model = advanceProgressPreview(t, model)
	if got := model.View().Content; !strings.Contains(got, "100/100") {
		t.Fatalf("view should reach 100/100, got: %q", got)
	}
}

func TestProgressPreviewModelClampsNonDivisibleStep(t *testing.T) {
	model := newProgressPreviewModel(progressPreviewOptions{
		Styles:   []components.ProgressBarStyle{components.ProgressBarStyleBlock},
		Step:     30,
		Interval: time.Nanosecond,
		MaxSteps: 100,
	})

	model = advanceProgressPreview(t, model)
	if got := model.View().Content; !strings.Contains(got, "100/100") {
		t.Fatalf("view should clamp to 100/100, got: %q", got)
	}
}

func TestProgressPreviewModelStepGreaterThanMax(t *testing.T) {
	model := newProgressPreviewModel(progressPreviewOptions{
		Styles:   []components.ProgressBarStyle{components.ProgressBarStyleBlock},
		Step:     25,
		Interval: time.Nanosecond,
		MaxSteps: 10,
	})

	if got := model.View().Content; !strings.Contains(got, "0/10") {
		t.Fatalf("initial view should start at 0/10, got: %q", got)
	}
	updated, _ := model.Update(progressPreviewTickMsg{})
	model, ok := updated.(progressPreviewModel)
	if !ok {
		t.Fatalf("updated model has type %T", updated)
	}
	if got := model.View().Content; !strings.Contains(got, "10/10") {
		t.Fatalf("view should clamp to 10/10, got: %q", got)
	}
}

func TestProgressPreviewModelCtrlCQuits(t *testing.T) {
	model := newProgressPreviewModel(progressPreviewOptions{
		Styles:   []components.ProgressBarStyle{components.ProgressBarStyleBlock},
		Step:     1,
		Interval: time.Nanosecond,
		MaxSteps: 100,
	})

	updated, cmd := model.Update(tea.KeyPressMsg(tea.Key{Code: 'c', Mod: tea.ModCtrl}))
	if _, ok := updated.(progressPreviewModel); !ok {
		t.Fatalf("updated model has type %T", updated)
	}
	if cmd == nil {
		t.Fatal("expected quit command")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Fatal("expected quit message")
	}
}

func TestProgressPreviewModelEndsWithBlankLine(t *testing.T) {
	model := newProgressPreviewModel(progressPreviewOptions{
		Styles:   []components.ProgressBarStyle{components.ProgressBarStyleBlock},
		Step:     1,
		Interval: time.Nanosecond,
		MaxSteps: 100,
	})

	if got := model.View().Content; !strings.HasSuffix(got, "\n\n") {
		t.Fatalf("view should end with a blank line, got: %q", got)
	}
}

func advanceProgressPreview(t *testing.T, model progressPreviewModel) progressPreviewModel {
	t.Helper()
	for model.completed < model.opts.MaxSteps {
		updated, _ := model.Update(progressPreviewTickMsg{})
		next, ok := updated.(progressPreviewModel)
		if !ok {
			t.Fatalf("updated model has type %T", updated)
		}
		model = next
	}
	return model
}
