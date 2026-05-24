package components

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	tuifmt "github.com/cruffinoni/llamacpp-perfkit/internal/tui/format"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/views"
)

// Header renders the top panel with run metadata.
func BenchmarkHeader(state views.BenchmarkTUIState, s theme.Styles) string {
	bi := state.BuildInfo
	line1 := s.Title.Render("llama-cpp-perfkit") + s.Muted.Render("  run ") + s.Accent.Render(state.RunID)
	line2 := s.Title.Render("llama.cpp: ") + s.Cyan.Render(or(bi.CommitShort, "unknown")) +
		s.TextBold.Render("  ") + s.Muted.Render(or(bi.Branch, "unknown")) + s.TextBold.Render("  ") +
		s.Cyan.Render(or(bi.Backend, "server"))
	line3 := s.Label.Render("Model: ") + s.Title.Render(state.ModelName)
	return s.Panel.Render(strings.Join([]string{line1, line2, line3}, "\n"))
}

// ProgressBlock renders the progress panel with 3 metric bars.
func ProgressBlock(state views.BenchmarkTUIState, s theme.Styles) string {
	p := state.Progress
	summary := fmt.Sprintf("Matrix: %d/%d servers   Jobs: %d/%d   Elapsed: %s   ETA: %s",
		p.ServersCompleted, p.ServersTotal, p.JobsCompleted, p.JobsTotal,
		tuifmt.FormatElapsed(state.ElapsedSeconds), tuifmt.FormatElapsed(state.ETASeconds),
	)
	bar := func(label string, done, total int, suffix string) string {
		return s.Label.Render(label) + " " + s.Cyan.Render(tuifmt.FormatProgress(done, total, 30)) +
			fmt.Sprintf(" %d/%d%s", done, max(total, 1), suffix)
	}
	lines := []string{
		summary,
		bar("Servers", p.ServersCompleted, p.ServersTotal, ""),
		bar("Jobs   ", p.JobsCompleted, p.JobsTotal, ""),
		bar("Current", p.CurrentPrompt, p.CurrentPromptTotal, " prompts"),
	}
	return s.Panel.Render(strings.Join(lines, "\n"))
}

// CurrentServerBlock renders the current server configuration panel.
func CurrentServerBlock(state views.BenchmarkTUIState, s theme.Styles) string {
	if state.CurrentServer == nil {
		return s.Panel.Render(s.Title.Render("Current server") + "\n" + s.Muted.Render("(none)"))
	}
	cs := state.CurrentServer
	line := fmt.Sprintf("ctx=%s  kv=%s  moe=%d  spec=%s  batch=%d  ubatch=%d",
		s.Blue.Render(tuifmt.FormatContextSize(cs.ContextSize)),
		s.Cyan.Render(cs.KVType),
		cs.NCPUMOE,
		s.Yellow.Render(cs.SpecType),
		cs.BatchSize,
		cs.UBatchSize,
	)
	return s.Panel.Render(s.Title.Render("Current server") + "\n" +
		s.Label.Render("id: ") + cs.ID + "\n" + line)
}

// PromptTable renders the prompts table panel with aligned columns.
func PromptTable(state views.BenchmarkTUIState, s theme.Styles) string {
	if len(state.PromptJobs) == 0 {
		return s.Panel.Render(s.Muted.Render("No prompts."))
	}
	headerFmt := "%-18s  %-9s  %-10s  %6s  %10s  %12s  %10s"
	lines := []string{
		s.TextBold.Render(fmt.Sprintf(headerFmt, "profile", "status", "phase", "time", "gen tok/s", "prompt tok/s", "min vram")),
		s.Muted.Render(strings.Repeat("-", 86)),
	}
	for _, job := range state.PromptJobs {
		status := theme.StatusStyle(s, job.Status).Render(job.Status)
		phase := theme.PhaseStyle(s, job.Phase).Render(job.Phase)
		lines = append(lines, fmt.Sprintf(headerFmt,
			job.Profile,
			status,
			phase,
			formatDuration(job.DurationSeconds),
			formatTokS(job.GenTokS, "gen"),
			formatTokS(job.PromptTokS, "prompt"),
			formatVRAM(job.MinVRAMMiB),
		))
	}
	return s.Panel.Render(strings.Join(lines, "\n"))
}

// Layout joins all panels vertically with left alignment.
func Layout(state views.BenchmarkTUIState, s theme.Styles) string {
	return s.Base.Render(lipgloss.JoinVertical(lipgloss.Left,
		BenchmarkHeader(state, s),
		ProgressBlock(state, s),
		CurrentServerBlock(state, s),
		PromptTable(state, s),
		s.Muted.Render(state.StatusMessage),
	))
}

// Utility functions for nil-safe formatting.
func formatDuration(value *float64) string {
	if value == nil {
		return "-"
	}
	return tuifmt.FormatDuration(*value)
}

func formatTokS(value *float64, kind string) string {
	if value == nil {
		return "-"
	}
	return tuifmt.FormatTokS(*value, kind)
}

func formatVRAM(value *float64) string {
	if value == nil {
		return "-"
	}
	return tuifmt.FormatGiBFromMiB(*value)
}

func or(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
