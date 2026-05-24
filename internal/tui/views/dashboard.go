package views

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/components"
	tuifmt "github.com/cruffinoni/llamacpp-perfkit/internal/tui/format"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/viewmodel"
)

const (
	// MinTerminalWidth is the minimum terminal width supported by the dashboard.
	MinTerminalWidth = 92
	// MinTerminalHeight is the minimum terminal height supported by the dashboard.
	MinTerminalHeight = 20
)

// TerminalSize describes the terminal dimensions available to the dashboard.
type TerminalSize struct {
	Width  int
	Height int
}

// LayoutOptions groups the rendering dependencies for the dashboard layout.
type LayoutOptions struct {
	Styles   theme.Styles
	BarStyle components.ProgressBarStyle
	Size     TerminalSize
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

func newLine(s theme.Styles) *components.LineBuilder {
	return components.NewLineBuilder(s.PanelBg, s.TextFg)
}

// BenchmarkHeader renders the top panel with run metadata.
func BenchmarkHeader(state viewmodel.BenchmarkTUIState, s theme.Styles) string {
	bi := state.BuildInfo

	line1 := newLine(s).
		Styled(s.Title, "llama-cpp-perfkit").
		Styled(s.Muted, "  run ").
		Styled(s.Accent, state.RunID).
		Render()

	line2 := newLine(s).
		Styled(s.Title, "llama.cpp: ").
		Styled(s.Info, or(bi.CommitShort, "unknown")).
		Styled(s.TextBold, "  ").
		Styled(s.Muted, or(bi.Branch, "unknown")).
		Styled(s.TextBold, "  ").
		Styled(s.Info, or(bi.Backend, "server")).
		Render()

	line3 := newLine(s).
		Styled(s.Label, "Model: ").
		Styled(s.Title, state.ModelName).
		Render()

	return components.Panel(s, strings.Join([]string{line1, line2, line3}, "\n"))
}

// ProgressBlock renders the progress panel with 3 metric bars.
func ProgressBlock(state viewmodel.BenchmarkTUIState, s theme.Styles, barStyle components.ProgressBarStyle) string {
	p := state.Progress

	summary := newLine(s).
		Rawf("Matrix: %d/%d servers   Jobs: %d/%d   Elapsed: %s   ETA: %s",
			p.ServersCompleted, p.ServersTotal,
			p.JobsCompleted, p.JobsTotal,
			tuifmt.FormatElapsed(state.ElapsedSeconds),
			tuifmt.FormatElapsed(state.ETASeconds),
		).
		Render()

	bar := func(label string, done, total int, suffix string) string {
		return newLine(s).
			Styled(s.Label, label).
			Raw(" ").
			Raw(components.ProgressBar(s, barStyle, done, total, 30)).
			Rawf(" %d/%d%s", done, max(total, 1), suffix).
			Render()
	}

	lines := []string{
		summary,
		bar("Servers", p.ServersCompleted, p.ServersTotal, ""),
		bar("Jobs   ", p.JobsCompleted, p.JobsTotal, ""),
		bar("Current", p.CurrentPrompt, p.CurrentPromptTotal, " prompts"),
	}
	return components.Panel(s, strings.Join(lines, "\n"))
}

// CurrentServerBlock renders the current server configuration panel.
func CurrentServerBlock(state viewmodel.BenchmarkTUIState, s theme.Styles) string {
	if state.CurrentServer == nil {
		line1 := newLine(s).Styled(s.Title, "Current server").Render()
		line2 := newLine(s).Styled(s.Muted, "(none)").Render()
		return components.Panel(s, line1+"\n"+line2)
	}

	cs := state.CurrentServer

	line1 := newLine(s).Styled(s.Title, "Current server").Render()

	line2 := newLine(s).
		Styled(s.Label, "id: ").
		Raw(cs.ID).
		Render()

	line3 := newLine(s).
		Raw("ctx=").
		Styled(s.Info, tuifmt.FormatContextSize(cs.ContextSize)).
		Raw("  kv=").
		Styled(s.Info, cs.KVType).
		Rawf("  moe=%d  spec=", cs.NCPUMOE).
		Styled(s.Warning, cs.SpecType).
		Rawf("  batch=%d  ubatch=%d", cs.BatchSize, cs.UBatchSize).
		Render()

	return components.Panel(s, strings.Join([]string{line1, line2, line3}, "\n"))
}

// PromptTable renders the prompts table panel with aligned columns.
func PromptTable(state viewmodel.BenchmarkTUIState, s theme.Styles) string {
	if len(state.PromptJobs) == 0 {
		return components.Panel(s, newLine(s).Styled(s.Muted, "No prompts.").Render())
	}

	t := components.NewTable(s,
		components.Column{Header: "profile", Width: 18, Align: components.AlignLeft},
		components.Column{Header: "status", Width: 9, Align: components.AlignLeft},
		components.Column{Header: "phase", Width: 10, Align: components.AlignLeft},
		components.Column{Header: "time", Width: 6, Align: components.AlignRight},
		components.Column{Header: "gen tok/s", Width: 10, Align: components.AlignRight},
		components.Column{Header: "prompt tok/s", Width: 12, Align: components.AlignRight},
		components.Column{Header: "min vram", Width: 10, Align: components.AlignRight},
	)
	t.AddSeparator()

	for _, job := range state.PromptJobs {
		t.AddRow(
			components.Cell{Text: job.Profile},
			components.Cell{Text: job.Status.String(), Style: theme.StatusStyle(s, job.Status)},
			components.Cell{Text: string(job.Phase), Style: theme.PhaseStyle(s, job.Phase)},
			components.Cell{Text: formatDuration(job.DurationSeconds)},
			components.Cell{Text: formatTokS(job.GenTokS, "gen")},
			components.Cell{Text: formatTokS(job.PromptTokS, "prompt")},
			components.Cell{Text: formatVRAM(job.MinVRAMMiB)},
		)
	}
	return t.Render()
}

func terminalSizeSupported(size TerminalSize) bool {
	return size.Width >= MinTerminalWidth && size.Height >= MinTerminalHeight
}

func unsupportedTerminalSize(size TerminalSize, s theme.Styles) string {
	message := fmt.Sprintf(
		"Terminal size unsupported.\nNeed at least %dx%d, got %dx%d.",
		MinTerminalWidth,
		MinTerminalHeight,
		size.Width,
		size.Height,
	)
	return s.Base.Render(message)
}

func centerComponent(width int, component string) string {
	return lipgloss.PlaceHorizontal(width, lipgloss.Center, component)
}

// Layout joins all panels vertically with left alignment.
func Layout(state viewmodel.BenchmarkTUIState, opts LayoutOptions) string {
	if !terminalSizeSupported(opts.Size) {
		return unsupportedTerminalSize(opts.Size, opts.Styles)
	}

	return opts.Styles.Base.Render(
		lipgloss.JoinVertical(lipgloss.Left,
			centerComponent(opts.Size.Width, BenchmarkHeader(state, opts.Styles)),
			ProgressBlock(state, opts.Styles, opts.BarStyle),
			CurrentServerBlock(state, opts.Styles),
			PromptTable(state, opts.Styles),
			opts.Styles.StatusLine.Render(state.StatusMessage),
		))
}
