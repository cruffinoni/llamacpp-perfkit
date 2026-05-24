// Package cli contains Cobra commands for llama-cpp-perfkit.
package cli

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/spf13/cobra"

	"github.com/cruffinoni/llamacpp-perfkit/internal/tui"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/app"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/components"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/sim"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
)

func runSimulation(ctx context.Context, loop bool, barStyle components.ProgressBarStyle) error {
	ctrl := make(sim.Controller, 8)
	tui.SetSimController(ctrl)
	defer tui.ClearSimController()

	s := sim.New(sim.MixedScenario(), loop)
	return app.Run(ctx, s.InitialState(), s.BenchmarkFunc(ctrl), barStyle)
}

type progressPreviewOptions struct {
	Styles   []components.ProgressBarStyle
	Step     int
	Interval time.Duration
	MaxSteps int
}

type progressPreviewTickMsg struct{}

type progressPreviewModel struct {
	styles    theme.Styles
	opts      progressPreviewOptions
	completed int
}

func progressPreviewTick(interval time.Duration) tea.Cmd {
	return tea.Tick(interval, func(time.Time) tea.Msg {
		return progressPreviewTickMsg{}
	})
}

func newProgressPreviewModel(opts progressPreviewOptions) progressPreviewModel {
	return progressPreviewModel{
		styles: theme.NewStyles(theme.SolarizedDark),
		opts:   opts,
	}
}

func validateProgressPreviewOptions(opts progressPreviewOptions) error {
	if len(opts.Styles) == 0 {
		return fmt.Errorf("at least one progress bar style is required")
	}
	if opts.Step <= 0 {
		return fmt.Errorf("step must be greater than 0")
	}
	if opts.Interval <= 0 {
		return fmt.Errorf("interval must be greater than 0")
	}
	if opts.MaxSteps <= 0 {
		return fmt.Errorf("max-steps must be greater than 0")
	}
	return nil
}

// Init starts the first progress preview tick.
func (m progressPreviewModel) Init() tea.Cmd {
	if m.completed >= m.opts.MaxSteps {
		return tea.Quit
	}
	return progressPreviewTick(m.opts.Interval)
}

// Update advances the preview progress and clamps it at the configured max.
func (m progressPreviewModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch t := msg.(type) {
	case tea.KeyPressMsg:
		if t.String() == "ctrl+c" {
			return m, tea.Quit
		}
	case progressPreviewTickMsg:
		m.completed += m.opts.Step
		if m.completed >= m.opts.MaxSteps {
			m.completed = m.opts.MaxSteps
			return m, tea.Quit
		}
		return m, progressPreviewTick(m.opts.Interval)
	}
	return m, nil
}

// View renders all selected progress bar styles for the current preview step.
func (m progressPreviewModel) View() tea.View {
	var b strings.Builder
	b.WriteString("Progress style preview\n")
	for _, style := range m.opts.Styles {
		label := fmt.Sprintf("%-10s", style.String())
		bar := components.ProgressBar(m.styles, style, m.completed, m.opts.MaxSteps, 30)
		fmt.Fprintf(&b, "%s %s %d/%d\n", label, bar, m.completed, m.opts.MaxSteps)
	}
	b.WriteByte('\n')
	return tea.NewView(b.String())
}

// renderProgressPreview renders an in-place animated progress bar preview.
func renderProgressPreview(w io.Writer, opts progressPreviewOptions) error {
	if err := validateProgressPreviewOptions(opts); err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	model := newProgressPreviewModel(opts)
	program := tea.NewProgram(
		model,
		tea.WithContext(ctx),
		tea.WithInput(nil),
		tea.WithOutput(w),
		tea.WithWindowSize(80, 24),
		tea.WithoutSignals(),
	)
	_, err := program.Run()
	if ctx.Err() != nil && err == tea.ErrProgramKilled {
		return nil
	}
	return err
}

func devCommand() *cobra.Command {
	var (
		barStyle         string
		loop             bool
		progressStep     int
		progressInterval time.Duration
		progressMaxSteps int
	)
	cmd := &cobra.Command{Use: "dev", Short: "Development helpers."}
	tuiCmd := &cobra.Command{
		Use:   "tui",
		Short: "Run an animated fake benchmark simulation in the TUI.",
		RunE: func(cmd *cobra.Command, _ []string) error {
			parsed, err := components.ParseProgressBarStyle(barStyle)
			if err != nil {
				return err
			}

			if os.Getenv("LLAMACPP_PERFKIT_TUI_ONCE") == "1" {
				fmt.Fprintln(cmd.OutOrStdout(), "dev tui simulation")
				return nil
			}
			return runSimulation(cmd.Context(), loop, parsed)
		},
	}
	tuiCmd.Flags().StringVar(&barStyle, "style", "dot",
		"Progress bar style: block, line, dot, segmented or braille")
	tuiCmd.Flags().BoolVar(&loop, "loop", false, "Restart simulation when complete.")

	progressBar := &cobra.Command{
		Use:   "progress-bar",
		Short: "Preview animated progress bar styles.",
		RunE: func(cmd *cobra.Command, _ []string) error {
			opts := progressPreviewOptions{
				Step:     progressStep,
				Interval: progressInterval,
				MaxSteps: progressMaxSteps,
			}
			if barStyle == "all" {
				opts.Styles = []components.ProgressBarStyle{
					components.ProgressBarStyleBlock,
					components.ProgressBarStyleLine,
					components.ProgressBarStyleDot,
					components.ProgressBarStyleSegmented,
					components.ProgressBarStyleBraille,
				}
				return renderProgressPreview(cmd.OutOrStdout(), opts)
			}
			progressBarStyle, err := components.ParseProgressBarStyle(barStyle)
			if err != nil {
				return err
			}
			opts.Styles = []components.ProgressBarStyle{progressBarStyle}
			return renderProgressPreview(cmd.OutOrStdout(), opts)
		},
	}
	progressBar.Flags().StringVar(&barStyle, "style", "dot",
		"Preview the progress bar style: block, line, dot, segmented, braille, or all)")
	progressBar.Flags().IntVar(&progressStep, "step", 1, "Progress amount added on each tick.")
	progressBar.Flags().DurationVar(&progressInterval, "interval", 300*time.Millisecond,
		"Delay between progress preview ticks.")
	progressBar.Flags().IntVar(&progressMaxSteps, "max-steps", 100, "Maximum progress count to preview.")
	cmd.AddCommand(tuiCmd)
	cmd.AddCommand(progressBar)
	return cmd
}
