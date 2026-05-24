package cli

import (
	"context"
	"fmt"
	"os"

	"github.com/cruffinoni/llamacpp-perfkit/internal/config"
	"github.com/cruffinoni/llamacpp-perfkit/internal/report"
	"github.com/cruffinoni/llamacpp-perfkit/internal/runner"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui"
	"github.com/spf13/cobra"
)

func NewRootCommand() *cobra.Command {
	root := &cobra.Command{
		Use:           "llama-cpp-perfkit",
		Short:         "Benchmark llama.cpp server configurations.",
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	root.AddCommand(runCommand(), reportCommand(), devCommand())
	return root
}

func runCommand() *cobra.Command {
	var opts runner.Options
	cmd := &cobra.Command{
		Use:   "run <config>",
		Short: "Run a llama.cpp benchmark matrix.",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load(args[0])
			if err != nil {
				return err
			}
			r, err := runner.New(cmd.Context(), cfg, opts)
			if err != nil {
				return err
			}
			return r.Run(cmd.Context())
		},
	}
	cmd.Flags().StringVar(&opts.Mode, "mode", "", "Override budget mode: smoke, quick, focused, full.")
	var maxRuns int
	cmd.Flags().IntVar(&maxRuns, "max-runs", -1, "Run at most N new benchmark requests.")
	cmd.Flags().BoolVar(&opts.RetryFailed, "retry-failed", false, "Rerun failed, OOM, timeout, or unsupported runs.")
	cmd.Flags().BoolVarP(&opts.Force, "force", "f", false, "Force rerun of all selected configs.")
	cmd.Flags().BoolVar(&opts.DryRun, "dry-run", false, "Print planned server commands without launching models.")
	cmd.PreRun = func(*cobra.Command, []string) {
		if maxRuns >= 0 {
			opts.MaxRuns = &maxRuns
		}
	}
	return cmd
}

func reportCommand() *cobra.Command {
	cmd := &cobra.Command{Use: "report", Short: "Inspect benchmark results."}
	cmd.AddCommand(reportSummaryCommand(), reportByProfileCommand(), reportCompareCommand())
	return cmd
}

func reportSummaryCommand() *cobra.Command {
	var details bool
	var sortKey string
	var limit int
	cmd := &cobra.Command{
		Use:   "summary <runs-path>",
		Short: "Show per-server-config benchmark summaries.",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			rows, err := report.Load(args[0])
			if err != nil {
				return err
			}
			report.PrintSummary(cmd.OutOrStdout(), rows, report.SummaryOptions{Details: details, Sort: sortKey, Limit: limit})
			return nil
		},
	}
	cmd.Flags().BoolVar(&details, "details", false, "Show expanded server config columns.")
	cmd.Flags().StringVar(&sortKey, "sort", "balanced", "Sort key.")
	cmd.Flags().IntVar(&limit, "limit", 20, "Maximum rows to print.")
	return cmd
}

func reportByProfileCommand() *cobra.Command {
	var details bool
	var limit int
	cmd := &cobra.Command{
		Use:   "by-profile <runs-path>",
		Short: "Show observations split by prompt profile.",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			rows, err := report.Load(args[0])
			if err != nil {
				return err
			}
			report.PrintByProfile(cmd.OutOrStdout(), rows, report.SummaryOptions{Details: details, Limit: limit})
			return nil
		},
	}
	cmd.Flags().BoolVar(&details, "details", false, "Show expanded server config columns.")
	cmd.Flags().IntVar(&limit, "limit", 20, "Maximum rows to print.")
	return cmd
}

func reportCompareCommand() *cobra.Command {
	var baseline string
	var details bool
	var limit int
	cmd := &cobra.Command{
		Use:   "compare --baseline <baseline-run-or-runs-path> <candidate-run-or-runs-path>",
		Short: "Compare candidate run configs against a baseline.",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			baseRows, err := report.Load(baseline)
			if err != nil {
				return err
			}
			candidateRows, err := report.Load(args[0])
			if err != nil {
				return err
			}
			return report.PrintCompare(cmd.OutOrStdout(), baseRows, candidateRows, report.SummaryOptions{Details: details, Limit: limit})
		},
	}
	cmd.Flags().StringVar(&baseline, "baseline", "", "Baseline run directory or run root.")
	_ = cmd.MarkFlagRequired("baseline")
	cmd.Flags().BoolVar(&details, "details", false, "Show expanded server config columns.")
	cmd.Flags().IntVar(&limit, "limit", 20, "Maximum rows to print.")
	return cmd
}

func devCommand() *cobra.Command {
	cmd := &cobra.Command{Use: "dev", Short: "Development helpers."}
	cmd.AddCommand(&cobra.Command{
		Use:   "tui",
		Short: "Render a static fake benchmark TUI.",
		RunE: func(cmd *cobra.Command, _ []string) error {
			if os.Getenv("LLAMACPP_PERFKIT_TUI_ONCE") == "1" {
				state := tui.StaticState()
				fmt.Fprintln(cmd.OutOrStdout(), state.StatusMessage)
				return nil
			}
			return tui.Run(cmd.Context(), tui.StaticState(), func(ctx context.Context, updates chan<- tui.StateUpdate) error {
				<-ctx.Done()
				return nil
			})
		},
	})
	return cmd
}
