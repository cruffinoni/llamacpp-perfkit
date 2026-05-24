package app

import (
	"context"
	"errors"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/viewmodel"
)

// Run launches the benchmark TUI, runs the given benchmark function, and
// blocks until the program exits or the benchmark completes.
func Run(
	ctx context.Context,
	initial viewmodel.BenchmarkTUIState,
	benchmark func(context.Context, chan<- viewmodel.StateUpdate) error,
) error {
	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	updates := make(chan viewmodel.StateUpdate, 128)
	errCh := make(chan error, 1)
	program := tui.NewProgram(runCtx, initial, updates, cancel)
	go func() {
		defer close(updates)
		err := benchmark(runCtx, updates)
		if err != nil {
			updates <- viewmodel.StateUpdate{Apply: func(s *viewmodel.BenchmarkTUIState) { s.StatusMessage = err.Error() }}
		}
		errCh <- err
	}()
	_, runErr := program.Run()
	cancel()
	benchErr := <-errCh
	if errors.Is(benchErr, context.Canceled) {
		benchErr = nil
	}
	if errors.Is(runErr, tea.ErrProgramKilled) {
		runErr = nil
	}
	if runErr != nil {
		return runErr
	}
	return benchErr
}
