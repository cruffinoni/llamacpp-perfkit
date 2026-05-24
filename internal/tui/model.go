package tui

import (
	"context"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/components"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
)

type doneMsg struct {
	err error
}

type tickMsg struct{}

type model struct {
	state   BenchmarkTUIState
	styles  theme.Styles
	updates <-chan StateUpdate
	cancel  context.CancelFunc
	done    bool
	err     error
}

func NewProgram(ctx context.Context, state BenchmarkTUIState, updates <-chan StateUpdate, cancel context.CancelFunc) *tea.Program {
	return tea.NewProgram(model{state: state, styles: theme.NewStyles(), updates: updates, cancel: cancel}, tea.WithContext(ctx), tea.WithAltScreen())
}

func (m model) Init() tea.Cmd {
	return tea.Batch(waitUpdate(m.updates), tick())
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		if msg.String() == "ctrl+c" || msg.String() == "q" {
			if m.cancel != nil {
				m.cancel()
			}
			return m, tea.Quit
		}
	case StateUpdate:
		if msg.Apply != nil {
			msg.Apply(&m.state)
		}
		return m, waitUpdate(m.updates)
	case doneMsg:
		m.done = true
		m.err = msg.err
		if msg.err != nil {
			m.state.StatusMessage = msg.err.Error()
		}
		return m, tea.Quit
	case tickMsg:
		return m, tick()
	}
	return m, nil
}

func (m model) View() string {
	return components.Layout(m.state, m.styles)
}

func waitUpdate(updates <-chan StateUpdate) tea.Cmd {
	return func() tea.Msg {
		update, ok := <-updates
		if !ok {
			return doneMsg{}
		}
		return update
	}
}

func tick() tea.Cmd {
	return tea.Tick(time.Second, func(time.Time) tea.Msg { return tickMsg{} })
}

func Run(ctx context.Context, initial BenchmarkTUIState, benchmark func(context.Context, chan<- StateUpdate) error) error {
	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	updates := make(chan StateUpdate, 128)
	errCh := make(chan error, 1)
	program := NewProgram(runCtx, initial, updates, cancel)
	go func() {
		err := benchmark(runCtx, updates)
		if err != nil {
			updates <- StateUpdate{Apply: func(s *BenchmarkTUIState) { s.StatusMessage = err.Error() }}
		}
		errCh <- err
		close(updates)
	}()
	_, runErr := program.Run()
	cancel()
	benchErr := <-errCh
	if runErr != nil {
		return runErr
	}
	return benchErr
}

func StaticState() BenchmarkTUIState {
	v1, v2, v3, v4, v5, v6 := 78.0, 63.2, 68.2, 65.3, 55.1, 79.3
	p1, p2, p3, p4, p5, p6 := 812.0, 691.0, 604.0, 755.0, 710.0, 902.0
	d1, d2, d3, d4, d5, d6 := 2.70, 3.21, 3.25, 3.35, 4.04, 1.91
	m1, m2, m3, m4, m5, m6 := 5520.0, 5417.0, 5448.0, 5489.0, 5499.0, 5509.0
	return BenchmarkTUIState{
		RunID:          "1779537110",
		BuildInfo:      BuildInfoView{CommitShort: "b64739ea", Branch: "master", Backend: "cuda"},
		ModelName:      "Qwen3.6-35B-A3B-MTP UD-Q4_K_M",
		Progress:       ProgressState{ServersCompleted: 14, ServersTotal: 96, JobsCompleted: 104, JobsTotal: 768, CurrentPrompt: 5, CurrentPromptTotal: 8},
		ElapsedSeconds: 522,
		ETASeconds:     2350,
		CurrentServer: &CurrentServerView{
			ID: "1779537195-server-0014", ContextSize: 8192, KVType: "q8_0", NCPUMOE: 18, SpecType: "draft-mtp", BatchSize: 512, UBatchSize: 512,
		},
		PromptJobs: []PromptJobView{
			{Profile: "code_python", Status: "success", Phase: "done", DurationSeconds: &d1, GenTokS: &v1, PromptTokS: &p1, MinVRAMMiB: &m1},
			{Profile: "code_cpp", Status: "success", Phase: "done", DurationSeconds: &d2, GenTokS: &v2, PromptTokS: &p2, MinVRAMMiB: &m2},
			{Profile: "long_code_review", Status: "success", Phase: "done", DurationSeconds: &d3, GenTokS: &v3, PromptTokS: &p3, MinVRAMMiB: &m3},
			{Profile: "long_prefill_8k", Status: "success", Phase: "done", DurationSeconds: &d4, GenTokS: &v4, PromptTokS: &p4, MinVRAMMiB: &m4},
			{Profile: "long_prefill_16k", Status: "success", Phase: "done", DurationSeconds: &d5, GenTokS: &v5, PromptTokS: &p5, MinVRAMMiB: &m5},
			{Profile: "long_prefill_32k", Status: "running", Phase: "generating", DurationSeconds: &d6, GenTokS: &v6, PromptTokS: &p6, MinVRAMMiB: &m6},
			{Profile: "long_prefill_48k", Status: "pending", Phase: "-"},
			{Profile: "long_prefill_60k", Status: "pending", Phase: "-"},
		},
		StatusMessage: "dev tui static benchmark state",
	}
}
