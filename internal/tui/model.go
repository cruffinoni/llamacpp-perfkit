package tui

import (
	"context"
	"sync"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/components"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/sim"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/viewmodel"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/views"
)

type doneMsg struct {
	err error
}

type tickMsg struct{}

const (
	keyCtrlC  = "ctrl+c"
	keyQ      = "q"
	keyEsc    = "esc"
	keySpace  = "space"
	keyR      = "r"
	keyRUpper = "R"
)

var (
	activeSimMu   sync.Mutex
	activeSimCtrl sim.Controller
)

// SetSimController registers a controller for the running simulation. The TUI
// model sends pause/reset commands on this channel when the user presses space
// or r. Call ClearSimController when the simulation ends.
func SetSimController(ctrl sim.Controller) {
	activeSimMu.Lock()
	activeSimCtrl = ctrl
	activeSimMu.Unlock()
}

// ClearSimController removes the registered simulation controller.
func ClearSimController() {
	activeSimMu.Lock()
	activeSimCtrl = nil
	activeSimMu.Unlock()
}

func getSimController() sim.Controller {
	activeSimMu.Lock()
	defer activeSimMu.Unlock()
	return activeSimCtrl
}

type model struct {
	state    viewmodel.BenchmarkTUIState
	styles   theme.Styles
	updates  <-chan viewmodel.StateUpdate
	cancel   context.CancelFunc
	done     bool
	err      error
	barStyle components.ProgressBarStyle
}

func tick() tea.Cmd {
	return tea.Tick(time.Second, func(time.Time) tea.Msg { return tickMsg{} })
}

func waitUpdate(updates <-chan viewmodel.StateUpdate) tea.Cmd {
	return func() tea.Msg {
		update, ok := <-updates
		if !ok {
			return doneMsg{}
		}
		return update
	}
}

// NewProgram creates a new Bubble Tea program configured for the benchmark TUI.
func NewProgram(
	ctx context.Context,
	state viewmodel.BenchmarkTUIState,
	updates <-chan viewmodel.StateUpdate,
	cancel context.CancelFunc,
	barStyle components.ProgressBarStyle,
) *tea.Program {
	return tea.NewProgram(model{
		state:    state,
		styles:   theme.NewStyles(theme.SolarizedDark),
		updates:  updates,
		cancel:   cancel,
		barStyle: barStyle,
	}, tea.WithContext(ctx))
}

// Init initialises the bubble tea model with update-waiting and tick commands.
func (m model) Init() tea.Cmd {
	return tea.Batch(waitUpdate(m.updates), tick())
}

// Update handles messages for the bubble tea model and returns an updated model.
func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch t := msg.(type) {
	case tea.KeyPressMsg:
		switch t.String() {
		case keyCtrlC, keyQ, keyEsc:
			return m, tea.Quit
		case keySpace:
			if ctrl := getSimController(); ctrl != nil {
				select {
				case ctrl <- sim.TogglePause:
				default:
				}
			}
		case keyR, keyRUpper:
			if ctrl := getSimController(); ctrl != nil {
				select {
				case ctrl <- sim.Reset:
				default:
				}
			}
		}
	case viewmodel.StateUpdate:
		if t.Apply != nil {
			t.Apply(&m.state)
		}
		return m, waitUpdate(m.updates)
	case doneMsg:
		m.done = true
		m.err = t.err
		if t.err != nil {
			m.state.StatusMessage = t.err.Error()
		}
		return m, tea.Quit
	case tickMsg:
		return m, tick()
	}
	return m, nil
}

// View renders the current TUI state as a tea.View.
func (m model) View() tea.View {
	v := tea.NewView(views.Layout(m.state, m.styles, m.barStyle))
	v.AltScreen = true
	v.BackgroundColor = m.styles.Base.GetBackground()
	return v
}
