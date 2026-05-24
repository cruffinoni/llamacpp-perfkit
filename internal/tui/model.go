package tui

import (
	"context"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/viewmodel"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/views"
)

type doneMsg struct {
	err error
}

type tickMsg struct{}

const (
	keyCtrlC = "ctrl+c"
	keyQ     = "q"
	keyEsc   = "esc"
)

type model struct {
	state   viewmodel.BenchmarkTUIState
	styles  theme.Styles
	updates <-chan viewmodel.StateUpdate
	cancel  context.CancelFunc
	done    bool
	err     error
	width   int
	height  int
}

// visibleWidth returns the terminal column width of s, stripping ANSI escape
// sequences and counting Unicode runes (each rune = 1 column for our purposes;
// box-drawing chars are single-column in modern terminals).
func visibleWidth(s string) int {
	w := 0
	esc := false
	for _, r := range s {
		if esc {
			if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') {
				esc = false
			}
			continue
		}
		if r == '\x1b' {
			esc = true
			continue
		}
		w++
	}
	return w
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
) *tea.Program {
	return tea.NewProgram(model{state: state, styles: theme.NewStyles(theme.SolarizedDark), updates: updates, cancel: cancel}, tea.WithContext(ctx), tea.WithAltScreen())
}

// Init initialises the bubble tea model with update-waiting and tick commands.
func (m model) Init() tea.Cmd {
	return tea.Batch(waitUpdate(m.updates), tick())
}

// Update handles messages for the bubble tea model and returns an updated model.
func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
	case tea.KeyMsg:
		if msg.String() == keyCtrlC || msg.String() == keyQ || msg.String() == keyEsc {
			return m, tea.Quit
		}
	case viewmodel.StateUpdate:
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

// View renders the current TUI state as a string.
func (m model) View() string {
	content := views.Layout(m.state, m.styles, 0)
	if m.width == 0 {
		return content
	}
	lines := strings.Split(content, "\n")
	padStyle := m.styles.Base
	bgLine := padStyle.Width(m.width).Render("")
	for i, line := range lines {
		vis := visibleWidth(line)
		if vis < m.width {
			lines[i] = line + padStyle.Render(strings.Repeat(" ", m.width-vis))
		}
	}
	for i := len(lines); i < m.height; i++ {
		lines = append(lines, bgLine)
	}
	return strings.Join(lines, "\n")
}
