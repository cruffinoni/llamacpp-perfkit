package sim

// ControlAction identifies a simulation control command sent from the TUI.
type ControlAction int

const (
	TogglePause ControlAction = iota
	Reset
)

// Controller is a buffered channel for sending control commands to a running
// simulation. The TUI model sends on this channel; the simulation's benchmark
// goroutine receives.
type Controller chan ControlAction
