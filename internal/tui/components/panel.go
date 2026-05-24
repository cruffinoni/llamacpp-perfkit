package components

import "github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"

// Panel renders content inside a themed panel border.
func Panel(s theme.Styles, content string) string {
	return s.Panel.Render(content)
}
