package components

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
)

// Alignment specifies text alignment within a table column.
type Alignment int

const (
	// AlignLeft aligns text to the left.
	AlignLeft Alignment = iota
	// AlignRight aligns text to the right.
	AlignRight
)

// Column describes a single table column header and its layout properties.
type Column struct {
	Header string
	Width  int
	Align  Alignment
}

// Cell holds the text and optional style for a single table cell.
type Cell struct {
	Text  string
	Style lipgloss.Style
}

type tableRow struct {
	cells []Cell
	isSep bool
}

// Table renders data in a columnar layout with optional separators.
type Table struct {
	cols []Column
	s    theme.Styles
	rows []tableRow
}

func isEmptyStyle(s lipgloss.Style) bool {
	return s.GetForeground() == lipgloss.NoColor{} && !s.GetBold() && !s.GetItalic()
}

func padText(s string, width int, align Alignment) string {
	if len(s) >= width {
		return s[:width]
	}
	pad := strings.Repeat(" ", width-len(s))
	if align == AlignRight {
		return pad + s
	}
	return s + pad
}

func (t *Table) cellAt(row tableRow, i int) Cell {
	if i < len(row.cells) {
		return row.cells[i]
	}
	return Cell{}
}

func (t *Table) renderRow(row tableRow) string {
	lb := NewLineBuilder(t.s.PanelBg, t.s.TextFg)
	for i, col := range t.cols {
		if i > 0 {
			lb.Raw("  ")
		}
		cell := t.cellAt(row, i)
		text := padText(cell.Text, col.Width, col.Align)
		if isEmptyStyle(cell.Style) {
			lb.Raw(text)
		} else {
			lb.Styled(cell.Style, text)
		}
	}
	return lb.Render()
}

func (t *Table) renderSeparator() string {
	lb := NewLineBuilder(t.s.PanelBg, t.s.TextFg)
	totalWidth := 0
	for i, col := range t.cols {
		if i > 0 {
			totalWidth += 2
		}
		totalWidth += col.Width
	}
	lb.Styled(t.s.Muted, strings.Repeat("-", totalWidth))
	return lb.Render()
}

func (t *Table) renderHeader() string {
	lb := NewLineBuilder(t.s.PanelBg, t.s.TextFg)
	headerStyle := t.s.TextBold.Copy().Bold(true)
	for i, col := range t.cols {
		if i > 0 {
			lb.Raw("  ")
		}
		lb.Styled(headerStyle, padText(col.Header, col.Width, col.Align))
	}
	return lb.Render()
}

// NewTable creates a table with the given columns and theme styles.
func NewTable(s theme.Styles, cols ...Column) *Table {
	return &Table{cols: cols, s: s}
}

// AddRow appends a row of cells to the table and returns the table for chaining.
func (t *Table) AddRow(cells ...Cell) *Table {
	t.rows = append(t.rows, tableRow{cells: cells})
	return t
}

// AddSeparator appends a visual separator row and returns the table for chaining.
func (t *Table) AddSeparator() *Table {
	t.rows = append(t.rows, tableRow{isSep: true})
	return t
}

// Render returns the fully rendered table string including header, rows,
// separators, and the outer panel border.
func (t *Table) Render() string {
	var lines []string
	lines = append(lines, t.renderHeader())
	for _, row := range t.rows {
		if row.isSep {
			lines = append(lines, t.renderSeparator())
		} else {
			lines = append(lines, t.renderRow(row))
		}
	}
	return t.s.Panel.Render(strings.Join(lines, "\n"))
}
