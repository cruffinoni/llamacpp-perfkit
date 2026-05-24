package components

import (
	"testing"

	"github.com/cruffinoni/llamacpp-perfkit/internal/tui/theme"
)

func TestTableRenderEmpty(t *testing.T) {
	s := theme.NewStyles(theme.SolarizedDark)
	tbl := NewTable(s, Column{Header: "name", Width: 10, Align: AlignLeft})
	rendered := tbl.Render()
	if rendered == "" {
		t.Fatal("Empty table should render header")
	}
	if !containsSub(rendered, "name") {
		t.Error("Empty table should contain header text")
	}
}

func TestTableRenderSingleRow(t *testing.T) {
	s := theme.NewStyles(theme.SolarizedDark)
	tbl := NewTable(s,
		Column{Header: "name", Width: 10, Align: AlignLeft},
		Column{Header: "score", Width: 6, Align: AlignRight},
	)
	tbl.AddSeparator()
	tbl.AddRow(Cell{Text: "code"}, Cell{Text: "42"})
	rendered := tbl.Render()
	if !containsSub(rendered, "name") || !containsSub(rendered, "score") {
		t.Error("Should contain column headers")
	}
	if !containsSub(rendered, "code") || !containsSub(rendered, "42") {
		t.Error("Should contain row data")
	}
}

func TestTableRenderStyledCells(t *testing.T) {
	s := theme.NewStyles(theme.SolarizedDark)
	tbl := NewTable(s,
		Column{Header: "name", Width: 8, Align: AlignLeft},
		Column{Header: "status", Width: 8, Align: AlignLeft},
	)
	tbl.AddSeparator()
	tbl.AddRow(
		Cell{Text: "code"},
		Cell{Text: "success", Style: s.Success},
	)
	rendered := tbl.Render()
	if !containsSub(rendered, "success") {
		t.Error("Should contain status value")
	}
}

func TestTableRenderFewerCells(t *testing.T) {
	s := theme.NewStyles(theme.SolarizedDark)
	tbl := NewTable(s,
		Column{Header: "a", Width: 5, Align: AlignLeft},
		Column{Header: "b", Width: 5, Align: AlignLeft},
		Column{Header: "c", Width: 5, Align: AlignLeft},
	)
	tbl.AddSeparator()
	tbl.AddRow(Cell{Text: "x"}) // Only 1 cell, 3 columns
	rendered := tbl.Render()
	if rendered == "" {
		t.Fatal("Should render with fewer cells than columns")
	}
}

func containsSub(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
