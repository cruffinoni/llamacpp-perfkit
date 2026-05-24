package cli

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"
)

func TestDevTUIOnceCommand(t *testing.T) {
	t.Setenv("LLAMACPP_PERFKIT_TUI_ONCE", "1")
	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"dev", "tui"})
	if err := cmd.ExecuteContext(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "dev tui static benchmark state") {
		t.Fatalf("unexpected output: %q", out.String())
	}
	_ = os.Getenv("LLAMACPP_PERFKIT_TUI_ONCE")
}
