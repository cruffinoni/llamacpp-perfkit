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
	if !strings.Contains(out.String(), "dev tui simulation") {
		t.Fatalf("unexpected output: %q", out.String())
	}
	_ = os.Getenv("LLAMACPP_PERFKIT_TUI_ONCE")
}

func TestDevTUIWithBarStyle(t *testing.T) {
	t.Setenv("LLAMACPP_PERFKIT_TUI_ONCE", "1")
	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"dev", "tui", "--bar-style", "block"})
	if err := cmd.ExecuteContext(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestDevTUIWithAllBarStyle(t *testing.T) {
	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"dev", "tui", "--bar-style", "all"})
	if err := cmd.ExecuteContext(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "block") {
		t.Error("all preview should contain block style")
	}
	if !strings.Contains(out.String(), "braille") {
		t.Error("all preview should contain braille style")
	}
}

func TestDevTUIInvalidBarStyle(t *testing.T) {
	t.Setenv("LLAMACPP_PERFKIT_TUI_ONCE", "1")
	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"dev", "tui", "--bar-style", "invalid"})
	err := cmd.ExecuteContext(context.Background())
	if err == nil {
		t.Fatal("expected error for invalid bar style")
	}
}
