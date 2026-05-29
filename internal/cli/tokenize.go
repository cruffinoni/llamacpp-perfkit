// Package cli provides command-line interface commands.
package cli

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/cruffinoni/llamacpp-perfkit/internal/tokenize"
)

func tokenizeCommand() *cobra.Command {
	var model string
	cmd := &cobra.Command{
		Use:   "tokenize <file>",
		Short: "Count tokens in a file.",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if model == "" {
				return fmt.Errorf("required flag \"model\" is missing")
			}
			t, err := tokenize.NewGPTTokenizer(model)
			if err != nil {
				return err
			}
			count, err := tokenize.CountTokensFromFile(args[0], t)
			if err != nil {
				return err
			}
			_, _ = fmt.Fprintf(os.Stdout, "File %q has %d tokens.\n", args[0], count)
			return nil
		},
	}
	cmd.Flags().StringVar(&model, "model", "", "Tokenizer model name (e.g. gpt2).")
	_ = cmd.MarkFlagRequired("model")
	return cmd
}
