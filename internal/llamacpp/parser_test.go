package llamacpp

import "testing"

func TestParseChatCompletionPreservesPersistedMapShape(t *testing.T) {
	body := map[string]any{
		"choices": []any{map[string]any{
			"finish_reason": "stop",
			"message":       map[string]any{"content": "hello", "reasoning": "scratch"},
		}},
		"usage": map[string]any{
			"completion_tokens":     float64(8),
			"prompt_tokens":         float64(16),
			"prompt_tokens_details": map[string]any{"cached_tokens": float64(4)},
		},
		"timings": map[string]any{"prompt_per_second": float64(100), "predicted_per_second": float64(25)},
	}

	parsed, response := ParseChatCompletion(body, 2, "")
	if parsed["generation_tok_s"] != float64(25) {
		t.Fatalf("generation_tok_s = %#v", parsed["generation_tok_s"])
	}
	if parsed["tokens_cached"] != float64(4) || parsed["throughput_source"] != "server_timings" {
		t.Fatalf("parsed = %#v", parsed)
	}
	if response["content"] != "hello" || response["reasoning_content"] != "scratch" {
		t.Fatalf("response = %#v", response)
	}
}

func TestParseLlamaOutputExtractsTerminalMetrics(t *testing.T) {
	text := `
llama_print_timings: load time = 10.25 ms
llama_print_timings: prompt eval time = 20.00 ms / 16 tokens ( 800.00 tokens per second)
llama_print_timings: eval time = 40.00 ms / 8 runs ( 200.00 tokens per second)
llama_print_timings: total time = 70.00 ms
speculative decoded draft accept`

	parsed := ParseLlamaOutput(text)
	if parsed["tokens_prompt"] != float64(16) || parsed["tokens_generated"] != float64(8) {
		t.Fatalf("tokens = %#v", parsed)
	}
	if parsed["prompt_eval_tok_s"] != float64(800) || parsed["generation_tok_s"] != float64(200) {
		t.Fatalf("speeds = %#v", parsed)
	}
	if parsed["speculative_stats"] == nil {
		t.Fatalf("speculative_stats missing: %#v", parsed)
	}
}
