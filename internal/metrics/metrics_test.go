package metrics

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
)

func TestLlamaCollectorSampleOnce(t *testing.T) {
	tests := map[string]struct {
		handler http.HandlerFunc
		wantOK  bool
		verify  func(*testing.T, domain.LlamaCppMetricSample)
	}{
		"valid slots response extracts tokens": {
			handler: func(w http.ResponseWriter, r *http.Request) {
				if strings.Contains(r.URL.Path, "/slots") {
					_, _ = w.Write([]byte(`[{"state":"processing","n_prompt_tokens":100,"n_decoded":50}]`))
					return
				}
				if strings.Contains(r.URL.Path, "/metrics") {
					w.WriteHeader(http.StatusOK)
					return
				}
				w.WriteHeader(http.StatusNotFound)
			},
			wantOK: true,
			verify: func(t *testing.T, s domain.LlamaCppMetricSample) {
				require.NotNil(t, s.SlotsProcessing)
				assert.Equal(t, 1, *s.SlotsProcessing)
				require.NotNil(t, s.PromptTokens)
				assert.InDelta(t, 100.0, *s.PromptTokens, 0.0001)
				require.NotNil(t, s.GeneratedTokens)
				assert.InDelta(t, 50.0, *s.GeneratedTokens, 0.0001)
			},
		},
		"prometheus metrics extracts tok s values": {
			handler: func(w http.ResponseWriter, r *http.Request) {
				if strings.Contains(r.URL.Path, "/slots") {
					w.WriteHeader(http.StatusServiceUnavailable)
					return
				}
				if strings.Contains(r.URL.Path, "/metrics") {
					_, _ = w.Write([]byte("llama_prompt_tokens_per_second 812.0\nllama_generation_tokens_per_second 78.0\n"))
					return
				}
				w.WriteHeader(http.StatusNotFound)
			},
			wantOK: true,
			verify: func(t *testing.T, s domain.LlamaCppMetricSample) {
				require.NotNil(t, s.PromptEvalTokS)
				assert.InDelta(t, 812.0, *s.PromptEvalTokS, 0.0001)
				require.NotNil(t, s.GenerationTokS)
				assert.InDelta(t, 78.0, *s.GenerationTokS, 0.0001)
			},
		},
		"all endpoints fail returns false": {
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
			},
			wantOK: false,
		},
		"malformed slots response returns false": {
			handler: func(w http.ResponseWriter, r *http.Request) {
				if strings.Contains(r.URL.Path, "/slots") {
					_, _ = w.Write([]byte(`not-json`))
					return
				}
				w.WriteHeader(http.StatusNotFound)
			},
			wantOK: false,
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(tc.handler)
			defer server.Close()
			collector := LlamaCollector{BaseURL: server.URL, Interval: time.Second}
			sample, ok := collector.SampleOnce(context.Background())
			assert.Equal(t, tc.wantOK, ok)
			if tc.verify != nil {
				tc.verify(t, sample)
			}
		})
	}

	t.Run("context cancelled before request returns false", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			time.Sleep(2 * time.Second)
		}))
		defer server.Close()
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		collector := LlamaCollector{BaseURL: server.URL, Interval: time.Second}
		_, ok := collector.SampleOnce(ctx)
		assert.False(t, ok)
	})
}

func TestSystemCollectorRun(t *testing.T) {
	dir := t.TempDir()
	collector := SystemCollector{RunDir: dir, Interval: 10 * time.Millisecond}
	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		errCh <- collector.Run(ctx)
	}()

	time.Sleep(30 * time.Millisecond)
	cancel()

	select {
	case err := <-errCh:
		assert.NoError(t, err)
	case <-time.After(time.Second):
		t.Fatal("SystemCollector.Run did not exit after cancellation")
	}
}

func TestApplySlots(t *testing.T) {
	tests := map[string]struct {
		input  any
		wantOK bool
		verify func(*testing.T, domain.LlamaCppMetricSample)
	}{
		"raw array with processing slot": {
			input:  []any{map[string]any{"state": "processing", "n_prompt_tokens": float64(200), "n_decoded": float64(80)}},
			wantOK: true,
			verify: func(t *testing.T, s domain.LlamaCppMetricSample) {
				assert.Equal(t, 1, *s.SlotsProcessing)
				assert.Equal(t, 0, *s.SlotsIdle)
			},
		},
		"object with slots key has idle and processing": {
			input: map[string]any{
				"slots": []any{
					map[string]any{"state": "idle", "prompt_tokens": float64(10), "generated_tokens": float64(5)},
					map[string]any{"state": "processing", "prompt_tokens": float64(20), "generated_tokens": float64(15)},
				},
			},
			wantOK: true,
			verify: func(t *testing.T, s domain.LlamaCppMetricSample) {
				assert.Equal(t, 1, *s.SlotsIdle)
				assert.Equal(t, 1, *s.SlotsProcessing)
			},
		},
		"empty array returns false": {
			input:  []any{},
			wantOK: false,
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			var sample domain.LlamaCppMetricSample
			ok := applySlots(&sample, tc.input)
			assert.Equal(t, tc.wantOK, ok)
			if tc.verify != nil {
				tc.verify(t, sample)
			}
		})
	}
}

func TestApplyPrometheus(t *testing.T) {
	tests := map[string]struct {
		input  string
		wantOK bool
		verify func(*testing.T, domain.LlamaCppMetricSample)
	}{
		"valid prometheus output extracts metrics": {
			input: `
# HELP llama_prompt_tokens_per_second ...
llama_prompt_tokens_per_second 812.0
llama_generation_tokens_per_second 78.0
llama_prompt_tokens 500
llama_generation_tokens 200
`,
			wantOK: true,
			verify: func(t *testing.T, s domain.LlamaCppMetricSample) {
				assert.InDelta(t, 812.0, *s.PromptEvalTokS, 0.0001)
				assert.InDelta(t, 78.0, *s.GenerationTokS, 0.0001)
			},
		},
		"empty input returns false": {
			input:  "",
			wantOK: false,
		},
		"only comments returns false": {
			input:  "# just a comment",
			wantOK: false,
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			var sample domain.LlamaCppMetricSample
			ok := applyPrometheus(&sample, tc.input)
			assert.Equal(t, tc.wantOK, ok)
			if tc.verify != nil {
				tc.verify(t, sample)
			}
		})
	}
}

func TestNumeric(t *testing.T) {
	tests := map[string]struct {
		input any
		want  *float64
	}{
		"float64":            {input: float64(3.14), want: new(3.14)},
		"int":                {input: 42, want: new(42.0)},
		"int64":              {input: int64(99), want: new(99.0)},
		"string returns nil": {input: "nope", want: nil},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			v := numeric(tc.input)
			if tc.want == nil {
				assert.Nil(t, v)
			} else {
				require.NotNil(t, v)
				assert.InDelta(t, *tc.want, *v, 0.0001)
			}
		})
	}
}
