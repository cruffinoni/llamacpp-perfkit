package llamacpp

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestClientJSONHandlesStatusBodyAndErrors(t *testing.T) {
	var gotContentType string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotContentType = r.Header.Get("Content-Type")
		w.WriteHeader(http.StatusTeapot)
		_, _ = w.Write([]byte(`{"error":"short and stout"}`))
	}))
	defer server.Close()

	resp, err := NewClient(server.Client()).JSON(context.Background(), http.MethodPost, server.URL, map[string]any{"prompt": "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusTeapot {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if resp.Body["error"] != "short and stout" || !strings.Contains(resp.Text, "short and stout") {
		t.Fatalf("unexpected response: %+v text=%q", resp.Body, resp.Text)
	}
	if gotContentType != "application/json" {
		t.Fatalf("content type = %q", gotContentType)
	}

	_, err = NewClient(server.Client()).JSON(context.Background(), http.MethodGet, "://bad-url", nil)
	if err == nil || !strings.Contains(err.Error(), "build GET request") {
		t.Fatalf("expected request build context, got %v", err)
	}
}

func TestWaitHealthySuccessTimeoutAndCancellation(t *testing.T) {
	healthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	}))
	defer healthy.Close()
	if err := NewClient(healthy.Client()).WaitHealthy(context.Background(), healthy.URL, time.Second); err != nil {
		t.Fatal(err)
	}

	unhealthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"status":"loading"}`))
	}))
	defer unhealthy.Close()
	err := NewClient(unhealthy.Client()).WaitHealthy(context.Background(), unhealthy.URL, 10*time.Millisecond)
	if err == nil || !strings.Contains(err.Error(), "server health timeout") {
		t.Fatalf("expected timeout, got %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err = NewClient(unhealthy.Client()).WaitHealthy(ctx, unhealthy.URL, time.Second)
	if err == nil || !strings.Contains(err.Error(), "context canceled") {
		t.Fatalf("expected cancellation, got %v", err)
	}
}
