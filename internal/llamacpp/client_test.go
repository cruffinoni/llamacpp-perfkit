package llamacpp

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	require.NoError(t, err)
	assert.Equal(t, http.StatusTeapot, resp.StatusCode)
	assert.Equal(t, "short and stout", resp.Body["error"])
	assert.Contains(t, resp.Text, "short and stout")
	assert.Equal(t, "application/json", gotContentType)

	_, err = NewClient(server.Client()).JSON(context.Background(), http.MethodGet, "://bad-url", nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "build GET request")
}

func TestWaitHealthySuccessTimeoutAndCancellation(t *testing.T) {
	healthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	}))
	defer healthy.Close()

	err := NewClient(healthy.Client()).WaitHealthy(context.Background(), healthy.URL, time.Second)
	require.NoError(t, err)

	unhealthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"status":"loading"}`))
	}))
	defer unhealthy.Close()

	err = NewClient(unhealthy.Client()).WaitHealthy(context.Background(), unhealthy.URL, 10*time.Millisecond)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "server health timeout")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err = NewClient(unhealthy.Client()).WaitHealthy(ctx, unhealthy.URL, time.Second)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "context canceled")
}

func TestClientJSONInvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`not-json`))
	}))
	defer server.Close()

	resp, err := NewClient(server.Client()).JSON(context.Background(), http.MethodGet, server.URL, nil)
	require.NoError(t, err)
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, "not-json", resp.Text)
	assert.Empty(t, resp.Body)
}

func TestClientJSONServer500(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"internal"}`))
	}))
	defer server.Close()

	resp, err := NewClient(server.Client()).JSON(context.Background(), http.MethodGet, server.URL, nil)
	require.NoError(t, err)
	assert.Equal(t, http.StatusInternalServerError, resp.StatusCode)
	assert.Equal(t, "internal", resp.Body["error"])
}

func TestClientJSONContextCancelled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// Hang
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := NewClient(server.Client()).JSON(ctx, http.MethodGet, server.URL, nil)
	require.Error(t, err)
	assert.True(t, strings.Contains(err.Error(), "canceled") || strings.Contains(err.Error(), "connect"), "expected context canceled, got: %v", err)
}

func TestClientJSONEmptyPayload(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "", r.Header.Get("Content-Type"))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok": true}`))
	}))
	defer server.Close()

	resp, err := NewClient(server.Client()).JSON(context.Background(), http.MethodGet, server.URL, nil)
	require.NoError(t, err)
	assert.Equal(t, http.StatusOK, resp.StatusCode)
}

func TestClientJSONEmptyResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	resp, err := NewClient(server.Client()).JSON(context.Background(), http.MethodGet, server.URL, nil)
	require.NoError(t, err)
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Empty(t, resp.Text)
	assert.Empty(t, resp.Body)
}

func TestHTTPJSONConvenience(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"value": 42}`))
	}))
	defer server.Close()

	status, body, text, err := HTTPJSON(context.Background(), http.MethodGet, server.URL, nil)
	require.NoError(t, err)
	assert.Equal(t, http.StatusOK, status)
	assert.Equal(t, float64(42), body["value"])
	assert.Contains(t, text, "42")
}
