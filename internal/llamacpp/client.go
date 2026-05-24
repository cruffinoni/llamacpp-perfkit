package llamacpp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client sends HTTP requests to a llama.cpp server and decodes JSON responses.
type Client struct {
	HTTP *http.Client
}

// NewClient creates a Client with the given HTTP client.
// If httpClient is nil, http.DefaultClient is used.
func NewClient(httpClient *http.Client) Client {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return Client{HTTP: httpClient}
}

// JSONResponse holds the decoded response from a JSON API call.
type JSONResponse struct {
	StatusCode int
	Body       map[string]any
	Text       string
}

// JSON sends an HTTP request and decodes the JSON response body.
func (c Client) JSON(ctx context.Context, method, url string, payload map[string]any) (JSONResponse, error) {
	var body io.Reader
	if payload != nil {
		data, err := json.Marshal(payload)
		if err != nil {
			return JSONResponse{}, fmt.Errorf("marshal %s request body: %w", method, err)
		}
		body = strings.NewReader(string(data))
	}
	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return JSONResponse{}, fmt.Errorf("build %s request %s: %w", method, url, err)
	}
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return JSONResponse{}, fmt.Errorf("%s %s: %w", method, url, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return JSONResponse{StatusCode: resp.StatusCode}, fmt.Errorf("read %s response body: %w", url, err)
	}
	text := string(data)
	decoded := map[string]any{}
	if len(data) > 0 {
		_ = json.Unmarshal(data, &decoded)
	}
	return JSONResponse{StatusCode: resp.StatusCode, Body: decoded, Text: text}, nil
}

// WaitHealthy polls the server health endpoint until it responds OK or the timeout expires.
func (c Client) WaitHealthy(ctx context.Context, baseURL string, startupTimeout time.Duration) error {
	if startupTimeout <= 0 {
		startupTimeout = time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, startupTimeout)
	defer cancel()

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	var last string
	for {
		reqCtx, reqCancel := context.WithTimeout(ctx, 5*time.Second)
		resp, err := c.JSON(reqCtx, http.MethodGet, strings.TrimRight(baseURL, "/")+"/health", nil)
		reqCancel()
		if err != nil {
			last = err.Error()
		} else {
			last = resp.Text
			if resp.StatusCode == http.StatusOK && fmt.Sprint(resp.Body["status"]) == "ok" {
				return nil
			}
		}

		select {
		case <-ctx.Done():
			if err := context.Cause(ctx); err != nil {
				if err == context.DeadlineExceeded {
					return fmt.Errorf("server health timeout after %s; last_response=%s", startupTimeout, last)
				}
				return err
			}
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

// HTTPJSON is a convenience function for making a JSON HTTP request.
func HTTPJSON(ctx context.Context, method, url string, payload map[string]any) (int, map[string]any, string, error) {
	resp, err := NewClient(nil).JSON(ctx, method, url, payload)
	return resp.StatusCode, resp.Body, resp.Text, err
}

// WaitHealthy is a convenience function that waits for a server to become healthy.
func WaitHealthy(ctx context.Context, baseURL string, startupTimeout time.Duration) error {
	return NewClient(nil).WaitHealthy(ctx, baseURL, startupTimeout)
}
