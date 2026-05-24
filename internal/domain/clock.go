package domain

import "time"

// Clock abstracts time operations for deterministic testing.
type Clock interface {
	Now() time.Time
	Since(t time.Time) time.Duration
}

// RealClock delegates to the standard time package.
type RealClock struct{}

// Now returns the current local time.
func (RealClock) Now() time.Time { return time.Now() }

// Since returns the duration since the given time.
func (RealClock) Since(t time.Time) time.Duration { return time.Since(t) }
