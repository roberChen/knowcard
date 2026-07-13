package knowcard

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
)

// NewLogger creates a project-scoped slog logger that writes JSON to the
// given file path. The file is opened in append mode so logs accumulate
// across sessions. The caller is responsible for closing the returned file.
func NewLogger(logPath string, level slog.Level) (*slog.Logger, *os.File, error) {
	if err := os.MkdirAll(filepath.Dir(logPath), 0755); err != nil {
		return nil, nil, fmt.Errorf("creating log directory: %w", err)
	}

	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, nil, fmt.Errorf("opening log file %s: %w", logPath, err)
	}

	handler := slog.NewJSONHandler(f, &slog.HandlerOptions{
		Level: level,
	})
	logger := slog.New(handler)
	return logger, f, nil
}

// parseLogLevel converts a string config value to slog.Level.
// Empty string defaults to LevelInfo.
func parseLogLevel(s string) slog.Level {
	switch s {
	case "debug":
		return slog.LevelDebug
	case "warn", "warning":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}
