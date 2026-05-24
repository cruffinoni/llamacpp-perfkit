package runner

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	"github.com/cruffinoni/llamacpp-perfkit/internal/config"
	"github.com/cruffinoni/llamacpp-perfkit/internal/domain"
	"github.com/cruffinoni/llamacpp-perfkit/internal/llamacpp"
)

type serverExecution struct {
	ID          string
	RawPath     string
	MonitorPath string
	CmdArgs     []string
	BaseURL     string
}

type serverProcess struct {
	cmd             *exec.Cmd
	cancel          context.CancelFunc
	logFile         *os.File
	shutdownTimeout int
}

func (r *Runner) prepareServer(group []domain.PlannedRun, groupIndex int) (serverExecution, error) {
	job := group[0].Job
	host := r.Config.Llama.Server.Host
	port, err := llamacpp.FreeTCPPort(host)
	if err != nil {
		return serverExecution{}, fmt.Errorf("find free TCP port on %s: %w", host, err)
	}
	serverRunID := fmt.Sprintf("%d-server-%04d", time.Now().Unix(), groupIndex)
	rawPath := filepath.Join(r.RawDir, serverRunID+".log")
	monitorPath := filepath.Join(r.MonitoringDir, serverRunID+".jsonl")
	cmdArgs := llamacpp.BuildServerCommand(r.Config, r.Features, job, port, rawPath)
	return serverExecution{
		ID:          serverRunID,
		RawPath:     rawPath,
		MonitorPath: monitorPath,
		CmdArgs:     cmdArgs,
		BaseURL:     fmt.Sprintf("http://%s:%d", host, port),
	}, nil
}

func startServer(ctx context.Context, cfg config.Config, client llamacpp.Client, server serverExecution) (*serverProcess, error) {
	logFile, err := os.OpenFile(server.RawPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open server log %s: %w", server.RawPath, err)
	}
	serverCtx, cancelServer := context.WithCancel(ctx)
	cmd := exec.CommandContext(serverCtx, server.CmdArgs[0], server.CmdArgs[1:]...)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	if runtime.GOOS != "windows" {
		cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	}
	if err := cmd.Start(); err != nil {
		cancelServer()
		_ = logFile.Close()
		return nil, fmt.Errorf("start llama-server %s: %w", llamacpp.CommandToShell(server.CmdArgs), err)
	}
	process := &serverProcess{
		cmd:             cmd,
		cancel:          cancelServer,
		logFile:         logFile,
		shutdownTimeout: cfg.Llama.Server.ShutdownTimeoutSeconds,
	}
	startupTimeout := time.Duration(cfg.Llama.Server.StartupTimeoutSeconds) * time.Second
	if err := client.WaitHealthy(ctx, server.BaseURL, startupTimeout); err != nil {
		process.Terminate()
		return nil, fmt.Errorf("wait for llama-server health at %s: %w", server.BaseURL, err)
	}
	return process, nil
}

func (p *serverProcess) Terminate() {
	if p == nil {
		return
	}
	if p.cancel != nil {
		p.cancel()
	}
	terminateProcess(p.cmd, p.shutdownTimeout)
	if p.logFile != nil {
		_ = p.logFile.Close()
	}
}

func terminateProcess(cmd *exec.Cmd, shutdownTimeout int) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	done := make(chan struct{})
	go func() {
		_ = cmd.Wait()
		close(done)
	}()
	if runtime.GOOS != "windows" {
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
	} else {
		_ = cmd.Process.Signal(os.Interrupt)
	}
	select {
	case <-done:
	case <-time.After(time.Duration(shutdownTimeout) * time.Second):
		if runtime.GOOS != "windows" {
			_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		} else {
			_ = cmd.Process.Kill()
		}
		<-done
	}
}
