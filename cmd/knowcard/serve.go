package main

import (
	"flag"
	"fmt"
	"io/fs"
	"os"
	"strings"

	kc "github.com/robert/knowcard"
	"github.com/robert/knowcard/mcp"
)

func cmdServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	fs.Parse(args)

	cfg := loadGlobalConfig()
	srv := mcp.NewServer(cfg)
	if err := srv.Serve(); err != nil {
		fmt.Fprintf(os.Stderr, "MCP server error: %v\n", err)
		os.Exit(1)
	}
}

// loadGlobalConfig loads the global config (embed backend, API keys) without
// requiring a .knowcard/ root. Falls back to defaults if no config file exists.
func loadGlobalConfig() kc.Config {
	configPath := ""
	for i, a := range os.Args {
		if a == "--config" && i+1 < len(os.Args) {
			configPath = os.Args[i+1]
		}
	}

	if configPath != "" {
		c, err := kc.Load(configPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error loading config: %v\n", err)
			os.Exit(1)
		}
		return *c
	}

	c, err := kc.LoadGlobal()
	if err != nil {
		if isNotExist(err) {
			return kc.DefaultConfig()
		}
		fmt.Fprintf(os.Stderr, "error loading config: %v\n", err)
		os.Exit(1)
	}
	return *c
}

func isNotExist(err error) bool {
	if os.IsNotExist(err) {
		return true
	}
	var pe *fs.PathError
	if strings.Contains(err.Error(), "no such file") {
		return true
	}
	_ = pe
	return false
}
