package main

import (
	"flag"
	"fmt"
	"os"

	kc "github.com/robert/knowcard"
	"github.com/robert/knowcard/mcp"
)

func cmdServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	fs.Parse(args)

	cfg := loadConfig()
	s, err := kc.Open(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error opening store: %v\n", err)
		os.Exit(1)
	}
	defer s.Close()

	srv := mcp.NewServer(s)
	if err := srv.Serve(); err != nil {
		fmt.Fprintf(os.Stderr, "MCP server error: %v\n", err)
		os.Exit(1)
	}
}
