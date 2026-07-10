package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	kc "github.com/robert/knowcard"
	"github.com/robert/knowcard/card"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	cmd := os.Args[1]
	args := os.Args[2:]

	switch cmd {
	case "init":
		cmdInit(args)
	case "recall":
		cmdRecall(args)
	case "show":
		cmdShow(args)
	case "add":
		cmdAdd(args)
	case "delete":
		cmdDelete(args)
	case "move":
		cmdMove(args)
	case "list":
		cmdList(args)
	case "rebuild":
		cmdRebuild(args)
	case "history":
		cmdHistory(args)
	case "serve":
		cmdServe(args)
	case "version":
		fmt.Println("knowcard v0.1.0")
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", cmd)
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Print(`knowcard - local agent memory system

Usage:
  knowcard <command> [options]

Commands:
  init                        Initialize .knowcard in current project directory
  recall <query>              Search cards (hybrid semantic + keyword)
  show <id> [<id>...]         Show full card content
  add                         Add a card (from file or flags)
  delete <id>                 Delete a card
  move <id> <new-path>        Move card to a new path
  list [--path PREFIX]        List all cards
  rebuild                     Rebuild indexes from .md files
  history <id>                Show revision history for a card
  serve                       Start MCP server (stdio)
  config                      Show current configuration

The knowledge base lives in .knowcard/ at the project root.
knowcard searches upward from CWD to find it, just like git finds .git/.
Global config (embed backend, API keys) lives in
~/.config/knowcard/config.yaml.

Options:
  --config <path>             Path to config file (overrides global config)
  --root <path>               Project root to search from (overrides CWD)
`)
}

// loadConfig loads the global config and resolves the knowledge base root
// by walking upward from CWD (or --root).
func loadConfig() kc.Config {
	configPath := ""
	rootDir := ""
	for i, a := range os.Args {
		if a == "--config" && i+1 < len(os.Args) {
			configPath = os.Args[i+1]
		}
		if a == "--root" && i+1 < len(os.Args) {
			rootDir = os.Args[i+1]
		}
	}

	var cfg kc.Config
	if configPath != "" {
		c, err := kc.Load(configPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error loading config: %v\n", err)
			os.Exit(1)
		}
		cfg = *c
	} else {
		c, err := kc.LoadGlobal()
		if err != nil {
			if os.IsNotExist(err.(*fs.PathError)) || strings.Contains(err.Error(), "no such file") {
				fmt.Fprintf(os.Stderr, "no global config found at %s\nRun 'knowcard init' first.\n", kc.GlobalConfigPath())
			} else {
				fmt.Fprintf(os.Stderr, "error loading config: %v\n", err)
			}
			os.Exit(1)
		}
		cfg = *c
	}

	if rootDir == "" {
		rootDir, _ = os.Getwd()
	}
	root, err := kc.FindRoot(rootDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\nRun 'knowcard init' in the project root first.\n", err)
		os.Exit(1)
	}
	cfg.Root = root
	return cfg
}

func openStore() *kc.Store {
	cfg := loadConfig()
	s, err := kc.Open(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error opening store: %v\n", err)
		os.Exit(1)
	}
	return s
}

func cmdInit(args []string) {
	fs := flag.NewFlagSet("init", flag.ExitOnError)
	rootDir := fs.String("root", ".", "project root directory (default: current directory)")
	fs.Parse(args)

	// 1. Resolve and create .knowcard directory in the project
	absRoot, err := filepath.Abs(*rootDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	kcDir := filepath.Join(absRoot, kc.DirName)
	if err := os.MkdirAll(kcDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "error creating %s: %v\n", kcDir, err)
		os.Exit(1)
	}
	for _, sub := range []string{"cards", "_vcs", "index"} {
		os.MkdirAll(filepath.Join(kcDir, sub), 0755)
	}

	// 2. Ensure global config exists; create with defaults if missing
	cfgPath := kc.GlobalConfigPath()
	if _, err := os.Stat(cfgPath); os.IsNotExist(err) {
		cfg := kc.DefaultConfig()
		if err := kc.SaveGlobal(&cfg); err != nil {
			fmt.Fprintf(os.Stderr, "error saving global config: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Created global config at %s\n", cfgPath)
	}

	fmt.Printf("Initialized knowcard knowledge base at %s\n", kcDir)
	fmt.Printf("  Cards:   %s\n", filepath.Join(kcDir, "cards"))
	fmt.Printf("  VCS:     %s\n", filepath.Join(kcDir, "_vcs"))
	fmt.Printf("  Index:   %s\n", filepath.Join(kcDir, "index"))
	fmt.Printf("\nGlobal config: %s\n", cfgPath)
	fmt.Printf("Edit it to configure your embedding backend and API keys.\n")
}

func cmdRecall(args []string) {
	fs := flag.NewFlagSet("recall", flag.ExitOnError)
	topK := fs.Int("k", 10, "number of results")
	pathPrefix := fs.String("path", "", "path prefix filter")
	tagsStr := fs.String("tags", "", "comma-separated tag filter")
	asJSON := fs.Bool("json", false, "output as JSON")
	fs.Parse(args)

	query := strings.Join(fs.Args(), " ")
	if query == "" {
		fmt.Fprintln(os.Stderr, "error: query is required")
		os.Exit(1)
	}

	s := openStore()
	defer s.Close()

	var tags []string
	if *tagsStr != "" {
		tags = strings.Split(*tagsStr, ",")
	}

	results, err := s.Recall(query, kc.RecallOpts{
		TopK:     *topK,
		PathPref: *pathPrefix,
		Tags:     tags,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	if *asJSON {
		out, _ := json.MarshalIndent(results, "", "  ")
		fmt.Println(string(out))
		return
	}

	if len(results) == 0 {
		fmt.Println("No results found.")
		return
	}

	for i, r := range results {
		fmt.Printf("%2d. %s\n", i+1, r.Title)
		fmt.Printf("    id: %s\n", r.ID)
		fmt.Printf("    path: %s\n", r.Path)
		fmt.Printf("    score: %.4f (%s)\n", r.Score, r.HitType)
		fmt.Printf("    %s\n\n", r.Summary)
	}
}

func cmdShow(args []string) {
	fs := flag.NewFlagSet("show", flag.ExitOnError)
	fs.Parse(args)

	if fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "error: at least one card ID required")
		os.Exit(1)
	}

	s := openStore()
	defer s.Close()

	cards, err := s.GetCards(fs.Args())
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	for i, c := range cards {
		if i > 0 {
			fmt.Println("\n" + strings.Repeat("-", 60) + "\n")
		}
		content, _ := card.Serialize(&c)
		fmt.Println(content)
	}
}

func cmdAdd(args []string) {
	fs := flag.NewFlagSet("add", flag.ExitOnError)
	filePath := fs.String("f", "", "read card from file")
	title := fs.String("title", "", "card title")
	path := fs.String("path", "", "card path (e.g. programming/go/escape)")
	summary := fs.String("summary", "", "card summary")
	bodyText := fs.String("body", "", "card body (markdown)")
	keywordsStr := fs.String("keywords", "", "comma-separated keywords")
	tagsStr := fs.String("tags", "", "comma-separated tags")
	reference := fs.String("ref", "", "reference document path")
	fs.Parse(args)

	var c *card.Card

	if *filePath != "" {
		data, err := os.ReadFile(*filePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error reading file: %v\n", err)
			os.Exit(1)
		}
		c, err = card.Parse(string(data))
		if err != nil {
			fmt.Fprintf(os.Stderr, "error parsing card: %v\n", err)
			os.Exit(1)
		}
		if c.ID == "" {
			c.ID = card.NewID()
		}
	} else {
		if *title == "" || *path == "" || *summary == "" {
			fmt.Fprintln(os.Stderr, "error: --title, --path, and --summary are required (or use -f)")
			fs.Usage()
			os.Exit(1)
		}
		c = &card.Card{
			ID:        card.NewID(),
			Path:      *path,
			Title:     *title,
			Summary:   *summary,
			Body:      *bodyText,
			Reference: *reference,
		}
	}

	if *keywordsStr != "" {
		c.Keywords = strings.Split(*keywordsStr, ",")
	}
	if *tagsStr != "" {
		c.Tags = strings.Split(*tagsStr, ",")
	}

	s := openStore()
	defer s.Close()

	if err := s.UpsertCard(c); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Added card:\n  id: %s\n  path: %s\n  title: %s\n", c.ID, c.Path, c.Title)
}

func cmdDelete(args []string) {
	fs := flag.NewFlagSet("delete", flag.ExitOnError)
	fs.Parse(args)
	if fs.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "error: card ID required")
		os.Exit(1)
	}

	s := openStore()
	defer s.Close()

	if err := s.DeleteCard(fs.Arg(0)); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Deleted.")
}

func cmdMove(args []string) {
	fs := flag.NewFlagSet("move", flag.ExitOnError)
	fs.Parse(args)
	if fs.NArg() != 2 {
		fmt.Fprintln(os.Stderr, "error: usage: knowcard move <id> <new-path>")
		os.Exit(1)
	}

	s := openStore()
	defer s.Close()

	if err := s.MoveCard(fs.Arg(0), fs.Arg(1)); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Moved.")
}

func cmdList(args []string) {
	fs := flag.NewFlagSet("list", flag.ExitOnError)
	pathPrefix := fs.String("path", "", "path prefix filter")
	fs.Parse(args)

	s := openStore()
	defer s.Close()

	results := s.ListCards(*pathPrefix)
	for _, r := range results {
		fmt.Printf("  %-40s  %s\n", r.Path, r.Title)
	}
	if len(results) == 0 {
		fmt.Println("(no cards)")
	}
}

func cmdRebuild(args []string) {
	fs := flag.NewFlagSet("rebuild", flag.ExitOnError)
	fs.Parse(args)

	s := openStore()
	defer s.Close()

	if err := s.Rebuild(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Rebuild complete.")
}

func cmdHistory(args []string) {
	fs := flag.NewFlagSet("history", flag.ExitOnError)
	fs.Parse(args)
	if fs.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "error: card ID required")
		os.Exit(1)
	}

	s := openStore()
	defer s.Close()

	revs, err := s.History(fs.Arg(0))
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	for _, r := range revs {
		shortHash := r.Hash
		if len(shortHash) > 8 {
			shortHash = shortHash[:8]
		}
		fmt.Printf("  %s  %s  (%s)\n", shortHash, r.Message, r.When.Format("2006-01-02 15:04:05"))
	}
}
