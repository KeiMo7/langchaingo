// Command eval runs the langchaingo evaluator over a sample file.
//
// It is a thin front end over the eval package: load samples, wrap the system
// under test in a target, evaluate, and write a report. With -baseline, it
// also compares the new report against a previous canonical JSON report and
// exits non-zero when a sampled metric regressed.
//
//	go run ./examples/eval-example -samples samples.json -model gpt-4o-mini
//	go run ./examples/eval-example -samples samples.json -baseline report.json
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/tmc/langchaingo/eval"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// options holds the command-line configuration.
type options struct {
	samplesPath  string
	modelName    string
	promptVer    string
	baselinePath string
	outPath      string
	concurrency  int
	timeout      time.Duration
	maxRetries   int
	useCache     bool
	jsonMode     bool
}

func parseFlags() options {
	var o options
	flag.StringVar(&o.samplesPath, "samples", "", "path to the sample file (JSON, JSON Lines, or YAML)")
	flag.StringVar(&o.modelName, "model", "gpt-4o-mini", "model name")
	flag.StringVar(&o.promptVer, "prompt-version", "v1", "prompt version; bump it whenever the prompt changes")
	flag.StringVar(&o.baselinePath, "baseline", "", "canonical report JSON to compare against")
	flag.StringVar(&o.outPath, "out", "", "write the full report JSON to this path")
	flag.IntVar(&o.concurrency, "concurrency", 2, "maximum samples evaluated at once")
	flag.DurationVar(&o.timeout, "timeout", 60*time.Second, "per-sample timeout")
	flag.IntVar(&o.maxRetries, "max-retries", 2, "retries for transient errors")
	flag.BoolVar(&o.useCache, "cache", false, "cache responses for repeated runs of the same samples")
	flag.BoolVar(&o.jsonMode, "json", false, "write JSON instead of text")
	flag.Parse()
	return o
}

func run() error {
	o := parseFlags()
	if o.samplesPath == "" {
		return fmt.Errorf("-samples is required")
	}
	samples, err := eval.LoadSamples(o.samplesPath)
	if err != nil {
		return err
	}
	target, cfg, err := buildTarget(o)
	if err != nil {
		return err
	}

	result, err := eval.Evaluate(context.Background(), samples, target, cfg)
	if err != nil {
		return err
	}
	if err := writeReport(o, result.Report); err != nil {
		return err
	}
	return compareWithBaseline(o, result.Report)
}

// buildTarget constructs the model target and evaluation config.
func buildTarget(o options) (*eval.ModelTarget, eval.Config, error) {
	model, err := openai.New(openai.WithModel(o.modelName))
	if err != nil {
		return nil, eval.Config{}, err
	}
	cfg := eval.DefaultConfig()
	cfg.Concurrency = o.concurrency
	cfg.Timeout = o.timeout
	cfg.MaxRetries = o.maxRetries
	cfg.ModelID = o.modelName
	cfg.PromptVersion = o.promptVer
	if o.useCache {
		cfg.Cache = eval.NewMemoryCache()
	}
	target := &eval.ModelTarget{
		Model: model,
		// A small fixed prompt keeps the system under test stable across runs.
		Prompt:     promptFor,
		TargetName: "openai/" + o.modelName,
	}
	return target, cfg, nil
}

func promptFor(s eval.Sample) []llms.MessageContent {
	return []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeSystem,
			Parts: []llms.ContentPart{llms.TextPart("Answer the user's question concisely. If you do not know, say so.")},
		},
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{llms.TextPart(s.Question)},
		},
	}
}

// writeReport prints the report and, when requested, saves the full JSON.
func writeReport(o options, r *eval.Report) error {
	if o.jsonMode {
		if err := r.WriteJSON(os.Stdout); err != nil {
			return err
		}
	} else if err := r.WriteText(os.Stdout); err != nil {
		return err
	}
	if o.outPath == "" {
		return nil
	}
	f, err := os.Create(o.outPath) // #nosec G304 -- caller-supplied output path
	if err != nil {
		return err
	}
	defer f.Close()
	return r.WriteJSON(f)
}

// compareWithBaseline reports deltas and fails on a drop in exact match.
func compareWithBaseline(o options, r *eval.Report) error {
	if o.baselinePath == "" {
		return nil
	}
	baseline, err := loadBaseline(o.baselinePath)
	if err != nil {
		return err
	}
	cmp := eval.Compare(baseline, r)
	if err := cmp.WriteText(os.Stdout); err != nil {
		return err
	}
	if !cmp.Comparable {
		return fmt.Errorf("baseline is not comparable: %v", cmp.Reasons)
	}
	if cmp.ExactMatch.Changed && *cmp.ExactMatch.Delta < 0 {
		return fmt.Errorf("exact match rate regressed: %.4f -> %.4f",
			*cmp.ExactMatch.Left, *cmp.ExactMatch.Right)
	}
	return nil
}

// loadBaseline reads a canonical report and decodes it into a Report. Only
// the fields needed for comparison are restored.
func loadBaseline(path string) (*eval.Report, error) {
	data, err := os.ReadFile(path) // #nosec G304 -- caller-supplied baseline path
	if err != nil {
		return nil, err
	}
	var r eval.Report
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, err
	}
	return &r, nil
}
