package eval_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/eval"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
)

// demoSystem is a deterministic stand-in for a retrieval-augmented chain: it
// answers from a table keyed by the question and returns canned retrieved
// documents. Usage is reported for only some samples, so the artifacts also
// show unknown usage being excluded from means. One sample fails upstream with
// a classified error, showing failure isolation and the retry policy.
type demoSystem struct {
	answers   map[string]string
	usage     map[string]map[string]any
	retrieved map[string][]schema.Document
	fail      map[string]error
}

// Run implements eval.Target.
func (d *demoSystem) Run(ctx context.Context, sample eval.Sample) (*eval.Output, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err, ok := d.fail[sample.Question]; ok {
		return nil, err
	}
	answer, ok := d.answers[sample.Question]
	if !ok {
		return nil, fmt.Errorf("demoSystem: no canned answer for %q", sample.Question)
	}
	out := &eval.Output{
		Answer:     answer,
		StopReason: "stop",
		Retrieved:  true,
		Documents:  d.retrieved[sample.Question],
	}
	if u, ok := d.usage[sample.Question]; ok {
		choice := &llms.ContentChoice{GenerationInfo: u}
		out.Usage = eval.UsageFromGenerationInfo(choice.GenerationInfo)
	}
	return out, nil
}

// Name implements eval.Target.
func (d *demoSystem) Name() string { return "demo-rag" }

// TestWriteBenchmarkOut regenerates the sample artifacts under benchmark_out.
// It is skipped unless EVAL_WRITE_BENCHMARK_OUT=1 so ordinary test runs stay
// hermetic and fast.
func TestWriteBenchmarkOut(t *testing.T) {
	if os.Getenv("EVAL_WRITE_BENCHMARK_OUT") != "1" {
		t.Skip("set EVAL_WRITE_BENCHMARK_OUT=1 to regenerate benchmark_out/")
	}
	ctx := context.Background()

	samples, err := eval.LoadSamples("testdata/demo_samples.json")
	require.NoError(t, err)

	outDir := filepath.Join("..", "benchmark_out")
	require.NoError(t, os.MkdirAll(outDir, 0o755))

	system := demoSystemData()
	cfg := eval.DefaultConfig()
	cfg.Concurrency = 3
	cfg.ModelID = "canned-qa-v1"
	cfg.PromptVersion = "prompt-v1"
	cfg.Cache = eval.NewMemoryCache()
	cfg.Now = func() time.Time { return time.Unix(1_700_000_000, 0).UTC() }

	before, err := eval.Evaluate(ctx, samples, system, cfg)
	require.NoError(t, err)
	writeReport(t, filepath.Join(outDir, "report_before"), before.Report)

	// A prompt change: the model now answers one account question in a way
	// that drops every required key point, which is exactly the kind of
	// regression the tool must expose. The prompt version changes with it, so
	// the cache cannot mask the regression.
	cfg.PromptVersion = "prompt-v2"
	system.answers["How do I reset my password?"] = "Go to your profile page."

	after, err := eval.Evaluate(ctx, samples, system, cfg)
	require.NoError(t, err)
	writeReport(t, filepath.Join(outDir, "report_after"), after.Report)

	cmp := eval.Compare(before.Report, after.Report)
	require.True(t, cmp.Comparable)
	txt, err := os.Create(filepath.Join(outDir, "comparison.txt"))
	require.NoError(t, err)
	require.NoError(t, cmp.WriteText(txt))
	require.NoError(t, txt.Close())
	js, err := os.Create(filepath.Join(outDir, "comparison.json"))
	require.NoError(t, err)
	require.NoError(t, cmp.WriteJSON(js))
	require.NoError(t, js.Close())
}

func writeReport(t *testing.T, base string, r *eval.Report) {
	t.Helper()
	f, err := os.Create(base + "_report.txt")
	require.NoError(t, err)
	require.NoError(t, r.WriteText(f))
	require.NoError(t, f.Close())

	data, err := r.CanonicalBytes()
	require.NoError(t, err)
	var pretty any
	require.NoError(t, json.Unmarshal(data, &pretty))
	prettyData, err := json.MarshalIndent(pretty, "", "  ")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(base+"_canonical.json", append(prettyData, '\n'), 0o600))

	full, err := os.Create(base + "_full.json")
	require.NoError(t, err)
	require.NoError(t, r.WriteJSON(full))
	require.NoError(t, full.Close())
}

// demoSystemData returns the canned system used for the sample artifacts.
func demoSystemData() *demoSystem {
	return &demoSystem{
		answers: map[string]string{ // #nosec G101 -- canned demo answers, not credentials
			"How do I reset my password?":         "Open Settings, choose Security, and click Reset password.",
			"What is the refund window?":          "Refunds are available within 30 days of purchase.",
			"Which plans include SSO?":            "SSO is included in the Enterprise plan only.",
			"How do I export my data?":            "Export from the Data tab; the download link lasts 24 hours.",
			"What languages does the UI support?": "English, Chinese, and Japanese.",
			"How do I contact support?":           "Use the in-app chat or email support@example.com.",
		},
		// Only the first two samples report usage; the rest stay unknown on purpose.
		usage: map[string]map[string]any{
			"How do I reset my password?": {
				"PromptTokens": 21, "CompletionTokens": 12, "TotalTokens": 33,
			},
			"What is the refund window?": {
				"PromptTokens": 19, "CompletionTokens": 10, "TotalTokens": 29,
			},
		},
		retrieved: map[string][]schema.Document{
			// The right passage, plus one irrelevant document: recall 1,
			// precision 1/2.
			"How do I reset my password?": {
				{PageContent: "Settings > Security > Reset password lets you choose a new password."},
				{PageContent: "Billing questions are handled from the Billing tab."},
			},
			// Only an unrelated document: recall 0.
			"Which plans include SSO?": {
				{PageContent: "Teams can invite up to ten members on the Starter plan."},
			},
			// Both relevant documents, no noise.
			"How do I export my data?": {
				{PageContent: "Data tab > Export. Download links expire after 24 hours."},
			},
		},
		fail: map[string]error{
			// A permanent upstream rejection: recorded once, not retried.
			"Do you offer invoices with a purchase order number?": llms.NewError(
				llms.ErrCodeInvalidRequest, "demo", "account is not entitled to purchase-order invoices"),
		},
	}
}
