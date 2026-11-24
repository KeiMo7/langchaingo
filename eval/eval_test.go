package eval

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/prompts"
	"github.com/tmc/langchaingo/schema"
)

// fakeModel is a controllable llms.Model for tests. Per-sample behavior is
// keyed by a substring of the prompt so tests can express all-correct,
// all-wrong, partial-failure, timeout, and missing-usage runs.
type fakeModel struct {
	mu sync.Mutex

	answers map[string]string
	usage   map[string]map[string]any
	errs    map[string]error
	delay   time.Duration
	calls   int32

	// handler, when set, receives LLM callbacks the way a provider's own
	// CallbacksHandler field would.
	handler callbacks.Handler
}

var _ llms.Model = (*fakeModel)(nil)

func newFakeModel() *fakeModel {
	return &fakeModel{
		answers: make(map[string]string),
		usage:   make(map[string]map[string]any),
		errs:    make(map[string]error),
	}
}

func promptText(messages []llms.MessageContent) string {
	if len(messages) == 0 || len(messages[0].Parts) == 0 {
		return ""
	}
	if tp, ok := messages[0].Parts[0].(llms.TextContent); ok {
		return tp.Text
	}
	return ""
}

func contains(s, sub string) bool { return strings.Contains(s, sub) }

func (m *fakeModel) GenerateContent(ctx context.Context, messages []llms.MessageContent, _ ...llms.CallOption) (*llms.ContentResponse, error) {
	atomic.AddInt32(&m.calls, 1)
	prompt := promptText(messages)
	if m.delay > 0 {
		select {
		case <-time.After(m.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.Lock()
	key := ""
	for needle := range m.answers {
		if contains(prompt, needle) {
			key = needle
			break
		}
	}
	if key == "" {
		for needle := range m.errs {
			if contains(prompt, needle) {
				key = needle
				break
			}
		}
	}
	answer, hasAnswer := m.answers[key]
	callErr, hasErr := m.errs[key]
	info := m.usage[key]
	m.mu.Unlock()

	if hasErr {
		return nil, callErr
	}
	if !hasAnswer {
		return nil, fmt.Errorf("fakeModel: no answer configured for prompt %q", prompt)
	}
	choice := &llms.ContentChoice{Content: answer, StopReason: "stop"}
	if info != nil {
		choice.GenerationInfo = info
	}
	resp := &llms.ContentResponse{Choices: []*llms.ContentChoice{choice}}
	if m.handler != nil {
		m.handler.HandleLLMGenerateContentEnd(ctx, resp)
	}
	return resp, nil
}

func (m *fakeModel) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, m, prompt, options...)
}

func (m *fakeModel) callCount() int { return int(atomic.LoadInt32(&m.calls)) }

func sampleSet() []Sample {
	return []Sample{
		{ID: "q1", Question: "alpha question", ReferenceAnswer: "alpha answer", Category: "a"},
		{ID: "q2", Question: "beta question", ReferenceAnswer: "beta answer", Category: "a"},
		{ID: "q3", Question: "gamma question", ReferenceAnswer: "gamma answer", Category: "b"},
	}
}

func TestEvaluateAllCorrect(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"
	model.answers["beta"] = "beta answer"
	model.answers["gamma"] = "gamma answer"

	res, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, testConfig())
	require.NoError(t, err)
	require.True(t, res.Report.Complete)
	require.Equal(t, 3, res.Report.Overall.Succeeded)
	require.Equal(t, *floatPtr(1), *res.Report.Overall.ExactMatchRate)
	require.Equal(t, *floatPtr(1), *res.Report.Overall.TokenF1)
	require.Equal(t, *floatPtr(1), *res.Report.Overall.KeyPointCoverage)
	require.Empty(t, res.Report.Overall.Failures)
}

func TestEvaluateAllWrong(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "something else entirely"
	model.answers["beta"] = "nope"
	model.answers["gamma"] = "unrelated"

	res, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, testConfig())
	require.NoError(t, err)
	require.Equal(t, 3, res.Report.Overall.Succeeded)
	require.Equal(t, *floatPtr(0), *res.Report.Overall.ExactMatchRate)
	require.Equal(t, *floatPtr(0), *res.Report.Overall.KeyPointCoverage)
}

func TestEvaluatePartialFailureIsIsolated(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"
	model.answers["gamma"] = "gamma answer"
	model.errs["beta"] = llms.NewError(llms.ErrCodeInvalidRequest, "fake", "bad request")

	res, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, testConfig())
	require.NoError(t, err)
	require.Equal(t, 2, res.Report.Overall.Succeeded)
	require.Equal(t, 1, res.Report.Overall.Failed)
	require.Equal(t, []FailureCount{{Class: string(ErrorClassInvalidRequest), Count: 1}}, res.Report.Overall.Failures)

	// A permanent error is not retried.
	var beta *SampleResult
	for i := range res.Results {
		if res.Results[i].ID == "q2" {
			beta = &res.Results[i]
		}
	}
	require.NotNil(t, beta)
	require.Equal(t, StatusFailed, beta.Status)
	require.Equal(t, 1, beta.Attempts)
}

func TestEvaluateRetriesTransientError(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"
	model.answers["beta"] = "beta answer"
	model.answers["gamma"] = "gamma answer"

	cfg := testConfig()
	cfg.MaxRetries = 2
	cfg.RetryBackoff = time.Millisecond

	res, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)
	require.Equal(t, 3, res.Report.Overall.Succeeded)
}

func TestEvaluateTimeoutMarksFailure(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"
	model.answers["gamma"] = "gamma answer"
	model.delay = 200 * time.Millisecond

	cfg := testConfig()
	cfg.Timeout = 20 * time.Millisecond
	cfg.MaxRetries = 0

	res, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)
	require.Equal(t, 3, res.Report.Overall.Failed)
	require.Equal(t, []FailureCount{{Class: string(ErrorClassTimeout), Count: 3}}, res.Report.Overall.Failures)
}

func TestEvaluateUsageUnknownIsNotZero(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"
	model.answers["beta"] = "beta answer"
	model.answers["gamma"] = "gamma answer"
	// One sample reports usage, two do not.
	model.usage["alpha"] = map[string]any{"PromptTokens": 10, "CompletionTokens": 5, "TotalTokens": 15}

	res, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, testConfig())
	require.NoError(t, err)
	require.Equal(t, 1, res.Report.Overall.UsageReported)
	require.Equal(t, 2, res.Report.Overall.UsageUnknown)
	require.Equal(t, *floatPtr(10), *res.Report.Overall.PromptTokens)
	require.Equal(t, *floatPtr(5), *res.Report.Overall.CompletionTokens)
	require.Equal(t, *floatPtr(15), *res.Report.Overall.TotalTokens)

	for _, s := range res.Report.Samples {
		if s.ID == "q1" {
			require.True(t, s.Usage.PromptTokens.Known)
		} else {
			require.False(t, s.Usage.PromptTokens.Known, "sample %s usage should be unknown", s.ID)
		}
	}
}

func TestEvaluateConcurrencyPreservesInputOrder(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	var samples []Sample
	for i := 0; i < 20; i++ {
		q := fmt.Sprintf("question %02d", i)
		samples = append(samples, Sample{
			ID:              fmt.Sprintf("q%02d", i),
			Question:        q,
			ReferenceAnswer: fmt.Sprintf("answer %02d", i),
		})
		model.answers[q] = fmt.Sprintf("answer %02d", i)
	}
	model.delay = time.Millisecond

	cfg := testConfig()
	cfg.Concurrency = 8
	res, err := Evaluate(context.Background(), samples, &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)
	require.Len(t, res.Results, len(samples))
	for i, s := range res.Results {
		require.Equal(t, samples[i].ID, s.ID, "result %d out of order", i)
		require.Equal(t, StatusSucceeded, s.Status)
	}
	// The report view is sorted by ID.
	require.Equal(t, res.Report.Samples[0].ID, "q00")
	require.Equal(t, res.Report.Samples[len(res.Report.Samples)-1].ID, "q19")
}

func TestEvaluateCancellationMarksIncomplete(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	var samples []Sample
	for i := 0; i < 10; i++ {
		q := fmt.Sprintf("question %02d", i)
		samples = append(samples, Sample{ID: fmt.Sprintf("q%02d", i), Question: q, ReferenceAnswer: q})
		model.answers[q] = q
	}
	model.delay = 100 * time.Millisecond

	cfg := testConfig()
	cfg.Concurrency = 1

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	res, err := Evaluate(ctx, samples, &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)
	require.False(t, res.Report.Complete)
	require.Positive(t, res.Report.Overall.NotRun, "expected undispatched samples")
	require.Greater(t, res.Report.Overall.Samples, 0)
	require.Equal(t, len(samples), res.Report.Overall.Samples)
}

func TestEvaluateReportIsByteStable(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"
	model.answers["beta"] = "beta answer"
	model.answers["gamma"] = "gamma answer"

	cfg := testConfig()
	cfg.Now = func() time.Time { return time.Unix(100, 0) }

	first, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)
	second, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)

	a, err := first.Report.CanonicalBytes()
	require.NoError(t, err)
	b, err := second.Report.CanonicalBytes()
	require.NoError(t, err)
	require.Equal(t, string(a), string(b))
}

func TestEvaluateCacheHitAvoidsCalls(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"
	model.answers["beta"] = "beta answer"
	model.answers["gamma"] = "gamma answer"

	cache := NewMemoryCache()
	cfg := testConfig()
	cfg.Cache = cache
	cfg.ModelID = "fake-v1"
	cfg.PromptVersion = "prompt-v1"

	first, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)
	require.Equal(t, 3, model.callCount())
	require.Zero(t, first.Report.Meta.CacheHits)

	second, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)
	require.Equal(t, 3, model.callCount(), "second run must be served from cache")
	require.Equal(t, 3, second.Report.Meta.CacheHits)
	for _, s := range second.Results {
		require.True(t, s.CacheHit)
	}

	// A prompt version change invalidates the cache.
	cfg.PromptVersion = "prompt-v2"
	third, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)
	require.Equal(t, 6, model.callCount(), "prompt version change must miss the cache")
	require.Zero(t, third.Report.Meta.CacheHits)
}

func TestEvaluateCategories(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"
	model.answers["beta"] = "wrong"
	model.answers["gamma"] = "gamma answer"

	res, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, testConfig())
	require.NoError(t, err)
	require.Len(t, res.Report.Categories, 2)
	require.Equal(t, "a", res.Report.Categories[0].Category)
	require.Equal(t, 2, res.Report.Categories[0].Samples)
	require.Equal(t, *floatPtr(0.5), *res.Report.Categories[0].ExactMatchRate)
	require.Equal(t, "b", res.Report.Categories[1].Category)
	require.Equal(t, *floatPtr(1), *res.Report.Categories[1].ExactMatchRate)
}

func TestEvaluateRequiresUsableSamples(t *testing.T) {
	t.Parallel()
	_, err := Evaluate(context.Background(), nil, &ModelTarget{Model: newFakeModel()}, testConfig())
	require.Error(t, err)

	bad := []Sample{{ID: "a", Question: "q", ReferenceAnswer: "r"}, {ID: "a", Question: "q", ReferenceAnswer: "r"}}
	_, err = Evaluate(context.Background(), bad, &ModelTarget{Model: newFakeModel()}, testConfig())
	require.ErrorContains(t, err, "duplicate sample ID")
}

func testConfig() Config {
	cfg := DefaultConfig()
	cfg.RetryBackoff = time.Millisecond
	return cfg
}

// recordingChain records the inputs it was called with, to prove that the
// chain's own interface is unchanged by evaluation.
type recordingChain struct {
	inputs  []map[string]any
	outputs map[string]any
	memory  schema.Memory
	outKeys []string
}

func (c *recordingChain) Call(_ context.Context, inputs map[string]any, _ ...chains.ChainCallOption) (map[string]any, error) {
	c.inputs = append(c.inputs, inputs)
	return c.outputs, nil
}

func (c *recordingChain) GetMemory() schema.Memory { return c.memory }
func (c *recordingChain) GetInputKeys() []string   { return []string{"question"} }
func (c *recordingChain) GetOutputKeys() []string {
	if c.outKeys != nil {
		return c.outKeys
	}
	return []string{"text"}
}

func TestChainTargetUsesExistingChain(t *testing.T) {
	t.Parallel()
	ch := &recordingChain{
		outputs: map[string]any{"text": "alpha answer"},
		memory:  memory.NewSimple(),
	}
	res, err := Evaluate(context.Background(), sampleSet()[:1], &ChainTarget{Chain: ch}, testConfig())
	require.NoError(t, err)
	require.Equal(t, StatusSucceeded, res.Results[0].Status)
	require.Equal(t, "alpha answer", res.Results[0].Answer)
	require.Len(t, ch.inputs, 1)
	require.Equal(t, "alpha question", ch.inputs[0]["question"])
}

func TestChainTargetLLMChainWithRecorder(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"
	model.usage["alpha"] = map[string]any{"InputTokens": 7, "OutputTokens": 3}

	// A recorder attached at model construction captures usage from callbacks,
	// exactly as with a real provider. The capture is carried by the call
	// context, so concurrent samples stay isolated.
	model.handler = NewRecorder()

	llmChain := chains.NewLLMChain(model, prompts.NewPromptTemplate("{{.question}}", []string{"question"}))
	ch := &ChainTarget{
		Chain:       llmChain,
		CallOptions: []chains.ChainCallOption{chains.WithCallback(NewRecorder())},
	}
	// The chain has a single output key ("text"); no AnswerKey needed.
	res, err := Evaluate(context.Background(), sampleSet()[:1], ch, testConfig())
	require.NoError(t, err)
	require.Equal(t, StatusSucceeded, res.Results[0].Status)
	require.Equal(t, "alpha answer", res.Results[0].Answer)
	require.True(t, res.Results[0].Usage.PromptTokens.Known)
	require.Equal(t, int64(7), res.Results[0].Usage.PromptTokens.Value)
	require.Equal(t, int64(3), res.Results[0].Usage.CompletionTokens.Value)
	require.Equal(t, int64(10), res.Results[0].Usage.TotalTokens.Value)
}

func TestRetrievalMetrics(t *testing.T) {
	t.Parallel()
	sample := Sample{
		ID:                 "r1",
		Question:           "how",
		ReferenceAnswer:    "answer",
		ReferenceMaterials: []string{"Password reset lives in Security"},
	}
	out := &Output{
		Answer:    "answer",
		Retrieved: true,
		Documents: []schema.Document{
			{PageContent: "Settings > Security: Password reset lives in Security."},
			{PageContent: "Unrelated billing FAQ."},
		},
	}
	m := Score(sample, out)
	require.NotNil(t, m.Retrieval)
	require.Equal(t, 1, m.Retrieval.MatchedRefs)
	require.Equal(t, 1.0, m.Retrieval.Recall)
	require.Equal(t, 0.5, m.Retrieval.Precision)

	// No retriever => no retrieval metrics, even with reference materials.
	out.Retrieved = false
	require.Nil(t, Score(sample, out).Retrieval)
}

func TestUsageFromGenerationInfoAliases(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		info map[string]any
		want Usage
	}{
		{
			name: "openai",
			info: map[string]any{"PromptTokens": 4, "CompletionTokens": 6, "TotalTokens": 10},
			want: Usage{tokenCount(4), tokenCount(6), tokenCount(10)},
		},
		{
			name: "anthropic",
			info: map[string]any{"InputTokens": int32(4), "OutputTokens": int32(6)},
			want: Usage{tokenCount(4), tokenCount(6), tokenCount(10)},
		},
		{
			name: "bedrock",
			info: map[string]any{"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
			want: Usage{tokenCount(4), tokenCount(6), tokenCount(10)},
		},
		{
			name: "missing",
			info: map[string]any{"model": "x"},
			want: Usage{},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			require.Equal(t, tc.want, UsageFromGenerationInfo(tc.info))
		})
	}
}

func TestCompareReports(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"
	model.answers["beta"] = "beta answer"
	model.answers["gamma"] = "gamma answer"

	left, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, testConfig())
	require.NoError(t, err)

	model.answers["beta"] = "something else"
	right, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, testConfig())
	require.NoError(t, err)

	cmp := Compare(left.Report, right.Report)
	require.True(t, cmp.Comparable)
	require.True(t, cmp.ExactMatch.Changed)
	require.InDelta(t, -1.0/3.0, *cmp.ExactMatch.Delta, 0.0001)
	require.Len(t, cmp.Changed, 1)
	require.Equal(t, "q2", cmp.Changed[0].ID)
	require.True(t, cmp.Changed[0].LeftExactMatch)
	require.False(t, cmp.Changed[0].RightExactMatch)
}

func TestCompareRejectsDifferentModels(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"
	model.answers["beta"] = "beta answer"
	model.answers["gamma"] = "gamma answer"

	cfg := testConfig()
	cfg.ModelID = "m1"
	left, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)
	cfg.ModelID = "m2"
	right, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)

	cmp := Compare(left.Report, right.Report)
	require.False(t, cmp.Comparable)
	require.Contains(t, cmp.Reasons, "model ID differs")
}

func TestClassifyError(t *testing.T) {
	t.Parallel()
	require.Equal(t, ErrorClassTimeout, classifyError(context.DeadlineExceeded))
	require.Equal(t, ErrorClassCanceled, classifyError(context.Canceled))
	require.Equal(t, ErrorClassRateLimit, classifyError(llms.NewError(llms.ErrCodeRateLimit, "p", "slow down")))
	require.Equal(t, ErrorClassProviderUnavailable, classifyError(llms.NewError(llms.ErrCodeProviderUnavailable, "p", "down")))
	require.Equal(t, ErrorClassAuthentication, classifyError(llms.NewError(llms.ErrCodeAuthentication, "p", "denied")))
	require.Equal(t, ErrorClassOther, classifyError(errors.New("boom")))
	require.True(t, ErrorClassTimeout.retryable())
	require.True(t, ErrorClassRateLimit.retryable())
	require.False(t, ErrorClassAuthentication.retryable())
	require.False(t, ErrorClassInvalidRequest.retryable())
}

func TestCacheKeyDependsOnContentModelAndPrompt(t *testing.T) {
	t.Parallel()
	cfg := Config{ModelID: "m1", PromptVersion: "p1"}
	sample := Sample{ID: "a", Question: "q", ReferenceAnswer: "r"}
	base := cacheKey(cfg, sample)

	otherSample := sample
	otherSample.Question = "different"
	require.NotEqual(t, base, cacheKey(cfg, otherSample), "sample content must affect the key")

	otherModel := cfg
	otherModel.ModelID = "m2"
	require.NotEqual(t, base, cacheKey(otherModel, sample), "model ID must affect the key")

	otherPrompt := cfg
	otherPrompt.PromptVersion = "p2"
	require.NotEqual(t, base, cacheKey(otherPrompt, sample), "prompt version must affect the key")

	otherSalt := cfg
	otherSalt.CacheSalt = "retriever=top5"
	require.NotEqual(t, base, cacheKey(otherSalt, sample), "salt must affect the key")

	require.Equal(t, base, cacheKey(cfg, sample), "the key must be stable")
}

func TestCacheRequiresModelAndPromptVersion(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"

	cfg := testConfig()
	cfg.Cache = NewMemoryCache()
	cfg.ModelID = "m"
	_, err := Evaluate(context.Background(), sampleSet()[:1], &ModelTarget{Model: model}, cfg)
	require.ErrorContains(t, err, "PromptVersion")

	cfg.PromptVersion = "p"
	cfg.ModelID = ""
	_, err = Evaluate(context.Background(), sampleSet()[:1], &ModelTarget{Model: model}, cfg)
	require.ErrorContains(t, err, "ModelID")
}

func TestFailedSamplesAreNotScored(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"
	model.answers["gamma"] = "gamma answer"
	model.errs["beta"] = llms.NewError(llms.ErrCodeInvalidRequest, "fake", "bad request")

	res, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, testConfig())
	require.NoError(t, err)
	for _, s := range res.Report.Samples {
		if s.Status == StatusFailed {
			require.Nil(t, s.Metrics, "failed sample %s must not carry zero metrics", s.ID)
		} else {
			require.NotNil(t, s.Metrics)
		}
	}
	// The overall mean covers the scored samples only.
	require.Equal(t, *floatPtr(1), *res.Report.Overall.ExactMatchRate)
	require.Equal(t, 2, res.Report.Overall.Succeeded)
}

func TestCompareDetectsFailureBecomingSuccess(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"
	model.answers["gamma"] = "gamma answer"
	model.errs["beta"] = llms.NewError(llms.ErrCodeProviderUnavailable, "fake", "down")

	cfg := testConfig()
	cfg.MaxRetries = 0
	left, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)
	require.Equal(t, 1, left.Report.Overall.Failed)

	// The upstream recovers: q2 now succeeds. Its metrics cannot be
	// compared (the left side was never scored), but the status change must
	// still be reported.
	delete(model.errs, "beta")
	model.answers["beta"] = "beta answer"
	right, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)

	cmp := Compare(left.Report, right.Report)
	require.True(t, cmp.Comparable)
	require.Equal(t, 1, cmp.FailedLeft)
	require.Equal(t, 0, cmp.FailedRight)
	require.Len(t, cmp.Changed, 1)
	require.Equal(t, "q2", cmp.Changed[0].ID)
	require.Equal(t, StatusFailed, cmp.Changed[0].LeftStatus)
	require.Equal(t, StatusSucceeded, cmp.Changed[0].RightStatus)
	require.Nil(t, cmp.Changed[0].TokenF1.Delta, "unscored left side has no token F1 delta")
}
