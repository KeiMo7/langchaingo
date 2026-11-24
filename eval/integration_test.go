package eval

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/prompts"
	"github.com/tmc/langchaingo/schema"
)

// staticRetriever returns fixed documents per query.
type staticRetriever struct {
	docs  map[string][]schema.Document
	calls int32
}

var _ schema.Retriever = (*staticRetriever)(nil)

func (r *staticRetriever) GetRelevantDocuments(_ context.Context, query string) ([]schema.Document, error) {
	atomic.AddInt32(&r.calls, 1)
	return r.docs[query], nil
}

func TestChainTargetRetrievalQA(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["How do I reset"] = "Open Settings and choose Security."
	model.answers["What is the refund"] = "Within 30 days."
	model.handler = NewRecorder()

	retriever := &staticRetriever{docs: map[string][]schema.Document{
		"How do I reset my password?": {
			{PageContent: "Settings > Security > Reset password."},
			{PageContent: "Billing FAQ."},
		},
		"What is the refund window?": {
			{PageContent: "Refunds are available within 30 days."},
		},
	}}

	qa := chains.NewRetrievalQAFromLLM(model, retriever)
	qa.ReturnSourceDocuments = true

	samples := []Sample{
		{
			ID:                 "r1",
			Question:           "How do I reset my password?",
			ReferenceAnswer:    "Open Settings and choose Security.",
			ReferenceMaterials: []string{"Settings > Security > Reset password"},
			KeyPoints:          []string{"Settings", "Security"},
			Category:           "account",
		},
		{
			ID:                 "r2",
			Question:           "What is the refund window?",
			ReferenceAnswer:    "Within 30 days.",
			ReferenceMaterials: []string{"Refunds are available within 30 days"},
			KeyPoints:          []string{"30 days"},
			Category:           "billing",
		},
	}

	res, err := Evaluate(context.Background(), samples, &ChainTarget{Chain: qa, AnswerKey: "text"}, testConfig())
	require.NoError(t, err)
	require.Equal(t, 2, res.Report.Overall.Succeeded)
	for _, s := range res.Report.Samples {
		require.True(t, s.Metrics.ExactMatch)
		require.NotNil(t, s.Metrics.Retrieval, "sample %s should have retrieval metrics", s.ID)
	}
	require.Equal(t, 1.0, *res.Report.Overall.RetrievalRecall)
	// r1 retrieved one relevant and one irrelevant document.
	require.InDelta(t, 0.75, *res.Report.Overall.RetrievalPrecision, 0.0001)
	require.Equal(t, int32(2), atomic.LoadInt32(&retriever.calls))
}

func TestChainTargetMultipleOutputKeysRequireAnswerKey(t *testing.T) {
	t.Parallel()
	ch := &recordingChain{
		outputs: map[string]any{"text": "a", "source_documents": []schema.Document{}},
		memory:  memory.NewSimple(),
		outKeys: []string{"text", "source_documents"},
	}
	// A chain with several outputs needs an explicit answer key; the sample
	// fails with the configuration error instead of guessing a value.
	res, err := Evaluate(context.Background(), sampleSet()[:1], &ChainTarget{Chain: ch}, testConfig())
	require.NoError(t, err)
	require.Equal(t, StatusFailed, res.Results[0].Status)
	require.Contains(t, res.Results[0].ErrorMessage, "AnswerKey")

	res, err = Evaluate(context.Background(), sampleSet()[:1], &ChainTarget{Chain: ch, AnswerKey: "text"}, testConfig())
	require.NoError(t, err)
	require.Equal(t, "a", res.Results[0].Answer)
}

func TestChainTargetPromptVersionChangesCacheKey(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"

	llmChain := chains.NewLLMChain(model, prompts.NewPromptTemplate("{{.question}}", []string{"question"}))
	cache := NewMemoryCache()
	cfg := testConfig()
	cfg.Cache = cache
	cfg.ModelID = "m"
	cfg.PromptVersion = "v1"

	target := &ChainTarget{Chain: llmChain}
	_, err := Evaluate(context.Background(), sampleSet()[:1], target, cfg)
	require.NoError(t, err)
	require.Equal(t, 1, model.callCount())

	cfg.PromptVersion = "v2"
	_, err = Evaluate(context.Background(), sampleSet()[:1], target, cfg)
	require.NoError(t, err)
	require.Equal(t, 2, model.callCount(), "a new prompt version must not reuse cached responses")
}

func TestCachedAndUncachedReportsAreIdentical(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"
	model.answers["beta"] = "beta answer"
	model.answers["gamma"] = "gamma answer"
	model.usage["alpha"] = map[string]any{"PromptTokens": 3, "CompletionTokens": 2}

	plainConfig := testConfig()
	plainConfig.ModelID = "m"
	plainConfig.PromptVersion = "v1"
	uncached, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, plainConfig)
	require.NoError(t, err)

	cachedConfig := plainConfig
	cachedConfig.Cache = NewMemoryCache()
	warm, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cachedConfig)
	require.NoError(t, err)
	hit, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cachedConfig)
	require.NoError(t, err)

	require.Equal(t, 3, hit.Report.Meta.CacheHits)
	a, err := uncached.Report.CanonicalBytes()
	require.NoError(t, err)
	b, err := warm.Report.CanonicalBytes()
	require.NoError(t, err)
	c, err := hit.Report.CanonicalBytes()
	require.NoError(t, err)
	require.Equal(t, string(a), string(b))
	require.Equal(t, string(b), string(c))
}

// blockingModel blocks until released, so a cancellation test can guarantee a
// sample is in flight when the run is canceled.
type blockingModel struct {
	started chan struct{}
	release chan struct{}
	once    sync.Once
	answers map[string]string
}

func (m *blockingModel) GenerateContent(ctx context.Context, messages []llms.MessageContent, _ ...llms.CallOption) (*llms.ContentResponse, error) {
	m.once.Do(func() { close(m.started) })
	select {
	case <-m.release:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	q := ""
	if len(messages) > 0 && len(messages[0].Parts) > 0 {
		if tp, ok := messages[0].Parts[0].(llms.TextContent); ok {
			q = tp.Text
		}
	}
	return &llms.ContentResponse{Choices: []*llms.ContentChoice{{Content: m.answers[q]}}}, nil
}

func (m *blockingModel) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, m, prompt, options...)
}

func TestCancellationWaitsForInFlightSample(t *testing.T) {
	t.Parallel()
	model := &blockingModel{
		started: make(chan struct{}),
		release: make(chan struct{}),
		answers: map[string]string{"alpha question": "alpha answer"},
	}
	// Give the in-flight sample time to be started before canceling, and
	// release it afterward so the worker can converge.
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-model.started
		cancel()
		time.Sleep(20 * time.Millisecond)
		close(model.release)
	}()

	cfg := testConfig()
	cfg.Concurrency = 1
	res, err := Evaluate(ctx, sampleSet(), &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)
	require.False(t, res.Report.Complete)
	require.Equal(t, StatusCanceled, res.Results[0].Status)
	require.Equal(t, StatusNotRun, res.Results[1].Status)
	require.Equal(t, 1, res.Report.Overall.Canceled)
	require.Equal(t, 2, res.Report.Overall.NotRun)
}

func TestComparePromptVersionIsAllowed(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"
	model.answers["beta"] = "beta answer"
	model.answers["gamma"] = "gamma answer"

	cfg := testConfig()
	cfg.ModelID = "m"
	cfg.PromptVersion = "v1"
	left, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)

	cfg.PromptVersion = "v2"
	model.answers["beta"] = "changed"
	right, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)

	cmp := Compare(left.Report, right.Report)
	require.True(t, cmp.Comparable)
	require.Len(t, cmp.Changed, 1)
}

func TestReportTextIsStable(t *testing.T) {
	t.Parallel()
	model := newFakeModel()
	model.answers["alpha"] = "alpha answer"
	model.answers["beta"] = "beta answer"
	model.answers["gamma"] = "gamma answer"

	cfg := testConfig()
	cfg.Now = func() time.Time { return time.Unix(42, 0).UTC() }
	a, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)
	b, err := Evaluate(context.Background(), sampleSet(), &ModelTarget{Model: model}, cfg)
	require.NoError(t, err)

	var bufA, bufB stringsBuilder
	require.NoError(t, a.Report.WriteText(&bufA))
	require.NoError(t, b.Report.WriteText(&bufB))
	require.Equal(t, bufA.String(), bufB.String())
}

type stringsBuilder struct{ b []byte }

func (s *stringsBuilder) Write(p []byte) (int, error) {
	s.b = append(s.b, p...)
	return len(p), nil
}

func (s *stringsBuilder) String() string { return string(s.b) }
