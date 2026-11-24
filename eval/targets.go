package eval

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
)

// ModelTarget evaluates an [llms.Model] directly with a single-prompt call.
// It is the simplest way to compare models or call options on a fixed prompt.
type ModelTarget struct {
	// Model is called once per sample.
	Model llms.Model
	// Options are applied to every call.
	Options []llms.CallOption
	// Prompt renders the messages sent for a sample. When nil, the sample
	// question is sent as a single human message. The default prompt does not
	// include the reference answer, so the evaluation measures the model, not
	// its ability to copy an answer it was shown.
	Prompt func(Sample) []llms.MessageContent
	// TargetName is reported in the report. It defaults to "model".
	TargetName string
}

var _ Target = (*ModelTarget)(nil)

// Run calls the model and extracts the answer, stop reason, and usage.
func (t *ModelTarget) Run(ctx context.Context, sample Sample) (*Output, error) {
	if t.Model == nil {
		return nil, errors.New("eval: ModelTarget has nil Model")
	}
	messages := []llms.MessageContent{{
		Role:  llms.ChatMessageTypeHuman,
		Parts: []llms.ContentPart{llms.TextPart(sample.Question)},
	}}
	if t.Prompt != nil {
		messages = t.Prompt(sample)
	}
	resp, err := t.Model.GenerateContent(ctx, messages, t.Options...)
	if err != nil {
		return nil, err
	}
	if resp == nil || len(resp.Choices) == 0 || resp.Choices[0] == nil {
		return nil, errors.New("eval: model returned no choices")
	}
	choice := resp.Choices[0]
	content := choice.Content
	if content == "" && len(choice.Parts) > 0 {
		content = textFromParts(choice.Parts)
	}
	return &Output{
		Answer:     content,
		StopReason: choice.StopReason,
		Usage:      UsageFromGenerationInfo(choice.GenerationInfo),
	}, nil
}

// Name implements Target.
func (t *ModelTarget) Name() string {
	if t.TargetName == "" {
		return "model"
	}
	return t.TargetName
}

// textFromParts concatenates the textual parts of a response choice.
func textFromParts(parts []llms.ContentPart) string {
	var b strings.Builder
	for _, p := range parts {
		switch part := p.(type) {
		case llms.TextContent:
			b.WriteString(part.Text)
		case llms.ThinkingContent:
			// Reasoning is not part of the answer.
		}
	}
	return b.String()
}

// ChainTarget evaluates an existing [chains.Chain], such as an LLMChain or a
// retrieval QA chain. The chain's interface is untouched: the adapter only
// supplies inputs and reads outputs.
// The chain's memory is used as the chain normally uses it. A chain with
// conversation memory keeps that state across samples, so evaluating such a
// chain should use Config.Concurrency = 1 and an explicit Memory reset, or the
// samples will influence each other.
//
// A chain that makes several model calls is scored on its final answer; the
// capture records the first call's stop reason and usage, so multi-call usage
// is a lower bound.
type ChainTarget struct {
	// Chain is called once per sample.
	Chain chains.Chain
	// Inputs builds the chain's input map for a sample. When nil, the chain
	// must declare exactly one input key, which receives the sample question.
	// That covers single-input chains such as RetrievalQA and most LLMChains;
	// chains with more inputs need an explicit function.
	Inputs func(Sample) map[string]any
	// CallOptions are passed to every chain call.
	CallOptions []chains.ChainCallOption
	// AnswerKey selects the output value under evaluation. When empty, the
	// chain's single output key is used; a chain with several output keys must
	// name one.
	AnswerKey string
	// DocumentsKey selects a []schema.Document output to score retrieval
	// against. When empty it defaults to "source_documents", the key used by
	// RetrievalQA when ReturnSourceDocuments is enabled. Set it to "-" to
	// disable reading retrieval documents from the output.
	DocumentsKey string
	// TargetName is reported in the report. It defaults to "chain".
	TargetName string
}

var _ Target = (*ChainTarget)(nil)

// Run calls the chain and extracts the answer, and any source documents.
func (t *ChainTarget) Run(ctx context.Context, sample Sample) (*Output, error) {
	if t.Chain == nil {
		return nil, errors.New("eval: ChainTarget has nil Chain")
	}
	inputs, err := t.inputsFor(sample)
	if err != nil {
		return nil, err
	}
	capture := NewCapture()
	ctx = WithCapture(ctx, capture)

	values, err := chains.Call(ctx, t.Chain, inputs, t.CallOptions...)
	if err != nil {
		return nil, err
	}
	answerKey := t.AnswerKey
	if answerKey == "" {
		keys := t.Chain.GetOutputKeys()
		if len(keys) != 1 {
			return nil, fmt.Errorf("eval: chain has %d output keys; set ChainTarget.AnswerKey", len(keys))
		}
		answerKey = keys[0]
	}
	answer, ok := values[answerKey]
	if !ok {
		return nil, fmt.Errorf("eval: chain output is missing answer key %q", answerKey)
	}
	answerText, ok := answer.(string)
	if !ok {
		answerText = fmt.Sprint(answer)
	}

	out := &Output{Answer: answerText}
	out.StopReason, out.Usage = capture.Result()

	docsKey := t.DocumentsKey
	if docsKey == "" {
		docsKey = "source_documents"
	}
	if docsKey != "-" {
		if docs, ok := values[docsKey].([]schema.Document); ok {
			out.Documents = docs
			out.Retrieved = true
		}
	}
	if !out.Retrieved {
		if docs, ok := capture.Documents(); ok {
			out.Documents = docs
			out.Retrieved = true
		}
	}
	return out, nil
}

// Name implements Target.
func (t *ChainTarget) Name() string {
	if t.TargetName == "" {
		return "chain"
	}
	return t.TargetName
}

// inputsFor builds the chain inputs, requiring an explicit builder for chains
// with more than one input key.
func (t *ChainTarget) inputsFor(s Sample) (map[string]any, error) {
	if t.Inputs != nil {
		return t.Inputs(s), nil
	}
	keys := t.Chain.GetInputKeys()
	if len(keys) != 1 {
		return nil, fmt.Errorf("eval: chain has %d input keys; set ChainTarget.Inputs", len(keys))
	}
	return map[string]any{keys[0]: s.Question}, nil
}

// Capture collects the stop reason, usage, and retriever documents emitted by
// callbacks during one target call. It is stored in a context so that an
// existing handler can populate it per call without changing the chain or
// model signatures.
type Capture struct {
	mu        sync.Mutex
	usage     Usage
	hasUsage  bool
	stop      string
	documents []schema.Document
	hasDocs   bool
}

// NewCapture returns an empty capture.
func NewCapture() *Capture { return &Capture{} }

type captureKey struct{}

// WithCapture returns a context carrying c.
func WithCapture(ctx context.Context, c *Capture) context.Context {
	return context.WithValue(ctx, captureKey{}, c)
}

// CaptureFromContext returns the capture in ctx, or nil.
func CaptureFromContext(ctx context.Context) *Capture {
	c, _ := ctx.Value(captureKey{}).(*Capture)
	return c
}

// Result returns the recorded stop reason and usage.
func (c *Capture) Result() (string, Usage) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.stop, c.usage
}

// Documents returns the recorded retriever documents and whether a retriever
// callback was observed.
func (c *Capture) Documents() ([]schema.Document, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]schema.Document(nil), c.documents...), c.hasDocs
}

// Recorder is a [callbacks.Handler] that fills the [Capture] found in the
// call context. Attach it to a chain or model whose calls should contribute
// stop reasons, token usage, and retrieved documents to an evaluation:
//
//	llmChain.CallbacksHandler = eval.NewRecorder()
//	target := &eval.ChainTarget{Chain: llmChain}
//
// Without a recorder, answers are still scored; stop reason and usage are
// simply reported as unknown.
type Recorder struct {
	callbacks.SimpleHandler
}

// NewRecorder returns a recorder.
func NewRecorder() *Recorder { return &Recorder{} }

var _ callbacks.Handler = (*Recorder)(nil)

// HandleLLMGenerateContentEnd records usage and the stop reason.
func (r *Recorder) HandleLLMGenerateContentEnd(ctx context.Context, resp *llms.ContentResponse) {
	c := CaptureFromContext(ctx)
	if c == nil || resp == nil || len(resp.Choices) == 0 || resp.Choices[0] == nil {
		return
	}
	choice := resp.Choices[0]
	c.mu.Lock()
	if !c.hasUsage {
		if usage := UsageFromGenerationInfo(choice.GenerationInfo); usage.anyKnown() {
			c.usage = usage
			c.hasUsage = true
		}
	}
	if c.stop == "" {
		c.stop = choice.StopReason
	}
	c.mu.Unlock()
}

// HandleRetrieverEnd records the documents a retriever returned.
func (r *Recorder) HandleRetrieverEnd(ctx context.Context, _ string, documents []schema.Document) {
	c := CaptureFromContext(ctx)
	if c == nil {
		return
	}
	c.mu.Lock()
	c.documents = append([]schema.Document(nil), documents...)
	c.hasDocs = true
	c.mu.Unlock()
}

func (u Usage) anyKnown() bool {
	return u.PromptTokens.Known || u.CompletionTokens.Known || u.TotalTokens.Known
}

// provider token keys, in the order they are consulted. Providers report
// usage under provider-family-specific names; unknown keys are ignored.
var (
	promptTokenKeys     = []string{"PromptTokens", "prompt_tokens", "InputTokens", "input_tokens"}
	completionTokenKeys = []string{"CompletionTokens", "completion_tokens", "OutputTokens", "output_tokens"}
	totalTokenKeys      = []string{"TotalTokens", "total_tokens"}
)

// UsageFromGenerationInfo extracts token counts from a provider's
// GenerationInfo map. Fields that are absent, non-numeric, or negative stay
// unknown; they are never reported as zero.
func UsageFromGenerationInfo(info map[string]any) Usage {
	var u Usage
	u.PromptTokens = firstTokenCount(info, promptTokenKeys)
	u.CompletionTokens = firstTokenCount(info, completionTokenKeys)
	u.TotalTokens = firstTokenCount(info, totalTokenKeys)
	if !u.TotalTokens.Known && u.PromptTokens.Known && u.CompletionTokens.Known {
		u.TotalTokens = tokenCount(u.PromptTokens.Value + u.CompletionTokens.Value)
	}
	return u
}

func firstTokenCount(info map[string]any, keys []string) TokenCount {
	for _, k := range keys {
		if v, ok := info[k]; ok {
			if n, ok := toInt64(v); ok && n >= 0 {
				return TokenCount{Value: n, Known: true}
			}
		}
	}
	return TokenCount{}
}

func toInt64(v any) (int64, bool) {
	switch n := v.(type) {
	case int:
		return int64(n), true
	case int8:
		return int64(n), true
	case int16:
		return int64(n), true
	case int32:
		return int64(n), true
	case int64:
		return n, true
	case uint:
		return int64(n), true
	case uint8:
		return int64(n), true
	case uint16:
		return int64(n), true
	case uint32:
		return int64(n), true
	case uint64:
		return int64(n), true
	case float32:
		return int64(n), true
	case float64:
		return int64(n), true
	default:
		return 0, false
	}
}
