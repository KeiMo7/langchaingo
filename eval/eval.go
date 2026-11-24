package eval

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
)

// SchemaVersion is the version of the report and sample file schema.
const SchemaVersion = 1

// Status describes how a sample ended. Every sample in a report has exactly
// one of these statuses.
type Status string

const (
	// StatusSucceeded means the target returned an answer and the sample was
	// scored.
	StatusSucceeded Status = "succeeded"
	// StatusFailed means the target returned an error after retries, and the
	// error is recorded with its class.
	StatusFailed Status = "failed"
	// StatusCanceled means the sample had started when the run was canceled.
	StatusCanceled Status = "canceled"
	// StatusNotRun means the run was canceled before the sample was dispatched.
	StatusNotRun Status = "not_run"
)

// ErrorClass is a stable classification of a sample failure.
type ErrorClass string

const (
	ErrorClassTimeout             ErrorClass = "timeout"
	ErrorClassCanceled            ErrorClass = "canceled"
	ErrorClassRateLimit           ErrorClass = "rate_limit"
	ErrorClassProviderUnavailable ErrorClass = "provider_unavailable"
	ErrorClassAuthentication      ErrorClass = "authentication"
	ErrorClassInvalidRequest      ErrorClass = "invalid_request"
	ErrorClassContentFilter       ErrorClass = "content_filter"
	ErrorClassTokenLimit          ErrorClass = "token_limit"
	ErrorClassOther               ErrorClass = "other"
)

// retryable reports whether a failure class can plausibly succeed on a retry.
// Permanent errors (authentication, invalid request, content filter) are not
// retried.
func (c ErrorClass) retryable() bool {
	switch c {
	case ErrorClassTimeout, ErrorClassRateLimit, ErrorClassProviderUnavailable:
		return true
	default:
		return false
	}
}

// Sample is one evaluation case.
type Sample struct {
	ID                 string   `json:"id"`
	Question           string   `json:"question"`
	ReferenceAnswer    string   `json:"reference_answer"`
	ReferenceMaterials []string `json:"reference_materials,omitempty"`
	KeyPoints          []string `json:"key_points,omitempty"`
	Category           string   `json:"category,omitempty"`
}

// EffectiveCategory returns the category a sample is grouped under.
func (s Sample) EffectiveCategory() string {
	if s.Category == "" {
		return Uncategorized
	}
	return s.Category
}

// EffectiveKeyPoints returns the configured key points, defaulting to the
// reference answer as the single required point.
func (s Sample) EffectiveKeyPoints() []string {
	if len(s.KeyPoints) > 0 {
		return s.KeyPoints
	}
	if s.ReferenceAnswer == "" {
		return nil
	}
	return []string{s.ReferenceAnswer}
}

// TokenCount is a token counter that distinguishes "not reported" from zero.
// In JSON it is either a number or null, so an unknown count is visibly
// absent rather than zero.
type TokenCount struct {
	Value int64
	Known bool
}

// MarshalJSON encodes a known count as a number and an unknown count as null.
func (t TokenCount) MarshalJSON() ([]byte, error) {
	if !t.Known {
		return []byte("null"), nil
	}
	return json.Marshal(t.Value)
}

// UnmarshalJSON decodes a number as a known count and null as unknown.
func (t *TokenCount) UnmarshalJSON(data []byte) error {
	if string(bytes.TrimSpace(data)) == "null" {
		*t = TokenCount{}
		return nil
	}
	var v int64
	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}
	*t = TokenCount{Value: v, Known: true}
	return nil
}

// Usage holds token usage reported by a target. Fields the provider did not
// report stay unknown; unknown usage is excluded from report means rather
// than counted as zero.
type Usage struct {
	PromptTokens     TokenCount `json:"prompt_tokens"`
	CompletionTokens TokenCount `json:"completion_tokens"`
	TotalTokens      TokenCount `json:"total_tokens"`
}

func tokenCount(v int64) TokenCount { return TokenCount{Value: v, Known: true} }

// Output is what a Target produces for one sample.
type Output struct {
	// Answer is the textual answer under evaluation.
	Answer string
	// StopReason is the model stop reason, when the target can report one.
	StopReason string
	// Usage is the token usage, with unreported fields left unknown.
	Usage Usage
	// Documents are the documents the target retrieved, when it retrieved any.
	Documents []schema.Document
	// Retrieved reports whether retrieval ran for this sample. It separates
	// "retriever returned nothing" from "no retriever in use".
	Retrieved bool
}

func (o *Output) clone() *Output {
	if o == nil {
		return nil
	}
	cp := *o
	cp.Documents = append([]schema.Document(nil), o.Documents...)
	return &cp
}

// Target is the system under evaluation. Implementations must honor context
// cancellation and deadlines. They are called concurrently, so they must be
// safe for concurrent use.
type Target interface {
	// Run produces an output for a single sample.
	Run(ctx context.Context, sample Sample) (*Output, error)
	// Name identifies the target in the report.
	Name() string
}

// FuncTarget adapts a function to Target.
type FuncTarget struct {
	TargetName string
	Fn         func(ctx context.Context, sample Sample) (*Output, error)
}

var _ Target = FuncTarget{}

// Run calls the underlying function.
func (t FuncTarget) Run(ctx context.Context, sample Sample) (*Output, error) {
	if t.Fn == nil {
		return nil, errors.New("eval: FuncTarget has nil Fn")
	}
	return t.Fn(ctx, sample)
}

// Name returns the configured name or "func".
func (t FuncTarget) Name() string {
	if t.TargetName == "" {
		return "func"
	}
	return t.TargetName
}

// DefaultConfig returns conservative defaults: two concurrent samples, a
// one-minute per-sample deadline, and at most two retries of transient
// errors.
func DefaultConfig() Config {
	return Config{
		Concurrency:  2,
		Timeout:      60 * time.Second,
		MaxRetries:   2,
		RetryBackoff: 250 * time.Millisecond,
	}
}

// Config controls a run.
type Config struct {
	// Concurrency bounds the number of samples evaluated at once. Values <= 0
	// use the default of 2.
	Concurrency int
	// Timeout bounds each sample, including any retries. Values <= 0 use 60s.
	Timeout time.Duration
	// MaxRetries is how many times a transient failure is retried. Zero
	// disables retries.
	MaxRetries int
	// RetryBackoff is the wait before the first retry; it doubles per try.
	// Values <= 0 use 250ms.
	RetryBackoff time.Duration
	// Cache, when non-nil, avoids calling the target for a sample whose
	// content, model ID, prompt version, and salt were already seen.
	Cache Cache
	// CacheSalt folds additional configuration, such as retrieval parameters,
	// into the cache key. It is required to be stable for a fixed configuration.
	CacheSalt string
	// ModelID names the model under evaluation and is part of the cache key.
	ModelID string
	// PromptVersion identifies the prompt under evaluation and is part of the
	// cache key. Changing the prompt requires changing this value.
	PromptVersion string
	// Now supplies the wall clock for report metadata; nil uses time.Now.
	Now func() time.Time
}

func (c Config) withDefaults() Config {
	if c.Concurrency <= 0 {
		c.Concurrency = 2
	}
	if c.Timeout <= 0 {
		c.Timeout = 60 * time.Second
	}
	if c.MaxRetries < 0 {
		c.MaxRetries = 0
	}
	if c.RetryBackoff <= 0 {
		c.RetryBackoff = 250 * time.Millisecond
	}
	if c.Now == nil {
		c.Now = time.Now
	}
	return c
}

func (c Config) validate() error {
	if c.Cache != nil {
		if c.ModelID == "" {
			return errors.New("eval: Config.ModelID is required when a cache is enabled")
		}
		if c.PromptVersion == "" {
			return errors.New("eval: Config.PromptVersion is required when a cache is enabled")
		}
	}
	return nil
}

// SampleResult is the outcome of one sample.
type SampleResult struct {
	ID              string     `json:"id"`
	Category        string     `json:"category"`
	Question        string     `json:"question"`
	ReferenceAnswer string     `json:"reference_answer"`
	Status          Status     `json:"status"`
	ErrorClass      ErrorClass `json:"error_class,omitempty"`
	ErrorMessage    string     `json:"error_message,omitempty"`
	Answer          string     `json:"answer,omitempty"`
	StopReason      string     `json:"stop_reason,omitempty"`
	Usage           Usage      `json:"usage"`
	// Metrics is nil for samples that were not scored (failed, canceled, or
	// not run), distinguishing "not scored" from "scored zero".
	Metrics *Metrics `json:"metrics"`
	// Attempts counts target calls for this sample, including retries. A
	// cache hit makes zero calls.
	Attempts int `json:"attempts"`
	// CacheHit is true when the output came from the cache. Attempts and
	// CacheHit are volatile and excluded from [Report.CanonicalBytes].
	CacheHit bool `json:"cache_hit"`
}

// Metrics are the per-sample scores. All answer metrics are defined for every
// successful sample; retrieval metrics are nil when retrieval did not run or
// the sample has no reference materials.
type Metrics struct {
	ExactMatch       bool              `json:"exact_match"`
	TokenF1          float64           `json:"token_f1"`
	KeyPointCoverage float64           `json:"key_point_coverage"`
	Retrieval        *RetrievalMetrics `json:"retrieval,omitempty"`
}

// RetrievalMetrics score retrieved documents against reference materials.
type RetrievalMetrics struct {
	Recall         float64 `json:"recall"`
	Precision      float64 `json:"precision"`
	MatchedRefs    int     `json:"matched_reference_materials"`
	ReferenceCount int     `json:"reference_materials"`
	RetrievedCount int     `json:"retrieved_documents"`
}

// Meta is the volatile section of a report. It is excluded from
// [Report.CanonicalBytes] so that runs over the same inputs compare equal.
type Meta struct {
	StartedAt   time.Time `json:"started_at"`
	DurationMS  int64     `json:"duration_ms"`
	TargetCalls int       `json:"target_calls"`
	CacheHits   int       `json:"cache_hits"`
	CacheMisses int       `json:"cache_misses"`
}

// ConfigSummary records the inputs that affect scoring, so that two reports
// can be checked for comparability.
type ConfigSummary struct {
	Target        string `json:"target"`
	ModelID       string `json:"model_id,omitempty"`
	PromptVersion string `json:"prompt_version,omitempty"`
	Concurrency   int    `json:"concurrency"`
	Timeout       string `json:"timeout"`
	MaxRetries    int    `json:"max_retries"`
	RetryBackoff  string `json:"retry_backoff"`
	CacheEnabled  bool   `json:"cache_enabled"`
}

// FailureCount counts failures of one class.
type FailureCount struct {
	Class string `json:"class"`
	Count int    `json:"count"`
}

// Aggregate summarizes a set of sample results.
type Aggregate struct {
	Samples   int `json:"samples"`
	Succeeded int `json:"succeeded"`
	Failed    int `json:"failed"`
	Canceled  int `json:"canceled"`
	NotRun    int `json:"not_run"`

	// Metric means are nil when no sample contributed a value; they are not
	// reported as zero.
	ExactMatchRate     *float64 `json:"exact_match_rate"`
	TokenF1            *float64 `json:"token_f1"`
	KeyPointCoverage   *float64 `json:"key_point_coverage"`
	RetrievalRecall    *float64 `json:"retrieval_recall"`
	RetrievalPrecision *float64 `json:"retrieval_precision"`

	PromptTokens     *float64 `json:"mean_prompt_tokens"`
	CompletionTokens *float64 `json:"mean_completion_tokens"`
	TotalTokens      *float64 `json:"mean_total_tokens"`
	UsageReported    int      `json:"usage_reported_samples"`
	UsageUnknown     int      `json:"usage_unknown_samples"`

	Failures []FailureCount `json:"failure_classes"`
}

// CategoryAggregate is an Aggregate for one category.
type CategoryAggregate struct {
	Category string `json:"category"`
	Aggregate
}

// Evaluate runs every sample against target with bounded concurrency, then
// aggregates the results into a report. Sample failures are isolated; only a
// configuration problem or an unusable sample list is returned as an error.
// If ctx is canceled, no new samples are dispatched, in-flight samples are
// awaited, and the returned report is marked incomplete.
//
// Evaluation.Results is in input order; Evaluation.Report is sorted by ID and
// is suitable for comparison across runs.
func Evaluate(ctx context.Context, samples []Sample, target Target, cfg Config) (*Evaluation, error) {
	if target == nil {
		return nil, errors.New("eval: nil target")
	}
	if err := validateSamples(samples); err != nil {
		return nil, err
	}
	cfg = cfg.withDefaults()
	if err := cfg.validate(); err != nil {
		return nil, err
	}

	start := cfg.Now()
	r := &runner{
		cfg:     cfg,
		target:  target,
		results: make([]SampleResult, len(samples)),
	}
	for i, s := range samples {
		r.results[i] = SampleResult{
			ID:              s.ID,
			Category:        s.EffectiveCategory(),
			Question:        s.Question,
			ReferenceAnswer: s.ReferenceAnswer,
			Status:          StatusNotRun,
		}
	}
	r.run(ctx, samples)

	report := buildReport(samples, r.results, cfg, target.Name())
	report.Meta = Meta{
		StartedAt:   start,
		DurationMS:  cfg.Now().Sub(start).Milliseconds(),
		TargetCalls: int(r.calls.Load()),
		CacheHits:   int(r.cacheHits.Load()),
		CacheMisses: int(r.cacheMisses.Load()),
	}
	return &Evaluation{Results: r.results, Report: report}, nil
}

// runner holds the mutable state of one evaluation run.
type runner struct {
	cfg         Config
	target      Target
	results     []SampleResult
	calls       atomic.Int64
	cacheHits   atomic.Int64
	cacheMisses atomic.Int64
}

func (r *runner) run(ctx context.Context, samples []Sample) {
	jobs := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < r.cfg.Concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r.worker(ctx, jobs, samples)
		}()
	}
	r.dispatch(ctx, jobs)
	wg.Wait()
}

// dispatch sends sample indexes to the workers until all samples are sent or
// the context is canceled. It never blocks indefinitely on a canceled run.
func (r *runner) dispatch(ctx context.Context, jobs chan<- int) {
	defer close(jobs)
	for i := range r.results {
		if ctx.Err() != nil {
			return
		}
		select {
		case jobs <- i:
		case <-ctx.Done():
			return
		}
	}
}

func (r *runner) worker(ctx context.Context, jobs <-chan int, samples []Sample) {
	for {
		if ctx.Err() != nil {
			return
		}
		select {
		case <-ctx.Done():
			return
		case i, ok := <-jobs:
			if !ok {
				return
			}
			r.results[i] = r.evaluateSample(ctx, samples[i])
		}
	}
}

func (r *runner) evaluateSample(parent context.Context, sample Sample) SampleResult {
	result := SampleResult{
		ID:              sample.ID,
		Category:        sample.EffectiveCategory(),
		Question:        sample.Question,
		ReferenceAnswer: sample.ReferenceAnswer,
		Status:          StatusCanceled,
		ErrorClass:      ErrorClassCanceled,
	}
	if parent.Err() != nil {
		result.ErrorMessage = parent.Err().Error()
		return result
	}

	key, cached := r.lookupCache(parent, sample)
	if cached != nil {
		result.CacheHit = true
		r.fillFromOutput(&result, sample, cached)
		return result
	}

	ctx, cancel := context.WithTimeout(parent, r.cfg.Timeout)
	defer cancel()
	out, class, lastErr := r.runWithRetries(ctx, sample, &result, key)

	if lastErr == nil {
		if out == nil {
			out = &Output{}
		}
		r.fillFromOutput(&result, sample, out)
		return result
	}

	// A sample interrupted because the whole run was canceled is reported as
	// canceled rather than failed; it is not a property of the system under
	// test. A per-sample timeout stays a failure.
	if parent.Err() != nil || class == ErrorClassCanceled {
		result.Status = StatusCanceled
		result.ErrorClass = ErrorClassCanceled
	} else {
		result.Status = StatusFailed
		result.ErrorClass = class
	}
	result.ErrorMessage = lastErr.Error()
	return result
}

// lookupCache returns the cache key and a cached output when caching is on and
// the sample is present.
func (r *runner) lookupCache(ctx context.Context, sample Sample) (string, *Output) {
	if r.cfg.Cache == nil {
		return "", nil
	}
	key := cacheKey(r.cfg, sample)
	if out, ok := r.cfg.Cache.Get(ctx, key); ok {
		r.cacheHits.Add(1)
		return key, out
	}
	r.cacheMisses.Add(1)
	return key, nil
}

// runWithRetries calls the target under the retry policy. It returns the
// output on success, or the last error with its class. Successful outputs are
// written to the cache when one is configured.
func (r *runner) runWithRetries(
	ctx context.Context, sample Sample, result *SampleResult, cacheKey string,
) (*Output, ErrorClass, error) {
	var lastErr error
	var class ErrorClass
	for attempt := 0; attempt <= r.cfg.MaxRetries; attempt++ {
		if attempt > 0 {
			if err := sleepContext(ctx, backoff(r.cfg.RetryBackoff, attempt)); err != nil {
				return nil, classifyError(err), err
			}
		}
		result.Attempts++
		r.calls.Add(1)
		out, err := r.target.Run(ctx, sample)
		if err == nil {
			if out == nil {
				out = &Output{}
			}
			if r.cfg.Cache != nil && cacheKey != "" {
				r.cfg.Cache.Put(ctx, cacheKey, out.clone())
			}
			return out, "", nil
		}
		lastErr = err
		class = classifyError(err)
		if !class.retryable() || ctx.Err() != nil {
			break
		}
	}
	return nil, class, lastErr
}

// fillFromOutput marks a result as succeeded and scores its output.
func (r *runner) fillFromOutput(result *SampleResult, sample Sample, out *Output) {
	result.Status = StatusSucceeded
	result.ErrorClass = ""
	result.ErrorMessage = ""
	result.Answer = out.Answer
	result.StopReason = out.StopReason
	result.Usage = out.Usage
	metrics := Score(sample, out)
	result.Metrics = &metrics
}

func backoff(base time.Duration, attempt int) time.Duration {
	d := base
	for i := 1; i < attempt; i++ {
		d *= 2
	}
	return d
}

func sleepContext(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// classifyError maps an error to a stable class without inspecting messages
// it does not recognize.
func classifyError(err error) ErrorClass {
	if err == nil {
		return ""
	}
	if errors.Is(err, context.Canceled) {
		return ErrorClassCanceled
	}
	if errors.Is(err, context.DeadlineExceeded) || llms.IsTimeoutError(err) {
		return ErrorClassTimeout
	}
	switch {
	case llms.IsAuthenticationError(err):
		return ErrorClassAuthentication
	case llms.IsRateLimitError(err):
		return ErrorClassRateLimit
	case llms.IsQuotaExceededError(err):
		return ErrorClassRateLimit
	case llms.IsInvalidRequestError(err):
		return ErrorClassInvalidRequest
	case llms.IsContentFilterError(err):
		return ErrorClassContentFilter
	case llms.IsTokenLimitError(err):
		return ErrorClassTokenLimit
	case llms.IsProviderUnavailableError(err):
		return ErrorClassProviderUnavailable
	case llms.IsCanceledError(err):
		return ErrorClassCanceled
	}
	return ErrorClassOther
}

// validateSamples rejects lists that cannot produce an unambiguous report.
func validateSamples(samples []Sample) error {
	if len(samples) == 0 {
		return errors.New("eval: no samples")
	}
	seen := make(map[string]struct{}, len(samples))
	for i, s := range samples {
		if s.ID == "" {
			return fmt.Errorf("eval: sample %d: empty ID", i+1)
		}
		if s.Question == "" {
			return fmt.Errorf("eval: sample %q: empty Question", s.ID)
		}
		if s.ReferenceAnswer == "" {
			return fmt.Errorf("eval: sample %q: empty ReferenceAnswer", s.ID)
		}
		if _, ok := seen[s.ID]; ok {
			return fmt.Errorf("eval: duplicate sample ID %q", s.ID)
		}
		seen[s.ID] = struct{}{}
	}
	return nil
}
