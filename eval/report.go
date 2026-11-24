package eval

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"
)

// Report is the result of one evaluation run. Its Samples are sorted by ID
// and its Categories by category name, so the same inputs always produce the
// same ordering. Volatile data (wall clock, durations, call and cache
// counters, per-sample retry and cache-hit flags) is kept in [Meta] and in
// the sample's Attempts/CacheHit fields, all of which [Report.CanonicalBytes]
// excludes.
type Report struct {
	SchemaVersion int                 `json:"schema_version"`
	Complete      bool                `json:"complete"`
	Config        ConfigSummary       `json:"config"`
	Overall       Aggregate           `json:"overall"`
	Categories    []CategoryAggregate `json:"categories"`
	Samples       []SampleResult      `json:"samples"`
	Meta          Meta                `json:"meta"`
}

// Evaluation pairs the report with the per-sample results in input order.
// Callers that need to align results with the sample file should use Results;
// callers that need a comparable artifact should use Report.
type Evaluation struct {
	// Results are in the same order as the input samples.
	Results []SampleResult
	// Report is the aggregated, ID-sorted report.
	Report *Report
}

// buildReport aggregates results and sorts the report views.
func buildReport(samples []Sample, results []SampleResult, cfg Config, targetName string) *Report {
	complete := true
	for _, res := range results {
		switch res.Status {
		case StatusCanceled, StatusNotRun:
			complete = false
		}
	}

	byCategory := make(map[string][]SampleResult)
	for _, res := range results {
		byCategory[res.Category] = append(byCategory[res.Category], res)
	}
	categoryNames := make([]string, 0, len(byCategory))
	for name := range byCategory {
		categoryNames = append(categoryNames, name)
	}
	sort.Strings(categoryNames)

	categories := make([]CategoryAggregate, 0, len(categoryNames))
	for _, name := range categoryNames {
		categories = append(categories, CategoryAggregate{
			Category:  name,
			Aggregate: aggregate(byCategory[name]),
		})
	}

	sorted := make([]SampleResult, len(results))
	copy(sorted, results)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].ID < sorted[j].ID })

	return &Report{
		SchemaVersion: SchemaVersion,
		Complete:      complete,
		Config: ConfigSummary{
			Target:        targetName,
			ModelID:       cfg.ModelID,
			PromptVersion: cfg.PromptVersion,
			Concurrency:   cfg.Concurrency,
			Timeout:       cfg.Timeout.String(),
			MaxRetries:    cfg.MaxRetries,
			RetryBackoff:  cfg.RetryBackoff.String(),
			CacheEnabled:  cfg.Cache != nil,
		},
		Overall:    aggregate(results),
		Categories: categories,
		Samples:    sorted,
	}
}

// aggregate summarizes results. Means only include samples that produced a
// value: failed and canceled samples never enter metric means, and unknown
// usage never counts as zero.
func aggregate(results []SampleResult) Aggregate {
	var a Aggregate
	var c collector
	failures := make(map[ErrorClass]int)

	for _, res := range results {
		a.Samples++
		switch res.Status {
		case StatusSucceeded:
			a.Succeeded++
			c.addSucceeded(res)
		case StatusFailed:
			a.Failed++
			failures[res.ErrorClass]++
		case StatusCanceled:
			a.Canceled++
		case StatusNotRun:
			a.NotRun++
		}
	}

	a.ExactMatchRate = mean(c.exact)
	a.TokenF1 = mean(c.f1)
	a.KeyPointCoverage = mean(c.coverage)
	a.RetrievalRecall = mean(c.recall)
	a.RetrievalPrecision = mean(c.precision)
	a.PromptTokens = mean(c.prompt)
	a.CompletionTokens = mean(c.completion)
	a.TotalTokens = mean(c.total)
	a.UsageReported = c.usageReported
	a.UsageUnknown = c.usageUnknown

	a.Failures = make([]FailureCount, 0, len(failures))
	for class, count := range failures {
		a.Failures = append(a.Failures, FailureCount{Class: string(class), Count: count})
	}
	sort.Slice(a.Failures, func(i, j int) bool { return a.Failures[i].Class < a.Failures[j].Class })
	return a
}

// collector accumulates the metric samples that contribute to means.
type collector struct {
	exact, f1, coverage         []float64
	recall, precision           []float64
	prompt, completion, total   []float64
	usageReported, usageUnknown int
}

func (c *collector) addSucceeded(res SampleResult) {
	if res.Metrics == nil {
		return
	}
	if res.Metrics.ExactMatch {
		c.exact = append(c.exact, 1)
	} else {
		c.exact = append(c.exact, 0)
	}
	c.f1 = append(c.f1, res.Metrics.TokenF1)
	c.coverage = append(c.coverage, res.Metrics.KeyPointCoverage)
	if res.Metrics.Retrieval != nil {
		c.recall = append(c.recall, res.Metrics.Retrieval.Recall)
		c.precision = append(c.precision, res.Metrics.Retrieval.Precision)
	}
	known := c.addUsage(res.Usage)
	if known {
		c.usageReported++
	} else {
		c.usageUnknown++
	}
}

// addUsage records only the known token counts and reports whether any were
// known.
func (c *collector) addUsage(u Usage) bool {
	if u.PromptTokens.Known {
		c.prompt = append(c.prompt, float64(u.PromptTokens.Value))
	}
	if u.CompletionTokens.Known {
		c.completion = append(c.completion, float64(u.CompletionTokens.Value))
	}
	if u.TotalTokens.Known {
		c.total = append(c.total, float64(u.TotalTokens.Value))
	}
	return u.PromptTokens.Known || u.CompletionTokens.Known || u.TotalTokens.Known
}

// canonicalSample is a SampleResult without volatile run information.
type canonicalSample struct {
	ID              string     `json:"id"`
	Category        string     `json:"category"`
	Question        string     `json:"question"`
	ReferenceAnswer string     `json:"reference_answer"`
	Status          Status     `json:"status"`
	ErrorClass      ErrorClass `json:"error_class,omitempty"`
	Answer          string     `json:"answer,omitempty"`
	StopReason      string     `json:"stop_reason,omitempty"`
	Usage           Usage      `json:"usage"`
	Metrics         *Metrics   `json:"metrics"`
}

type canonicalReport struct {
	SchemaVersion int                 `json:"schema_version"`
	Complete      bool                `json:"complete"`
	Config        ConfigSummary       `json:"config"`
	Overall       Aggregate           `json:"overall"`
	Categories    []CategoryAggregate `json:"categories"`
	Samples       []canonicalSample   `json:"samples"`
}

// CanonicalBytes returns a deterministic JSON encoding of the report's
// results. Wall-clock data, durations, retry counts, cache-hit flags, and
// free-form error messages are omitted, so two runs over the same samples with
// the same configuration produce identical bytes, whether or not the response
// cache was used. The error class is retained, since it is a stable outcome.
func (r *Report) CanonicalBytes() ([]byte, error) {
	if r == nil {
		return nil, fmt.Errorf("eval: nil report")
	}
	cfg := r.Config
	// Whether a response cache was enabled does not change what is measured;
	// leaving it out lets cached and uncached runs compare as equal.
	cfg.CacheEnabled = false
	cr := canonicalReport{
		SchemaVersion: r.SchemaVersion,
		Complete:      r.Complete,
		Config:        cfg,
		Overall:       r.Overall,
		Categories:    r.Categories,
		Samples:       make([]canonicalSample, len(r.Samples)),
	}
	for i, s := range r.Samples {
		cr.Samples[i] = canonicalSample{
			ID:              s.ID,
			Category:        s.Category,
			Question:        s.Question,
			ReferenceAnswer: s.ReferenceAnswer,
			Status:          s.Status,
			ErrorClass:      s.ErrorClass,
			Answer:          s.Answer,
			StopReason:      s.StopReason,
			Usage:           s.Usage,
			Metrics:         s.Metrics,
		}
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(cr); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// WriteJSON renders the full report, including volatile metadata, as indented
// JSON. Use [Report.CanonicalBytes] when the output must be comparable.
func (r *Report) WriteJSON(w io.Writer) error {
	if r == nil {
		return fmt.Errorf("eval: nil report")
	}
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	return enc.Encode(r)
}

// WriteText renders the report as a fixed-format text table. It includes the
// volatile metadata, so it is meant for humans; use [Report.CanonicalBytes]
// for machine comparison.
func (r *Report) WriteText(w io.Writer) error {
	if r == nil {
		return fmt.Errorf("eval: nil report")
	}
	b := new(strings.Builder)
	fmt.Fprintf(b, "Evaluation report (schema %d)\n", r.SchemaVersion)
	fmt.Fprintf(b, "target=%s model=%s prompt_version=%s\n",
		display(r.Config.Target), display(r.Config.ModelID), display(r.Config.PromptVersion))
	fmt.Fprintf(b, "complete=%t started_at=%s duration_ms=%d\n",
		r.Complete, r.Meta.StartedAt.UTC().Format(time.RFC3339), r.Meta.DurationMS)
	fmt.Fprintf(b, "concurrency=%d timeout=%s max_retries=%d retry_backoff=%s cache=%t\n",
		r.Config.Concurrency, r.Config.Timeout, r.Config.MaxRetries, r.Config.RetryBackoff, r.Config.CacheEnabled)
	fmt.Fprintf(b, "target_calls=%d cache_hits=%d cache_misses=%d\n",
		r.Meta.TargetCalls, r.Meta.CacheHits, r.Meta.CacheMisses)

	writeAggregate(b, "OVERALL", r.Overall)
	for _, c := range r.Categories {
		writeAggregate(b, "CATEGORY "+c.Category, c.Aggregate)
	}

	fmt.Fprintln(b, "SAMPLES")
	fmt.Fprintln(b, "id\tcategory\tstatus\texact\ttoken_f1\tkey_points\tretrieval_recall\tretrieval_precision\tattempts\tcache_hit")
	for _, s := range r.Samples {
		exact, f1, coverage, recall, precision := "not_scored", "-", "-", "-", "-"
		if s.Metrics != nil {
			exact = fmt.Sprintf("%t", s.Metrics.ExactMatch)
			f1 = formatFloat(s.Metrics.TokenF1)
			coverage = formatFloat(s.Metrics.KeyPointCoverage)
			if s.Metrics.Retrieval != nil {
				recall = formatFloat(s.Metrics.Retrieval.Recall)
				precision = formatFloat(s.Metrics.Retrieval.Precision)
			}
		}
		fmt.Fprintf(b, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%t\n",
			s.ID, s.Category, s.Status, exact, f1, coverage,
			recall, precision, s.Attempts, s.CacheHit)
	}
	_, err := io.WriteString(w, b.String())
	return err
}

func writeAggregate(b *strings.Builder, label string, a Aggregate) {
	fmt.Fprintf(b, "%s samples=%d succeeded=%d failed=%d canceled=%d not_run=%d\n",
		label, a.Samples, a.Succeeded, a.Failed, a.Canceled, a.NotRun)
	fmt.Fprintf(b, "%s exact_match_rate=%s token_f1=%s key_point_coverage=%s\n",
		label, formatOpt(a.ExactMatchRate), formatOpt(a.TokenF1), formatOpt(a.KeyPointCoverage))
	fmt.Fprintf(b, "%s retrieval_recall=%s retrieval_precision=%s\n",
		label, formatOpt(a.RetrievalRecall), formatOpt(a.RetrievalPrecision))
	fmt.Fprintf(b, "%s mean_prompt_tokens=%s mean_completion_tokens=%s mean_total_tokens=%s usage_reported=%d usage_unknown=%d\n",
		label, formatOpt(a.PromptTokens), formatOpt(a.CompletionTokens), formatOpt(a.TotalTokens),
		a.UsageReported, a.UsageUnknown)
	if len(a.Failures) > 0 {
		parts := make([]string, len(a.Failures))
		for i, f := range a.Failures {
			parts[i] = fmt.Sprintf("%s=%d", f.Class, f.Count)
		}
		fmt.Fprintf(b, "%s failures %s\n", label, strings.Join(parts, " "))
	}
}

func formatOpt(v *float64) string {
	if v == nil {
		return "unknown"
	}
	return formatFloat(*v)
}

func formatFloat(v float64) string {
	return fmt.Sprintf("%.4f", v)
}

func display(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

// MetricDelta is the change in one metric between two reports. Values are
// nil when either side is unknown.
type MetricDelta struct {
	Left    *float64 `json:"left"`
	Right   *float64 `json:"right"`
	Delta   *float64 `json:"delta"`
	Changed bool     `json:"changed"`
}

// CategoryDelta is a per-category comparison.
type CategoryDelta struct {
	Category    string      `json:"category"`
	ExactMatch  MetricDelta `json:"exact_match_rate"`
	TokenF1     MetricDelta `json:"token_f1"`
	KeyPoints   MetricDelta `json:"key_point_coverage"`
	FailedLeft  int         `json:"failed_left"`
	FailedRight int         `json:"failed_right"`
}

// SampleDelta records a sample whose status or exact-match outcome changed
// between two runs. Exact-match fields are meaningful only for samples that
// were scored; a failure that became a success shows up through the status
// fields. Callers decide which direction is a regression.
type SampleDelta struct {
	ID              string      `json:"id"`
	Category        string      `json:"category"`
	LeftStatus      Status      `json:"left_status"`
	RightStatus     Status      `json:"right_status"`
	LeftExactMatch  bool        `json:"left_exact_match"`
	RightExactMatch bool        `json:"right_exact_match"`
	TokenF1         MetricDelta `json:"token_f1"`
	KeyPoints       MetricDelta `json:"key_point_coverage"`
}

// Comparison is the difference between two reports. Comparable is false when
// the reports were produced with different configurations, in which case
// Reasons explains why.
type Comparison struct {
	Comparable      bool            `json:"comparable"`
	Reasons         []string        `json:"reasons,omitempty"`
	Left            ConfigSummary   `json:"left"`
	Right           ConfigSummary   `json:"right"`
	LeftComplete    bool            `json:"left_complete"`
	RightComplete   bool            `json:"right_complete"`
	ExactMatch      MetricDelta     `json:"exact_match_rate"`
	TokenF1         MetricDelta     `json:"token_f1"`
	KeyPoints       MetricDelta     `json:"key_point_coverage"`
	RetrievalRecall MetricDelta     `json:"retrieval_recall"`
	FailedLeft      int             `json:"failed_left"`
	FailedRight     int             `json:"failed_right"`
	Categories      []CategoryDelta `json:"categories"`
	// Changed lists samples whose status or exact-match outcome changed,
	// sorted by ID.
	Changed []SampleDelta `json:"changed_samples"`
}

// Compare reports how right differs from left. The two reports are expected
// to cover the same sample IDs; a differing sample set is reported as not
// comparable. Configuration fields that affect scoring are compared too.
func Compare(left, right *Report) *Comparison {
	c := &Comparison{}
	if left == nil || right == nil {
		c.Reasons = append(c.Reasons, "nil report")
		return c
	}
	c.Left = left.Config
	c.Right = right.Config
	c.LeftComplete = left.Complete
	c.RightComplete = right.Complete
	c.Comparable = true
	c.checkComparable(left, right)

	c.ExactMatch = metricDelta(left.Overall.ExactMatchRate, right.Overall.ExactMatchRate)
	c.TokenF1 = metricDelta(left.Overall.TokenF1, right.Overall.TokenF1)
	c.KeyPoints = metricDelta(left.Overall.KeyPointCoverage, right.Overall.KeyPointCoverage)
	c.RetrievalRecall = metricDelta(left.Overall.RetrievalRecall, right.Overall.RetrievalRecall)
	c.FailedLeft = left.Overall.Failed
	c.FailedRight = right.Overall.Failed

	c.Categories = compareCategories(left, right)
	c.Changed = changedSamples(left, right)
	return c
}

// checkComparable records the reasons two reports cannot be compared. A
// differing prompt version is allowed: comparing prompts is the point of the
// tool, and the cache key already keeps their responses apart.
func (c *Comparison) checkComparable(left, right *Report) {
	if left.SchemaVersion != right.SchemaVersion {
		c.markIncomparable(fmt.Sprintf("schema version %d != %d", left.SchemaVersion, right.SchemaVersion))
	}
	if left.Config.Target != right.Config.Target {
		c.markIncomparable("target differs")
	}
	if left.Config.ModelID != right.Config.ModelID {
		c.markIncomparable("model ID differs")
	}
	if !equalStrings(sampleIDs(left), sampleIDs(right)) {
		c.markIncomparable("sample sets differ")
	}
}

func (c *Comparison) markIncomparable(reason string) {
	c.Comparable = false
	c.Reasons = append(c.Reasons, reason)
}

func compareCategories(left, right *Report) []CategoryDelta {
	leftCats := categoryMap(left)
	rightCats := categoryMap(right)
	names := make([]string, 0, len(leftCats)+len(rightCats))
	for name := range leftCats {
		names = append(names, name)
	}
	for name := range rightCats {
		if _, ok := leftCats[name]; !ok {
			names = append(names, name)
		}
	}
	sort.Strings(names)

	deltas := make([]CategoryDelta, 0, len(names))
	for _, name := range names {
		l := leftCats[name]
		r := rightCats[name]
		deltas = append(deltas, CategoryDelta{
			Category:    name,
			ExactMatch:  metricDelta(l.ExactMatchRate, r.ExactMatchRate),
			TokenF1:     metricDelta(l.TokenF1, r.TokenF1),
			KeyPoints:   metricDelta(l.KeyPointCoverage, r.KeyPointCoverage),
			FailedLeft:  l.Failed,
			FailedRight: r.Failed,
		})
	}
	return deltas
}

func metricDelta(left, right *float64) MetricDelta {
	d := MetricDelta{Left: left, Right: right}
	if left != nil && right != nil {
		v := round4(*right - *left)
		d.Delta = &v
		d.Changed = *left != *right
	}
	return d
}

func sampleIDs(r *Report) []string {
	ids := make([]string, len(r.Samples))
	for i, s := range r.Samples {
		ids[i] = s.ID
	}
	return ids
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func categoryMap(r *Report) map[string]Aggregate {
	m := make(map[string]Aggregate, len(r.Categories))
	for _, c := range r.Categories {
		m[c.Category] = c.Aggregate
	}
	return m
}

func changedSamples(left, right *Report) []SampleDelta {
	rightByID := make(map[string]SampleResult, len(right.Samples))
	for _, s := range right.Samples {
		rightByID[s.ID] = s
	}
	var changed []SampleDelta
	for _, l := range left.Samples {
		r, ok := rightByID[l.ID]
		if !ok {
			continue
		}
		// A sample changed if it ended in a different status (for example a
		// failure became a success), or if both runs scored it and the
		// exact-match outcome differs.
		statusChanged := l.Status != r.Status
		exactChanged := l.Metrics != nil && r.Metrics != nil && l.Metrics.ExactMatch != r.Metrics.ExactMatch
		if !statusChanged && !exactChanged {
			continue
		}
		d := SampleDelta{
			ID:          l.ID,
			Category:    l.Category,
			LeftStatus:  l.Status,
			RightStatus: r.Status,
		}
		if l.Metrics != nil {
			d.LeftExactMatch = l.Metrics.ExactMatch
		}
		if r.Metrics != nil {
			d.RightExactMatch = r.Metrics.ExactMatch
		}
		if l.Metrics != nil && r.Metrics != nil {
			d.TokenF1 = metricDelta(floatPtr(l.Metrics.TokenF1), floatPtr(r.Metrics.TokenF1))
			d.KeyPoints = metricDelta(floatPtr(l.Metrics.KeyPointCoverage), floatPtr(r.Metrics.KeyPointCoverage))
		}
		changed = append(changed, d)
	}
	sort.Slice(changed, func(i, j int) bool { return changed[i].ID < changed[j].ID })
	return changed
}

func floatPtr(v float64) *float64 { return &v }

// WriteJSON renders the comparison as indented JSON.
func (c *Comparison) WriteJSON(w io.Writer) error {
	if c == nil {
		return fmt.Errorf("eval: nil comparison")
	}
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	return enc.Encode(c)
}

// WriteText renders the comparison as a fixed-format text summary: overall
// deltas first, then per-category deltas, then the samples that changed.
func (c *Comparison) WriteText(w io.Writer) error {
	if c == nil {
		return fmt.Errorf("eval: nil comparison")
	}
	b := new(strings.Builder)
	fmt.Fprintf(b, "Comparison comparable=%t", c.Comparable)
	if len(c.Reasons) > 0 {
		fmt.Fprintf(b, " reasons=%s", strings.Join(c.Reasons, "; "))
	}
	fmt.Fprintln(b)
	fmt.Fprintf(b, "left  complete=%t model=%s prompt_version=%s\n",
		c.LeftComplete, display(c.Left.ModelID), display(c.Left.PromptVersion))
	fmt.Fprintf(b, "right complete=%t model=%s prompt_version=%s\n",
		c.RightComplete, display(c.Right.ModelID), display(c.Right.PromptVersion))
	fmt.Fprintf(b, "overall exact_match_rate %s\n", describeDelta(c.ExactMatch))
	fmt.Fprintf(b, "overall token_f1 %s\n", describeDelta(c.TokenF1))
	fmt.Fprintf(b, "overall key_point_coverage %s\n", describeDelta(c.KeyPoints))
	fmt.Fprintf(b, "overall retrieval_recall %s\n", describeDelta(c.RetrievalRecall))
	fmt.Fprintf(b, "overall failed %d -> %d\n", c.FailedLeft, c.FailedRight)
	for _, cat := range c.Categories {
		fmt.Fprintf(b, "category %s exact_match_rate %s token_f1 %s key_point_coverage %s failed %d -> %d\n",
			cat.Category, describeDelta(cat.ExactMatch), describeDelta(cat.TokenF1),
			describeDelta(cat.KeyPoints), cat.FailedLeft, cat.FailedRight)
	}
	if len(c.Changed) > 0 {
		fmt.Fprintln(b, "CHANGED SAMPLES")
		for _, s := range c.Changed {
			fmt.Fprintf(b, "%s\t%s\t%s\t%s\texact_match=%t->%t\ttoken_f1 %s\tkey_points %s\n",
				s.ID, s.Category, s.LeftStatus, s.RightStatus,
				s.LeftExactMatch, s.RightExactMatch, describeDelta(s.TokenF1), describeDelta(s.KeyPoints))
		}
	}
	_, err := io.WriteString(w, b.String())
	return err
}

func describeDelta(d MetricDelta) string {
	if d.Left == nil || d.Right == nil {
		return "unknown"
	}
	sign := ""
	if d.Changed {
		if *d.Delta > 0 {
			sign = "+"
		}
	}
	return fmt.Sprintf("%s -> %s (%s%.4f)", formatFloat(*d.Left), formatFloat(*d.Right), sign, *d.Delta)
}
