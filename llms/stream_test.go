package llms_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/langchaingo/llms"
)

// recorder collects the events of one stream.
type recorder struct {
	mu     sync.Mutex
	events []llms.StreamEvent
	// stopAfter, when >= 0, makes the recorder return an error once that
	// many events have been delivered.
	stopAfter int
	stopErr   error
}

func newRecorder() *recorder { return &recorder{stopAfter: -1} }

func (r *recorder) emit(_ context.Context, ev llms.StreamEvent) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, ev)
	if r.stopAfter >= 0 && len(r.events) > r.stopAfter {
		return r.stopErr
	}
	return nil
}

func (r *recorder) ofKind(kind llms.StreamEventKind) []llms.StreamEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []llms.StreamEvent
	for _, ev := range r.events {
		if ev.Kind == kind {
			out = append(out, ev)
		}
	}
	return out
}

func (r *recorder) text() string {
	var b strings.Builder
	for _, ev := range r.ofKind(llms.StreamEventText) {
		b.WriteString(ev.Text)
	}
	return b.String()
}

func (r *recorder) summary(t *testing.T) llms.StreamSummary {
	t.Helper()
	ends := r.ofKind(llms.StreamEventEnd)
	if len(ends) != 1 {
		t.Fatalf("got %d end events, want exactly 1", len(ends))
	}
	if ends[0].Summary == nil {
		t.Fatal("end event carries no summary")
	}
	return *ends[0].Summary
}

func feed(t *testing.T, s *llms.Stream, ctx context.Context, deltas ...llms.ToolCallDelta) {
	t.Helper()
	for _, d := range deltas {
		if err := s.AddToolCall(ctx, d); err != nil {
			t.Fatalf("AddToolCall(%+v): %v", d, err)
		}
	}
}

func finishAndClose(t *testing.T, s *llms.Stream, ctx context.Context, reason llms.StreamStopReason) llms.StreamSummary {
	t.Helper()
	if err := s.Finish(ctx); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return s.Close(ctx, reason, nil)
}

// splitEvery splits s at every byte boundary in turn and feeds the fragments.
func splitEvery(t *testing.T, body string, step int) []llms.ToolCallDelta {
	t.Helper()
	var deltas []llms.ToolCallDelta
	for i := 0; i < len(body); i += step {
		end := i + step
		if end > len(body) {
			end = len(body)
		}
		deltas = append(deltas, llms.ToolCallDelta{
			Index:     0,
			ID:        pick(i == 0, "call_1", ""),
			Name:      pick(i == 0, "get_weather", ""),
			Arguments: body[i:end],
		})
	}
	return deltas
}

func pick(cond bool, a, b string) string {
	if cond {
		return a
	}
	return b
}

func TestStreamPureText(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	rec := newRecorder()
	s := llms.NewStream("test", rec.emit)

	for _, frag := range []string{"Hello", "", " ", "world", ""} {
		if err := s.Text(ctx, frag); err != nil {
			t.Fatal(err)
		}
	}
	sum := finishAndClose(t, s, ctx, llms.StreamStopReasonComplete)

	if got, want := rec.text(), "Hello world"; got != want {
		t.Fatalf("text = %q, want %q", got, want)
	}
	if got := len(rec.ofKind(llms.StreamEventText)); got != 3 {
		t.Fatalf("text events = %d, want 3 (empty fragments dropped)", got)
	}
	if sum.TextLength != len("Hello world") {
		t.Fatalf("summary text length = %d", sum.TextLength)
	}
	if sum.StopReason != llms.StreamStopReasonComplete {
		t.Fatalf("stop reason = %s", sum.StopReason)
	}
	if len(sum.Calls) != 0 {
		t.Fatalf("unexpected calls: %+v", sum.Calls)
	}
}

func TestStreamNewCallOpening(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	rec := newRecorder()
	s := llms.NewStream("test", rec.emit)

	// The first fragment opens the call with identifier and name. Later
	// fragments must not open a second call even though they carry no
	// identifier, as the OpenAI wire format does.
	feed(t, s, ctx,
		llms.ToolCallDelta{Index: 0, ID: "call_a", Name: "lookup", Arguments: `{"q":`},
		llms.ToolCallDelta{Index: 0, Arguments: `"x"`},
		llms.ToolCallDelta{Index: 0, Arguments: `}`},
	)
	finishAndClose(t, s, ctx, llms.StreamStopReasonComplete)

	starts := rec.ofKind(llms.StreamEventToolCallStart)
	if len(starts) != 1 {
		t.Fatalf("start events = %d, want 1", len(starts))
	}
	if starts[0].ToolCallID != "call_a" || starts[0].ToolName != "lookup" {
		t.Fatalf("start = %+v", starts[0])
	}
	done := rec.ofKind(llms.StreamEventToolCallDone)
	if len(done) != 1 || done[0].Status.State != llms.ToolCallStateComplete {
		t.Fatalf("done = %+v", done)
	}
	if got := done[0].ToolCall.FunctionCall.Arguments; got != `{"q":"x"}` {
		t.Fatalf("arguments = %q", got)
	}
}

func TestStreamArgumentsAggregateAtEverySplit(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	body := `{"city":"東京","note":"quote \" and slash \\ and é"}`

	// Sanity: one-shot delivery is the reference.
	collect := func(deltas []llms.ToolCallDelta) (string, llms.StreamSummary) {
		rec := newRecorder()
		s := llms.NewStream("test", rec.emit)
		feed(t, s, ctx, deltas...)
		sum := finishAndClose(t, s, ctx, llms.StreamStopReasonComplete)
		return s.CompleteCalls()[0].FunctionCall.Arguments, sum
	}
	want, wantSum := collect([]llms.ToolCallDelta{{
		Index: 0, ID: "c", Name: "f", Arguments: body,
	}})
	if want != body {
		t.Fatalf("one-shot arguments = %q, want %q", want, body)
	}
	if wantSum.Calls[0].State != llms.ToolCallStateComplete {
		t.Fatalf("one-shot state = %s", wantSum.Calls[0].State)
	}

	for _, step := range []int{1, 2, 3, 5, 7, 11, len(body)} {
		step := step
		t.Run(fmt.Sprintf("step_%d", step), func(t *testing.T) {
			t.Parallel()
			got, sum := collect(splitEvery(t, body, step))
			if got != body {
				t.Fatalf("step %d aggregated = %q, want %q", step, got, body)
			}
			if sum.Calls[0].State != llms.ToolCallStateComplete {
				t.Fatalf("step %d state = %s (%s)", step, sum.Calls[0].State, sum.Calls[0].Error)
			}
			if len(sum.Calls) != 1 {
				t.Fatalf("step %d calls = %d, want 1", step, len(sum.Calls))
			}
		})
	}
}

func TestStreamInterleavedCalls(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	rec := newRecorder()
	s := llms.NewStream("test", rec.emit)

	argsA := `{"a":1}`
	argsB := `{"b":2}`
	feed(t, s, ctx,
		llms.ToolCallDelta{Index: 0, ID: "call_a", Name: "fa", Arguments: argsA[:2]},
		llms.ToolCallDelta{Index: 1, ID: "call_b", Name: "fb", Arguments: argsB[:2]},
		llms.ToolCallDelta{Index: 0, Arguments: argsA[2:4]},
		llms.ToolCallDelta{Index: 1, Arguments: argsB[2:4]},
		llms.ToolCallDelta{Index: 0, Arguments: argsA[4:]},
		llms.ToolCallDelta{Index: 1, Arguments: argsB[4:]},
	)
	finishAndClose(t, s, ctx, llms.StreamStopReasonComplete)

	calls := s.CompleteCalls()
	if len(calls) != 2 {
		t.Fatalf("complete calls = %d, want 2", len(calls))
	}
	if calls[0].ID != "call_a" || calls[0].FunctionCall.Arguments != argsA {
		t.Fatalf("call 0 = %+v", calls[0])
	}
	if calls[1].ID != "call_b" || calls[1].FunctionCall.Arguments != argsB {
		t.Fatalf("call 1 = %+v", calls[1])
	}

	// Every delta event must be attributed to the call it extended.
	for _, ev := range rec.ofKind(llms.StreamEventToolCallDelta) {
		switch ev.Index {
		case 0:
			if ev.ToolCallID != "call_a" {
				t.Fatalf("index 0 delta routed to %q", ev.ToolCallID)
			}
		case 1:
			if ev.ToolCallID != "call_b" {
				t.Fatalf("index 1 delta routed to %q", ev.ToolCallID)
			}
		}
	}
}

func TestStreamIdentityMissingOrLate(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	cases := []struct {
		name   string
		deltas []llms.ToolCallDelta
	}{
		{
			name: "position only",
			deltas: []llms.ToolCallDelta{
				{Index: 0, Name: "solo", Arguments: `{"x":`},
				{Index: 0, Arguments: `1}`},
			},
		},
		{
			name: "name arrives after arguments",
			deltas: []llms.ToolCallDelta{
				{Index: 0, ID: "c1", Arguments: `{"x":`},
				{Index: 0, Arguments: `1}`, Name: "late"},
			},
		},
		{
			name: "id arrives after the opening fragment",
			deltas: []llms.ToolCallDelta{
				{Index: 0, Name: "f", Arguments: `{"x":`},
				{Index: 0, ID: "c1", Arguments: `1}`},
			},
		},
		{
			name: "partial fields split",
			deltas: []llms.ToolCallDelta{
				{Index: 0, ID: "c1"},
				{Index: 0, Name: "f"},
				{Index: 0, Arguments: `{"x":1}`},
			},
		},
		{
			name: "id with slipped position",
			deltas: []llms.ToolCallDelta{
				{Index: 0, ID: "c1", Name: "f", Arguments: `{"x":`},
				{Index: 3, ID: "c1", Arguments: `1}`},
			},
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			s := llms.NewStream("test", nil)
			feed(t, s, ctx, tc.deltas...)
			if err := s.Finish(ctx); err != nil {
				t.Fatal(err)
			}
			calls := s.Assembled()
			if len(calls) != 1 {
				t.Fatalf("assembled calls = %d, want 1 (%+v)", len(calls), calls)
			}
			if calls[0].Status.State != llms.ToolCallStateComplete {
				t.Fatalf("state = %s (%s)", calls[0].Status.State, calls[0].Status.Error)
			}
			if got := calls[0].ToolCall.FunctionCall.Arguments; got != `{"x":1}` {
				t.Fatalf("arguments = %q", got)
			}
			if calls[0].ToolCall.FunctionCall.Name == "" {
				t.Fatal("call has no name")
			}
		})
	}
}

func TestStreamDuplicateAndCumulative(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	cases := []struct {
		name    string
		deltas  []llms.ToolCallDelta
		wantArg string
	}{
		{
			name: "repeated fragment",
			deltas: []llms.ToolCallDelta{
				{Index: 0, ID: "c", Name: "f", Arguments: `{"a":`},
				{Index: 0, Arguments: `{"a":`},
				{Index: 0, Arguments: `1}`},
			},
			wantArg: `{"a":1}`,
		},
		{
			name: "cumulative resend",
			deltas: []llms.ToolCallDelta{
				{Index: 0, ID: "c", Name: "f", Arguments: `{"a":`},
				{Index: 0, Arguments: `{"a":1}`},
			},
			wantArg: `{"a":1}`,
		},
		{
			name: "cumulative resend growing",
			deltas: []llms.ToolCallDelta{
				{Index: 0, ID: "c", Name: "f", Arguments: `{"a"`},
				{Index: 0, Arguments: `{"a":1`},
				{Index: 0, Arguments: `{"a":1}`},
			},
			wantArg: `{"a":1}`,
		},
		{
			name: "full call delivered twice",
			deltas: []llms.ToolCallDelta{
				{Index: 0, ID: "c", Name: "f", Arguments: `{"a":1}`},
				{Index: 0, ID: "c", Name: "f", Arguments: `{"a":1}`},
			},
			wantArg: `{"a":1}`,
		},
		{
			name: "legitimate repeated bytes",
			deltas: []llms.ToolCallDelta{
				{Index: 0, ID: "c", Name: "f", Arguments: `{"a":"`},
				{Index: 0, Arguments: `aa`},
				{Index: 0, Arguments: `aa`},
				{Index: 0, Arguments: `"}`},
			},
			wantArg: `{"a":"aaaa"}`,
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			s := llms.NewStream("test", nil)
			feed(t, s, ctx, tc.deltas...)
			if err := s.Finish(ctx); err != nil {
				t.Fatal(err)
			}
			if got := len(s.CompleteCalls()); got != 1 {
				t.Fatalf("complete calls = %d, want 1", got)
			}
			if got := s.CompleteCalls()[0].FunctionCall.Arguments; got != tc.wantArg {
				t.Fatalf("arguments = %q, want %q", got, tc.wantArg)
			}
		})
	}
}

func TestStreamExecutability(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	cases := []struct {
		name   string
		deltas []llms.ToolCallDelta
		state  llms.ToolCallState
	}{
		{
			name:   "valid object",
			deltas: []llms.ToolCallDelta{{Index: 0, ID: "c", Name: "f", Arguments: `{"a":1}`}},
			state:  llms.ToolCallStateComplete,
		},
		{
			name:   "truncated",
			deltas: []llms.ToolCallDelta{{Index: 0, ID: "c", Name: "f", Arguments: `{"a":`}},
			state:  llms.ToolCallStateIncomplete,
		},
		{
			name:   "not JSON",
			deltas: []llms.ToolCallDelta{{Index: 0, ID: "c", Name: "f", Arguments: `not json`}},
			state:  llms.ToolCallStateIncomplete,
		},
		{
			name:   "scalar",
			deltas: []llms.ToolCallDelta{{Index: 0, ID: "c", Name: "f", Arguments: `42`}},
			state:  llms.ToolCallStateIncomplete,
		},
		{
			name:   "no name",
			deltas: []llms.ToolCallDelta{{Index: 0, ID: "c", Arguments: `{"a":1}`}},
			state:  llms.ToolCallStateIncomplete,
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			rec := newRecorder()
			s := llms.NewStream("test", rec.emit)
			feed(t, s, ctx, tc.deltas...)
			sum := finishAndClose(t, s, ctx, llms.StreamStopReasonComplete)
			if len(sum.Calls) != 1 {
				t.Fatalf("calls = %d", len(sum.Calls))
			}
			if sum.Calls[0].State != tc.state {
				t.Fatalf("state = %s, want %s", sum.Calls[0].State, tc.state)
			}
			if tc.state == llms.ToolCallStateComplete && len(s.CompleteCalls()) != 1 {
				t.Fatal("complete call not executable")
			}
			if tc.state != llms.ToolCallStateComplete {
				if len(s.CompleteCalls()) != 0 {
					t.Fatal("incomplete call reported as executable")
				}
				if len(s.IncompleteCalls()) != 1 {
					t.Fatal("incomplete call not reported")
				}
			}
			// Exactly one Done event, always.
			if got := len(rec.ofKind(llms.StreamEventToolCallDone)); got != 1 {
				t.Fatalf("done events = %d, want 1", got)
			}
		})
	}
}

func TestStreamLifecycleAndSummary(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	rec := newRecorder()
	s := llms.NewStream("test", rec.emit)

	if err := s.Text(ctx, "prefix"); err != nil {
		t.Fatal(err)
	}
	feed(t, s, ctx,
		llms.ToolCallDelta{Index: 0, ID: "c1", Name: "f", Arguments: `{"a":`},
		llms.ToolCallDelta{Index: 0, Arguments: `1}`},
		llms.ToolCallDelta{Index: 1, ID: "c2", Name: "g", Arguments: `{"b":`},
	)
	sum := finishAndClose(t, s, ctx, llms.StreamStopReasonComplete)

	if len(sum.Calls) != 2 {
		t.Fatalf("summary calls = %d", len(sum.Calls))
	}
	if sum.Calls[0].State != llms.ToolCallStateComplete {
		t.Fatalf("call 0 = %s", sum.Calls[0].State)
	}
	if sum.Calls[1].State != llms.ToolCallStateIncomplete {
		t.Fatalf("call 1 = %s", sum.Calls[1].State)
	}
	if sum.Calls[1].ErrorOffset == 0 || sum.Calls[1].Error == "" {
		t.Fatalf("incomplete call lacks failure detail: %+v", sum.Calls[1])
	}

	// Summary must not contain argument bodies.
	if strings.Contains(sum.String(), `"a"`) || strings.Contains(fmt.Sprint(sum.Calls), `"a"`) {
		t.Fatalf("summary exposes arguments: %s", sum)
	}

	// Lifecycle ordering: start before delta before done, per call.
	seen := map[int][]llms.StreamEventKind{}
	for _, ev := range rec.events {
		switch ev.Kind {
		case llms.StreamEventToolCallStart, llms.StreamEventToolCallDelta, llms.StreamEventToolCallDone:
			seen[ev.Index] = append(seen[ev.Index], ev.Kind)
		}
	}
	for idx, kinds := range seen {
		if kinds[0] != llms.StreamEventToolCallStart {
			t.Fatalf("call %d first event = %s", idx, kinds[0])
		}
		last := kinds[len(kinds)-1]
		if last != llms.StreamEventToolCallDone {
			t.Fatalf("call %d last event = %s", idx, last)
		}
	}
}

func TestStreamSummaryDeterministic(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	run := func() string {
		s := llms.NewStream("test", nil)
		feed(t, s, ctx,
			llms.ToolCallDelta{Index: 1, ID: "b", Name: "fb", Arguments: `{"b":2}`},
			llms.ToolCallDelta{Index: 0, ID: "a", Name: "fa", Arguments: `{"a":`},
			llms.ToolCallDelta{Index: 0, Arguments: `1}`},
			llms.ToolCallDelta{Index: 2, ID: "c", Name: "fc", Arguments: `{"c":`},
		)
		sum := s.Close(ctx, llms.StreamStopReasonComplete, nil)
		return fmt.Sprintf("%s|%+v", sum.String(), sum.Calls)
	}
	first := run()
	for i := 0; i < 10; i++ {
		if got := run(); got != first {
			t.Fatalf("run %d differs:\n%s\n%s", i, first, got)
		}
	}
}

func TestStreamEndDeliveredOnce(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	rec := newRecorder()
	s := llms.NewStream("test", rec.emit)
	s.Close(ctx, llms.StreamStopReasonComplete, nil)
	s.Close(ctx, llms.StreamStopReasonCanceled, errors.New("late"))
	s.Close(ctx, llms.StreamStopReasonUpstreamError, errors.New("later"))
	if got := len(rec.ofKind(llms.StreamEventEnd)); got != 1 {
		t.Fatalf("end events = %d, want 1", got)
	}
	if sum := rec.summary(t); sum.StopReason != llms.StreamStopReasonComplete {
		t.Fatalf("first reason lost: %s", sum.StopReason)
	}
}

func TestStreamConsumerErrorStopsDelivery(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	sentinel := errors.New("stop")
	rec := newRecorder()
	rec.stopAfter = 1 // allow the first event, then fail
	rec.stopErr = sentinel
	s := llms.NewStream("test", rec.emit)

	if err := s.Text(ctx, "one"); err != nil {
		t.Fatalf("first text: %v", err)
	}
	if err := s.Text(ctx, "two"); !errors.Is(err, sentinel) {
		t.Fatalf("second text err = %v, want sentinel", err)
	}
	// Later deliveries are suppressed; the partial result is still there.
	if err := s.Text(ctx, "three"); err != nil {
		t.Fatalf("delivery after error should be suppressed, got %v", err)
	}
	if got := rec.text(); got != "onetwo" {
		t.Fatalf("delivered text = %q", got)
	}
	// The rejected fragment was delivered once but never accumulated, so
	// the partial result matches what the consumer accepted.
	if s.AccumulatedText() != "one" {
		t.Fatalf("accumulated text = %q", s.AccumulatedText())
	}
}

func TestStreamCancelledContextDeliversNothingFurther(t *testing.T) {
	t.Parallel()
	rec := newRecorder()
	s := llms.NewStream("test", rec.emit)
	ctx, cancel := context.WithCancel(context.Background())
	if err := s.Text(ctx, "before"); err != nil {
		t.Fatal(err)
	}
	cancel()
	if err := s.Text(ctx, "after"); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	sum := s.Close(context.Background(), llms.StreamStopReasonCanceled, ctx.Err())
	if got := len(rec.ofKind(llms.StreamEventEnd)); got != 1 {
		t.Fatalf("end events = %d", got)
	}
	if sum.StopReason != llms.StreamStopReasonCanceled {
		t.Fatalf("reason = %s", sum.StopReason)
	}
	if rec.text() != "before" {
		t.Fatalf("text = %q", rec.text())
	}
}

func TestStreamConcurrentStreamsIsolated(t *testing.T) {
	t.Parallel()
	const streams = 32
	var wg sync.WaitGroup
	errs := make(chan error, streams)
	for i := 0; i < streams; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx := context.Background()
			name := fmt.Sprintf("f%d", i)
			body := fmt.Sprintf(`{"i":%d}`, i)
			s := llms.NewStream("test", nil)
			for j := 0; j < len(body); j++ {
				d := llms.ToolCallDelta{Index: 0, Arguments: body[j : j+1]}
				if j == 0 {
					d.ID = fmt.Sprintf("c%d", i)
					d.Name = name
				}
				if err := s.AddToolCall(ctx, d); err != nil {
					errs <- err
					return
				}
			}
			if err := s.Finish(ctx); err != nil {
				errs <- err
				return
			}
			calls := s.CompleteCalls()
			if len(calls) != 1 {
				errs <- fmt.Errorf("stream %d: %d calls", i, len(calls))
				return
			}
			if calls[0].ID != fmt.Sprintf("c%d", i) || calls[0].FunctionCall.Arguments != body {
				errs <- fmt.Errorf("stream %d: cross-talk %+v", i, calls[0])
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestStreamUnknownFragmentWarnsAndContinues(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	rec := newRecorder()
	s := llms.NewStream("test", rec.emit)
	feed(t, s, ctx,
		llms.ToolCallDelta{Index: 5},
		llms.ToolCallDelta{Index: 0, ID: "c", Name: "f", Arguments: `{"a":1}`},
	)
	sum := finishAndClose(t, s, ctx, llms.StreamStopReasonComplete)
	if got := len(rec.ofKind(llms.StreamEventWarning)); got != 1 {
		t.Fatalf("warnings = %d, want 1", got)
	}
	if len(sum.Calls) != 1 || sum.Calls[0].State != llms.ToolCallStateComplete {
		t.Fatalf("stream did not continue after warning: %+v", sum.Calls)
	}
}

// ensure the recorder's event type matches the public contract.
var _ llms.StreamEventFunc = newRecorder().emit

func TestMergeArgumentsDirect(t *testing.T) {
	t.Parallel()
	// The merge policy is observable through public behavior; this pins
	// the boundary that duplicate detection never corrupts valid JSON.
	body := `{"s":"aabb","n":[1,1,2],"o":{"k":"k"}}`
	for step := 1; step <= len(body); step++ {
		s := llms.NewStream("", nil)
		for i := 0; i < len(body); i += step {
			end := i + step
			if end > len(body) {
				end = len(body)
			}
			d := llms.ToolCallDelta{Index: 0, Arguments: body[i:end]}
			if i == 0 {
				d.ID, d.Name = "c", "f"
			}
			if err := s.AddToolCall(context.Background(), d); err != nil {
				t.Fatal(err)
			}
		}
		if err := s.Finish(context.Background()); err != nil {
			t.Fatal(err)
		}
		var got any
		if err := json.Unmarshal([]byte(s.CompleteCalls()[0].FunctionCall.Arguments), &got); err != nil {
			t.Fatalf("step %d: invalid assembled JSON: %v", step, err)
		}
	}
}

func TestStreamEndSuppressedAfterConsumerError(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	sentinel := errors.New("stop")
	rec := newRecorder()
	rec.stopAfter = 0
	rec.stopErr = sentinel
	s := llms.NewStream("test", rec.emit)
	if err := s.Text(ctx, "one"); !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want sentinel", err)
	}
	sum := s.Close(context.Background(), llms.StreamStopReasonConsumerError, sentinel)
	if got := len(rec.ofKind(llms.StreamEventEnd)); got != 0 {
		t.Fatalf("end delivered after consumer error: %d", got)
	}
	if sum.StopReason != llms.StreamStopReasonConsumerError {
		t.Fatalf("reason = %s", sum.StopReason)
	}
	if len(sum.IncompleteCalls()) != 0 {
		// no calls were opened
		t.Fatalf("unexpected incomplete calls: %+v", sum.Calls)
	}
}

func TestStreamCompletedCallDeliveredTwiceCompletesOnce(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	rec := newRecorder()
	s := llms.NewStream("test", rec.emit)
	full := `{"a":1}`
	feed(t, s, ctx,
		llms.ToolCallDelta{Index: 0, ID: "c", Name: "f", Arguments: full},
		llms.ToolCallDelta{Index: 0, ID: "c", Name: "f", Arguments: full},
	)
	// Force the call terminal, then deliver the whole thing again.
	if err := s.Finish(ctx); err != nil {
		t.Fatal(err)
	}
	feed(t, s, ctx,
		llms.ToolCallDelta{Index: 0, ID: "c", Name: "f", Arguments: full},
	)
	sum := s.Close(ctx, llms.StreamStopReasonComplete, nil)

	if got := len(rec.ofKind(llms.StreamEventToolCallDone)); got != 1 {
		t.Fatalf("done events = %d, want exactly 1", got)
	}
	if len(sum.Calls) != 1 || sum.Calls[0].State != llms.ToolCallStateComplete {
		t.Fatalf("summary = %+v", sum.Calls)
	}
	if got := s.CompleteCalls()[0].FunctionCall.Arguments; got != full {
		t.Fatalf("arguments = %q", got)
	}
}

func TestStreamPositionOnlyCallGetsStableID(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := llms.NewStream("test", nil)
	feed(t, s, ctx,
		llms.ToolCallDelta{Index: 2, Name: "solo", Arguments: `{"x":1}`},
	)
	sum := s.Close(ctx, llms.StreamStopReasonComplete, nil)
	if len(sum.Calls) != 1 {
		t.Fatalf("calls = %d", len(sum.Calls))
	}
	if sum.Calls[0].ID != "call_2" {
		t.Fatalf("synthesized id = %q, want call_2", sum.Calls[0].ID)
	}
	got := s.CompleteCalls()[0]
	if got.ID != "call_2" {
		t.Fatalf("assembled id = %q, want call_2", got.ID)
	}
	if got.ID != sum.Calls[0].ID {
		t.Fatal("status and assembled call disagree on identity")
	}
}

func TestStreamEventFuncFromText(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	var got []string
	h := llms.StreamEventFuncFromText(func(_ context.Context, chunk []byte) error {
		got = append(got, string(chunk))
		return nil
	})
	if err := h(ctx, llms.StreamEvent{Kind: llms.StreamEventText, Text: "a"}); err != nil {
		t.Fatal(err)
	}
	if err := h(ctx, llms.StreamEvent{Kind: llms.StreamEventToolCallStart, ToolName: "x"}); err != nil {
		t.Fatal(err)
	}
	if err := h(ctx, llms.StreamEvent{Kind: llms.StreamEventText, Text: "b"}); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("text events = %v, want [a b]", got)
	}
}

func TestWithStreamingHandlerOption(t *testing.T) {
	t.Parallel()
	var opts llms.CallOptions
	h := func(context.Context, llms.StreamEvent) error { return nil }
	llms.WithStreamingHandler(h)(&opts)
	if opts.StreamingEventFunc == nil {
		t.Fatal("StreamingEventFunc not set")
	}
	// The historical text callback is untouched by the new option.
	if opts.StreamingFunc != nil {
		t.Fatal("StreamingFunc unexpectedly set")
	}
}
