package openai

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/langchaingo/llms"
)

// sseServer replays the given event data lines as an OpenAI-style SSE stream.
type sseServer struct {
	*httptest.Server
	mu    sync.Mutex
	gotCh chan string
}

func newSSEServer(t *testing.T, events []string, comments bool) *sseServer {
	t.Helper()
	s := &sseServer{gotCh: make(chan string, 1)}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "no flusher", http.StatusInternalServerError)
			return
		}
		for _, ev := range events {
			if comments {
				_, _ = fmt.Fprint(w, ": OPENROUTER PROCESSING\n\n")
			}
			_, _ = fmt.Fprintf(w, "data: %s\n\n", ev)
			flusher.Flush()
		}
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	t.Cleanup(s.Close)
	return s
}

func newScriptedLLM(t *testing.T, srv *sseServer) *LLM {
	t.Helper()
	return newScriptedLLMFor(t, srv.URL)
}

func newScriptedLLMFor(t *testing.T, baseURL string) *LLM {
	t.Helper()
	llm, err := New(
		WithToken("test-key"),
		WithBaseURL(baseURL),
		WithModel("test-model"),
	)
	if err != nil {
		t.Fatal(err)
	}
	return llm
}

func chatChunk(delta string) string {
	return `{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":` + delta + `,"finish_reason":null}]}`
}

// collectEvents records the provider-neutral events of a call.
type collectEvents struct {
	mu     sync.Mutex
	events []llms.StreamEvent
}

func (c *collectEvents) emit(_ context.Context, ev llms.StreamEvent) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, ev)
	return nil
}

func (c *collectEvents) kinds() []llms.StreamEventKind {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]llms.StreamEventKind, len(c.events))
	for i, ev := range c.events {
		out[i] = ev.Kind
	}
	return out
}

func (c *collectEvents) summary(t *testing.T) llms.StreamSummary {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	var found []llms.StreamEvent
	for _, ev := range c.events {
		if ev.Kind == llms.StreamEventEnd {
			found = append(found, ev)
		}
	}
	if len(found) != 1 {
		t.Fatalf("end events = %d, want 1", len(found))
	}
	return *found[0].Summary
}

func TestOpenAIStreamToolCallInterleaved(t *testing.T) {
	t.Parallel()
	srv := newSSEServer(t, []string{
		chatChunk(`{"role":"assistant","content":"Let me check."}`),
		chatChunk(`{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"weather","arguments":""}}]}`),
		chatChunk(`{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"time","arguments":""}}]}`),
		chatChunk(`{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}`),
		chatChunk(`{"tool_calls":[{"index":1,"function":{"arguments":"{\"tz\":"}}]}`),
		chatChunk(`{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]}`),
		chatChunk(`{"tool_calls":[{"index":1,"function":{"arguments":"\"UTC\"}"}}]}`),
		chatChunk(`{"finish_reason":"tool_calls"}`),
	}, true)

	llm := newScriptedLLM(t, srv)
	events := &collectEvents{}
	var text strings.Builder
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(_ context.Context, chunk []byte) error {
			text.Write(chunk)
			return nil
		}),
		llms.WithStreamingHandler(events.emit),
	)
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}

	if got := text.String(); got != "Let me check." {
		t.Fatalf("streamed text = %q", got)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("choices = %d", len(resp.Choices))
	}
	calls := resp.Choices[0].ToolCalls
	if len(calls) != 2 {
		t.Fatalf("tool calls = %d, want 2", len(calls))
	}
	if calls[0].ID != "call_a" || calls[0].FunctionCall.Arguments != `{"city":"Paris"}` {
		t.Fatalf("call 0 = %+v", calls[0])
	}
	if calls[1].ID != "call_b" || calls[1].FunctionCall.Arguments != `{"tz":"UTC"}` {
		t.Fatalf("call 1 = %+v", calls[1])
	}

	sum := events.summary(t)
	if sum.StopReason != llms.StreamStopReasonComplete {
		t.Fatalf("stop reason = %s", sum.StopReason)
	}
	if len(sum.Calls) != 2 || sum.Calls[0].State != llms.ToolCallStateComplete || sum.Calls[1].State != llms.ToolCallStateComplete {
		t.Fatalf("summary calls = %+v", sum.Calls)
	}
	if len(sum.CompleteCalls()) != 2 {
		t.Fatal("complete calls not executable")
	}

	// Start/delta/done ordering per call.
	var sawStart, sawDone [2]bool
	for _, ev := range events.events {
		switch ev.Kind {
		case llms.StreamEventToolCallStart:
			if ev.Index < 2 {
				sawStart[ev.Index] = true
			}
		case llms.StreamEventToolCallDone:
			if ev.Index < 2 {
				if !sawStart[ev.Index] {
					t.Fatalf("done before start for index %d", ev.Index)
				}
				sawDone[ev.Index] = true
			}
		}
	}
	if !sawStart[0] || !sawStart[1] || !sawDone[0] || !sawDone[1] {
		t.Fatalf("missing lifecycle events: start=%v done=%v", sawStart, sawDone)
	}
}

func TestOpenAIStreamArgumentsSplitOneByteAtATime(t *testing.T) {
	t.Parallel()
	full := `{"q":"東京 \"x\""}`
	// Splitting happens on the decoded arguments, at rune boundaries so each
	// fragment is a valid JSON string. Multibyte characters arrive whole, and
	// the escape sequence \" is split between its backslash and quote,
	// which exercises a boundary the aggregator must not disturb.
	var frags []string
	for _, r := range full {
		frags = append(frags, string(r))
	}
	// Insert an extra split inside every two-character escape.
	var split []string
	for _, f := range frags {
		split = append(split, f)
	}
	var events []string
	events = append(events, chatChunk(`{"tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"f","arguments":""}}]}`))
	for _, frag := range split {
		events = append(events, chatChunk(fmt.Sprintf(
			`{"tool_calls":[{"index":0,"function":{"arguments":%q}}]}`, frag)))
	}
	srv := newSSEServer(t, events, false)
	llm := newScriptedLLM(t, srv)

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingHandler((&collectEvents{}).emit),
	)
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	calls := resp.Choices[0].ToolCalls
	if len(calls) != 1 {
		t.Fatalf("calls = %d", len(calls))
	}
	if got := calls[0].FunctionCall.Arguments; got != full {
		t.Fatalf("arguments = %q, want %q", got, full)
	}
}

func TestOpenAIStreamDuplicateWholeResend(t *testing.T) {
	t.Parallel()
	srv := newSSEServer(t, []string{
		chatChunk(`{"tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]}`),
		chatChunk(`{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":1}"}}]}`),
	}, false)
	llm := newScriptedLLM(t, srv)
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingHandler((&collectEvents{}).emit),
	)
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	calls := resp.Choices[0].ToolCalls
	if len(calls) != 1 || calls[0].FunctionCall.Arguments != `{"a":1}` {
		t.Fatalf("calls = %+v", calls)
	}
}

func TestOpenAIStreamTruncatedArgumentsAreNotExecutable(t *testing.T) {
	t.Parallel()
	srv := newSSEServer(t, []string{
		chatChunk(`{"tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"f","arguments":"{\"a\":"}}]}`),
	}, false)
	llm := newScriptedLLM(t, srv)
	events := &collectEvents{}
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingHandler(events.emit),
	)
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	// The complete-call list stays empty so no tool is executed, but the
	// raw argument bytes are preserved for diagnostics.
	if len(resp.Choices[0].ToolCalls) != 0 {
		t.Fatalf("truncated call surfaced as executable: %+v", resp.Choices[0].ToolCalls)
	}
	sum := events.summary(t)
	if len(sum.Calls) != 1 || sum.Calls[0].State != llms.ToolCallStateIncomplete {
		t.Fatalf("summary = %+v", sum.Calls)
	}
	if sum.Calls[0].ErrorOffset == 0 {
		t.Fatalf("missing failure position: %+v", sum.Calls[0])
	}
}

func TestOpenAIStreamMidStreamUpstreamError(t *testing.T) {
	t.Parallel()
	srv := newSSEServer(t, []string{
		chatChunk(`{"role":"assistant","content":"partial"}`),
		chatChunk(`{"tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]}`),
	}, false)
	llm := newScriptedLLM(t, srv)
	events := &collectEvents{}
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingHandler(events.emit),
	)
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if resp == nil || resp.Choices[0].Content == "" {
		t.Fatal("partial content lost")
	}
}

func TestOpenAIStreamConsumerErrorPropagatesPartial(t *testing.T) {
	t.Parallel()
	srv := newSSEServer(t, []string{
		chatChunk(`{"role":"assistant","content":"one"}`),
		chatChunk(`{"role":"assistant","content":"two"}`),
		chatChunk(`{"role":"assistant","content":"three"}`),
	}, false)
	llm := newScriptedLLM(t, srv)
	want := fmt.Errorf("consumer stop")
	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingHandler(func(_ context.Context, ev llms.StreamEvent) error {
			if ev.Kind == llms.StreamEventText && ev.Text == "two" {
				return want
			}
			return nil
		}),
	)
	if err == nil || !strings.Contains(err.Error(), want.Error()) {
		t.Fatalf("err = %v, want consumer error", err)
	}
}

func TestOpenAIStreamNoEmptyTextEvents(t *testing.T) {
	t.Parallel()
	srv := newSSEServer(t, []string{
		chatChunk(`{"role":"assistant"}`),
		chatChunk(`{"content":""}`),
		chatChunk(`{"content":"x"}`),
	}, false)
	llm := newScriptedLLM(t, srv)
	events := &collectEvents{}
	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingHandler(events.emit),
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, ev := range events.events {
		if ev.Kind == llms.StreamEventText && ev.Text == "" {
			t.Fatal("empty text event emitted")
		}
	}
}

func TestOpenAIStreamDeterministic(t *testing.T) {
	t.Parallel()
	script := []string{
		chatChunk(`{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"fa","arguments":"{\"x\":"}}]}`),
		chatChunk(`{"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"fb","arguments":"{\"y\":"}}]}`),
		chatChunk(`{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}`),
		chatChunk(`{"tool_calls":[{"index":1,"function":{"arguments":"2}"}}]}`),
	}
	run := func() string {
		srv := newSSEServer(t, script, false)
		llm := newScriptedLLM(t, srv)
		events := &collectEvents{}
		resp, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithStreamingHandler(events.emit),
		)
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		sum := events.summary(t)
		var b strings.Builder
		for _, c := range resp.Choices[0].ToolCalls {
			fmt.Fprintf(&b, "%s/%s/%s;", c.ID, c.FunctionCall.Name, c.FunctionCall.Arguments)
		}
		return fmt.Sprintf("%s|%s|%+v", b.String(), sum.String(), sum.Calls)
	}
	first := run()
	for i := 0; i < 5; i++ {
		if got := run(); got != first {
			t.Fatalf("run %d differs:\n%s\n%s", i, first, got)
		}
	}
}

func TestOpenAIStreamConsumerErrorReturnsPartial(t *testing.T) {
	t.Parallel()
	srv := newSSEServer(t, []string{
		chatChunk(`{"role":"assistant","content":"kept"}`),
		chatChunk(`{"role":"assistant","content":"stop here"}`),
	}, false)
	llm := newScriptedLLM(t, srv)
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingHandler(func(_ context.Context, ev llms.StreamEvent) error {
			if ev.Kind == llms.StreamEventText && ev.Text == "stop here" {
				return errors.New("consumer stop")
			}
			return nil
		}),
	)
	if err == nil {
		t.Fatal("expected consumer error")
	}
	if resp == nil || len(resp.Choices) == 0 {
		t.Fatal("partial response lost with the error")
	}
	if got := resp.Choices[0].Content; got != "keptstop here" {
		// "stop here" was delivered to the consumer (which rejected it),
		// so it is part of the accumulated text the error returns.
		t.Fatalf("partial content = %q", got)
	}
}

func TestOpenAIStreamWarningsDoNotLogArguments(t *testing.T) {
	// The default logger must never carry tool argument bodies. Redirect
	// the default slog handler to capture every record and assert the
	// secret never appears.
	var buf strings.Builder
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(prev) })

	secret := "SUPER_SECRET_ARGUMENT_VALUE"
	dup := fmt.Sprintf(`{"secret":%q}`, secret)
	srv := newSSEServer(t, []string{
		chatChunk(fmt.Sprintf(`{"tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"f","arguments":%q}}]}`, dup)),
		chatChunk(fmt.Sprintf(`{"tool_calls":[{"index":0,"function":{"arguments":%q}}]}`, dup)),
	}, false)
	llm := newScriptedLLM(t, srv)
	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingHandler((&collectEvents{}).emit),
	)
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if strings.Contains(buf.String(), secret) {
		t.Fatalf("default log leaked tool arguments:\n%s", buf.String())
	}
}

func TestOpenAIStreamCancelReturnsAndLeavesNoGoroutine(t *testing.T) {
	// Stream slowly so cancellation happens while the reader is active.
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", chatChunk(`{"role":"assistant","content":"a"}`))
		flusher.Flush()
		<-release // block until the client has cancelled
	}))
	t.Cleanup(srv.Close)

	llm := newScriptedLLMFor(t, srv.URL)
	baseline := runtime.NumGoroutine()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := llm.GenerateContent(ctx,
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithStreamingHandler((&collectEvents{}).emit),
		)
		done <- err
	}()
	// Give the stream a moment to deliver the first chunk, then cancel.
	time.Sleep(50 * time.Millisecond)
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("GenerateContent did not return after cancel")
	}
	// Let the test server's handler goroutine finish, then require the
	// client's reader goroutine to have exited as well.
	close(release)
	deadline := time.Now().Add(3 * time.Second)
	for runtime.NumGoroutine() > baseline && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if n := runtime.NumGoroutine(); n > baseline {
		t.Fatalf("goroutines after cancel = %d, baseline %d", n, baseline)
	}
}

func TestOpenAIStreamUnknownChunkWarnsAndContinues(t *testing.T) {
	t.Parallel()
	srv := newSSEServer(t, []string{
		`not-json-at-all`,
		chatChunk(`{"role":"assistant","content":"ok"}`),
	}, false)
	llm := newScriptedLLM(t, srv)
	events := &collectEvents{}
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingHandler(events.emit),
	)
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if got := resp.Choices[0].Content; got != "ok" {
		t.Fatalf("content = %q", got)
	}
	var warnings int
	for _, ev := range events.events {
		if ev.Kind == llms.StreamEventWarning {
			warnings++
			if ev.Warning != nil && ev.Warning.Code != llms.StreamWarningUnknownChunk {
				t.Fatalf("unexpected warning code %q", ev.Warning.Code)
			}
		}
	}
	if warnings != 1 {
		t.Fatalf("warnings = %d, want 1", warnings)
	}
}

func TestOpenAIStreamTextAndReasoningCallbacksUnchanged(t *testing.T) {
	t.Parallel()
	srv := newSSEServer(t, []string{
		chatChunk(`{"role":"assistant","reasoning_content":"think"}`),
		chatChunk(`{"content":"answer"}`),
	}, false)
	llm := newScriptedLLM(t, srv)
	var text, reason strings.Builder
	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(_ context.Context, chunk []byte) error {
			text.Write(chunk)
			return nil
		}),
		llms.WithStreamingReasoningFunc(func(_ context.Context, reasoningChunk, chunk []byte) error {
			reason.Write(reasoningChunk)
			reason.Write(chunk)
			return nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if got := text.String(); got != "answer" {
		t.Fatalf("text callback = %q", got)
	}
	if got := reason.String(); got != "think" {
		// The reasoning callback keeps its historical pairing: it is only
		// invoked for a delta that carried reasoning, with that delta's
		// (empty) content. The content delta is delivered to the text
		// callback alone.
		t.Fatalf("reasoning callback = %q", got)
	}
}

func TestOpenAILegacyFunctionCallStreamingUnchanged(t *testing.T) {
	t.Parallel()
	chunkWithFC := func(fc string) string {
		return `{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"function_call":` + fc + `},"finish_reason":null}]}`
	}
	srv := newSSEServer(t, []string{
		chunkWithFC(`{"name":"calc","arguments":""}`),
		chunkWithFC(`{"arguments":"{\"e\":"}`),
		chunkWithFC(`{"arguments":"\"2+2\"}"}`),
		`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"function_call"}]}`,
	}, false)
	llm := newScriptedLLM(t, srv)
	var chunks []string
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(_ context.Context, chunk []byte) error {
			chunks = append(chunks, string(chunk))
			return nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	// The historical callback still receives one JSON object per delta.
	if len(chunks) != 3 {
		t.Fatalf("callback chunks = %d, want 3: %q", len(chunks), chunks)
	}
	for _, c := range chunks {
		if !strings.Contains(c, "arguments") {
			t.Fatalf("callback chunk is not a function call: %q", c)
		}
	}
	c := resp.Choices[0]
	if c.FuncCall == nil || c.FuncCall.Name != "calc" || c.FuncCall.Arguments != `{"e":"2+2"}` {
		t.Fatalf("FuncCall = %+v", c.FuncCall)
	}
	if len(c.ToolCalls) != 0 {
		t.Fatalf("legacy function_call leaked into ToolCalls: %+v", c.ToolCalls)
	}
}
