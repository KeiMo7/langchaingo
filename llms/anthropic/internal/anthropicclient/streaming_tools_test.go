package anthropicclient

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
)

type eventSink struct {
	mu     sync.Mutex
	events []llms.StreamEvent
}

func (s *eventSink) emit(_ context.Context, ev llms.StreamEvent) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, ev)
	return nil
}

func (s *eventSink) summary(t *testing.T) llms.StreamSummary {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	var ends []llms.StreamEvent
	for _, ev := range s.events {
		if ev.Kind == llms.StreamEventEnd {
			ends = append(ends, ev)
		}
	}
	require.Len(t, ends, 1, "want exactly one end event")
	return *ends[0].Summary
}

func anthropicSSE(events ...string) *http.Response {
	var b strings.Builder
	for _, ev := range events {
		b.WriteString("event: x\n")
		b.WriteString("data: ")
		b.WriteString(ev)
		b.WriteString("\n\n")
	}
	return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(b.String()))}
}

func blockStart(index int, typ, id, name string) string {
	cb, _ := json.Marshal(map[string]any{
		"type": typ, "id": id, "name": name, "input": map[string]any{},
	})
	return fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":%s}`, index, cb)
}

func jsonDelta(index int, partial string) string {
	b, _ := json.Marshal(partial)
	return fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"input_json_delta","partial_json":%s}}`, index, b)
}

func textDelta(index int, text string) string {
	b, _ := json.Marshal(text)
	return fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"text_delta","text":%s}}`, index, b)
}

func TestAnthropicStreamAssemblesToolCalls(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	sink := &eventSink{}
	payload := &messagePayload{StreamingEventFunc: sink.emit, StreamingProvider: "anthropic"}

	r := anthropicSSE(
		`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		textDelta(0, "Working"),
		textDelta(0, "..."),
		`{"type":"content_block_stop","index":0}`,
		blockStart(1, "tool_use", "toolu_a", "fa"),
		jsonDelta(1, `{"x":`),
		jsonDelta(1, `1}`),
		`{"type":"content_block_stop","index":1}`,
		blockStart(2, "tool_use", "toolu_b", "fb"),
		jsonDelta(2, `{"y":`),
		jsonDelta(2, `2}`),
		`{"type":"content_block_stop","index":2}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`,
		`{"type":"message_stop"}`,
	)
	resp, err := parseStreamingMessageResponse(ctx, r, payload)
	require.NoError(t, err)
	require.NotNil(t, resp)

	sum := sink.summary(t)
	require.Equal(t, llms.StreamStopReasonComplete, sum.StopReason)
	require.Less(t, 1, len(sum.Calls))
	byID := map[string]llms.ToolCallStatus{}
	for _, c := range sum.Calls {
		byID[c.ID] = c
	}
	require.Equal(t, llms.ToolCallStateComplete, byID["toolu_a"].State)
	require.Equal(t, llms.ToolCallStateComplete, byID["toolu_b"].State)

	// The historical response still carries the tool_use blocks.
	var sawToolA bool
	for _, c := range resp.Content {
		if tu, ok := c.(*ToolUseContent); ok {
			if tu.ID == "toolu_a" {
				sawToolA = true
				require.Equal(t, map[string]interface{}{"x": float64(1)}, tu.Input)
			}
		}
	}
	require.True(t, sawToolA)
}

func TestAnthropicStreamTruncatedToolCall(t *testing.T) {
	t.Parallel()
	sink := &eventSink{}
	payload := &messagePayload{StreamingEventFunc: sink.emit}

	r := anthropicSSE(
		`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`,
		blockStart(0, "tool_use", "toolu_cut", "f"),
		jsonDelta(0, `{"x":`),
		`{"type":"message_stop"}`,
	)
	_, err := parseStreamingMessageResponse(context.Background(), r, payload)
	require.NoError(t, err)

	sum := sink.summary(t)
	require.Len(t, sum.Calls, 1)
	require.Equal(t, llms.ToolCallStateIncomplete, sum.Calls[0].State)
	require.Len(t, sum.CompleteCalls(), 0)
}

func TestAnthropicStreamMidStreamErrorKeepsPartial(t *testing.T) {
	t.Parallel()
	sink := &eventSink{}
	payload := &messagePayload{StreamingEventFunc: sink.emit}

	r := anthropicSSE(
		`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		textDelta(0, "partial"),
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
	)
	// The stream simply ends without message_stop: the consumer loop ends
	// and the terminal summary is still delivered.
	resp, err := parseStreamingMessageResponse(context.Background(), r, payload)
	require.NoError(t, err)
	require.NotNil(t, resp)
}

func TestAnthropicStreamContextCancelDeliversSummaryOnce(t *testing.T) {
	t.Parallel()
	sink := &eventSink{}
	payload := &messagePayload{StreamingEventFunc: sink.emit}

	ctx, cancel := context.WithCancel(context.Background())
	pr, pw := io.Pipe()
	go func() {
		_, _ = io.WriteString(pw, "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n")
		// Leave the stream open, then cancel.
		cancel()
		_ = pw.Close()
	}()

	_, _ = parseStreamingMessageResponse(ctx, &http.Response{Body: pr}, payload)
	sum := sink.summary(t)
	require.Equal(t, llms.StreamStopReasonCanceled, sum.StopReason)
}

func TestAnthropicStreamUnknownEventWarnsAndContinues(t *testing.T) {
	t.Parallel()
	sink := &eventSink{}
	payload := &messagePayload{StreamingEventFunc: sink.emit}

	r := anthropicSSE(
		`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`,
		`{"type":"brand_new_event","index":0}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		textDelta(0, "ok"),
		`{"type":"message_stop"}`,
	)
	_, err := parseStreamingMessageResponse(context.Background(), r, payload)
	require.NoError(t, err)

	sink.mu.Lock()
	defer sink.mu.Unlock()
	var warnings int
	for _, ev := range sink.events {
		if ev.Kind == llms.StreamEventWarning {
			warnings++
		}
	}
	require.Equal(t, 1, warnings)
}

// streamingTransport routes a canned SSE body through a real client so the
// request path is exercised end to end.
func newAnthropicTestServer(t *testing.T, events []string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range events {
			_, _ = io.WriteString(w, "data: "+ev+"\n\n")
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestAnthropicStreamDeterministic(t *testing.T) {
	t.Parallel()
	events := []string{
		`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`,
		blockStart(0, "tool_use", "a", "fa"),
		jsonDelta(0, `{"x":`),
		jsonDelta(0, `1}`),
		`{"type":"content_block_stop","index":0}`,
		blockStart(1, "tool_use", "b", "fb"),
		jsonDelta(1, `{"y":2}`),
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_stop"}`,
	}
	run := func() string {
		sink := &eventSink{}
		_, err := parseStreamingMessageResponse(context.Background(), anthropicSSE(events...),
			&messagePayload{StreamingEventFunc: sink.emit})
		require.NoError(t, err)
		sum := sink.summary(t)
		var b strings.Builder
		for _, c := range sum.Calls {
			fmt.Fprintf(&b, "%d/%s/%s/%s;", c.Index, c.ID, c.Name, c.State)
		}
		return b.String() + "|" + sum.String()
	}
	first := run()
	for i := 0; i < 5; i++ {
		require.Equal(t, first, run(), "run %d differs", i)
	}
}

func TestAnthropicStreamLogsNoArguments(t *testing.T) {
	var buf strings.Builder
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(prev) })

	secret := "ANTHROPIC_SECRET_ARG"
	events := []string{
		`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`,
		blockStart(0, "tool_use", "a", "f"),
		// Deliberately invalid arguments so the call is incomplete and a
		// warning-bearing summary is produced.
		jsonDelta(0, `{"secret":"`+secret+`"`),
		jsonDelta(0, `{"secret":"`+secret+`"`),
		`{"type":"unknown_event"}`,
		`{"type":"message_stop"}`,
	}
	sink := &eventSink{}
	_, err := parseStreamingMessageResponse(context.Background(), anthropicSSE(events...),
		&messagePayload{StreamingEventFunc: sink.emit})
	require.NoError(t, err)

	if strings.Contains(buf.String(), secret) {
		t.Fatalf("default log leaked arguments:\n%s", buf.String())
	}
}
