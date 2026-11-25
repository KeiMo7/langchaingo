package agents_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/internal/httprr"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/prompts"
	"github.com/tmc/langchaingo/tools"
)

// hasExistingRecording checks if a httprr recording exists for this test
func hasExistingRecording(t *testing.T) bool {
	testName := strings.ReplaceAll(t.Name(), "/", "_")
	testName = strings.ReplaceAll(testName, " ", "_")
	recordingPath := filepath.Join("testdata", testName+".httprr")
	_, err := os.Stat(recordingPath)
	return err == nil
}

func TestOpenAIFunctionsAgentWithHTTPRR(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	// Skip if no recording available and no credentials
	if !hasExistingRecording(t) {
		t.Skip("No httprr recording available. Hint: Re-run tests with -httprecord=. to record new HTTP interactions")
	}

	rr := httprr.OpenForTest(t, http.DefaultTransport)

	// Configure OpenAI client with httprr
	opts := []openai.Option{
		openai.WithModel("gpt-4o"),
		openai.WithHTTPClient(rr.Client()),
	}
	if rr.Replaying() {
		opts = append(opts, openai.WithToken("test-api-key"))
	}

	llm, err := openai.New(opts...)
	if err != nil {
		t.Fatal(err)
	}

	// Create a simple calculator tool
	calculator := tools.Calculator{}

	// Create the OpenAI Functions agent
	agent := agents.NewOpenAIFunctionsAgent(
		llm,
		[]tools.Tool{calculator},
		agents.NewOpenAIOption().WithSystemMessage("You are a helpful assistant that can perform calculations."),
	)

	// Create executor
	executor := agents.NewExecutor(agent)

	// Run a simple calculation
	result, err := chains.Run(ctx, executor, "What is 15 multiplied by 4?")
	if err != nil {
		// Check if this is a recording mismatch error. The openai client
		// reports a replay miss as a network error.
		if strings.Contains(err.Error(), "cached HTTP response not found") ||
			(rr.Replaying() && strings.Contains(err.Error(), "network error")) {
			t.Skip("Recording format has changed or is incompatible. Hint: Re-run tests with -httprecord=. to record new HTTP interactions")
		}
		t.Fatal(err)
	}

	t.Logf("Agent response: %s", result)

	// Verify the result contains 60
	if !strings.Contains(result, "60") {
		t.Errorf("expected calculation result 60 in response, got: %s", result)
	}
}

func TestOpenAIFunctionsAgentComplexCalculation(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	// Skip if no recording available and no credentials
	if !hasExistingRecording(t) {
		t.Skip("No httprr recording available. Hint: Re-run tests with -httprecord=. to record new HTTP interactions")
	}

	rr := httprr.OpenForTest(t, http.DefaultTransport)

	// Configure OpenAI client with httprr
	opts := []openai.Option{
		openai.WithModel("gpt-4o"),
		openai.WithHTTPClient(rr.Client()),
	}
	if rr.Replaying() {
		opts = append(opts, openai.WithToken("test-api-key"))
	}

	llm, err := openai.New(opts...)
	require.NoError(t, err)

	// Create a calculator tool
	calculator := tools.Calculator{}

	// Create the OpenAI Functions agent with extra messages
	agent := agents.NewOpenAIFunctionsAgent(
		llm,
		[]tools.Tool{calculator},
		agents.NewOpenAIOption().WithSystemMessage("You are a helpful math assistant."),
		agents.NewOpenAIOption().WithExtraMessages([]prompts.MessageFormatter{
			prompts.NewHumanMessagePromptTemplate("Please show your work step by step.", nil),
		}),
	)

	// Create executor with options
	executor := agents.NewExecutor(
		agent,
		agents.WithMaxIterations(5),
	)

	// Run a more complex calculation
	result, err := chains.Run(ctx, executor, "If I have 3 groups of 7 items, and I add 9 more items, how many items do I have in total?")
	if err != nil {
		// Check if this is a recording mismatch error. The openai client
		// reports a replay miss as a network error.
		if strings.Contains(err.Error(), "cached HTTP response not found") ||
			(rr.Replaying() && strings.Contains(err.Error(), "network error")) {
			t.Skip("Recording format has changed or is incompatible. Hint: Re-run tests with -httprecord=. to record new HTTP interactions")
		}
		t.Fatalf("failed to run agent: %v", err)
	}
	t.Logf("Agent response: %s", result)

	// Verify the result contains 30 (3*7 + 9 = 21 + 9 = 30)
	if !strings.Contains(result, "30") {
		t.Errorf("expected calculation result 30 in response, got: %s", result)
	}
}

// TestOpenAIFunctionsAgent_ParseOutput_NilResponse tests that ParseOutput handles nil response gracefully
func TestOpenAIFunctionsAgent_ParseOutput_NilResponse(t *testing.T) {
	t.Parallel()
	agent := &agents.OpenAIFunctionsAgent{}

	// Test with nil response - should return error instead of panic
	_, _, err := agent.ParseOutput(nil)
	if err == nil {
		t.Error("expected error for nil response")
	}
}

// TestOpenAIFunctionsAgent_ParseOutput_EmptyChoices tests that ParseOutput handles empty choices gracefully
func TestOpenAIFunctionsAgent_ParseOutput_EmptyChoices(t *testing.T) {
	t.Parallel()
	agent := &agents.OpenAIFunctionsAgent{}

	// Test with empty choices - should return error instead of panic
	resp := &llms.ContentResponse{
		Choices: []*llms.ContentChoice{},
	}
	_, _, err := agent.ParseOutput(resp)
	if err == nil {
		t.Error("expected error for empty choices")
	}
}

// TestOpenAIFunctionsAgent_ParseOutput_MultipleToolCalls tests multiple tool calls handling
func TestOpenAIFunctionsAgent_ParseOutput_MultipleToolCalls(t *testing.T) {
	t.Parallel()
	agent := &agents.OpenAIFunctionsAgent{}

	// Test multiple tool calls - should handle all calls, not just first one
	resp := &llms.ContentResponse{
		Choices: []*llms.ContentChoice{
			{
				ToolCalls: []llms.ToolCall{
					{
						ID: "call1",
						FunctionCall: &llms.FunctionCall{
							Name:      "calculator",
							Arguments: `{"__arg1": "2+2"}`,
						},
					},
					{
						ID: "call2",
						FunctionCall: &llms.FunctionCall{
							Name:      "weather",
							Arguments: `{"__arg1": "Seattle"}`,
						},
					},
				},
			},
		},
	}

	actions, finish, err := agent.ParseOutput(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if finish != nil {
		t.Error("expected actions, got finish")
	}
	if len(actions) != 2 {
		t.Errorf("expected 2 actions, got %d", len(actions))
	}
}

// scriptedSSE serves one fixed SSE script for every request. Multiple
// requests (one per agent iteration) replay the same script, so a test can
// drive a full agent loop deterministically without a provider.
func scriptedSSE(t *testing.T, events []string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "no flusher", http.StatusInternalServerError)
			return
		}
		for _, ev := range events {
			_, _ = io.WriteString(w, "data: "+ev+"\n\n")
			flusher.Flush()
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	t.Cleanup(srv.Close)
	return srv
}

func chunk(delta string) string {
	return `{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":` + delta + `,"finish_reason":null}]}`
}

type countingTool struct {
	name  string
	calls atomic.Int32
}

func (c *countingTool) Name() string        { return c.name }
func (c *countingTool) Description() string { return "test tool " + c.name }
func (c *countingTool) Call(_ context.Context, input string) (string, error) {
	c.calls.Add(1)
	return "ok:" + input, nil
}

// TestOpenAIFunctionsAgentStreamsToolCalls drives the agent over a scripted
// SSE stream whose tool call arguments arrive in fragments. The assembled
// call must reach the agent loop and the tool must run.
func TestOpenAIFunctionsAgentStreamsToolCalls(t *testing.T) {
	t.Parallel()
	srv := scriptedSSE(t, []string{
		chunk(`{"role":"assistant","content":"calling"}`),
		chunk(`{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"demo","arguments":""}}]}`),
		// Fragments split mid-arguments, with no type field on continuation.
		chunk(`{"tool_calls":[{"index":0,"function":{"arguments":"{\"__arg1\":"}}]}`),
		chunk(`{"tool_calls":[{"index":0,"function":{"arguments":"\"7*6\"}"}}]}`),
		chunk(`{"finish_reason":"tool_calls"}`),
	})
	llm, err := openai.New(openai.WithToken("k"), openai.WithBaseURL(srv.URL), openai.WithModel("m"))
	require.NoError(t, err)

	demo := &countingTool{name: "demo"}
	ag := agents.NewOpenAIFunctionsAgent(llm, []tools.Tool{demo})
	// A callbacks handler is what makes the agent install its streaming
	// callback; without one it does not stream at all.
	ag.CallbacksHandler = callbacks.SimpleHandler{}
	ex := agents.NewExecutor(ag, agents.WithMaxIterations(1))

	_, _ = chains.Run(context.Background(), ex, "compute")
	require.Equal(t, int32(1), demo.calls.Load(), "streamed tool call was not executed exactly once")
}

// TestOpenAIFunctionsAgentTruncatedToolCallNotExecuted shows that a call whose
// arguments never parse is never handed to a tool.
func TestOpenAIFunctionsAgentTruncatedToolCallNotExecuted(t *testing.T) {
	t.Parallel()
	srv := scriptedSSE(t, []string{
		chunk(`{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"demo","arguments":"{\"__arg1\":"}}]}`),
		chunk(`{"finish_reason":"tool_calls"}`),
	})
	llm, err := openai.New(openai.WithToken("k"), openai.WithBaseURL(srv.URL), openai.WithModel("m"))
	require.NoError(t, err)

	demo := &countingTool{name: "demo"}
	ag := agents.NewOpenAIFunctionsAgent(llm, []tools.Tool{demo})
	ag.CallbacksHandler = callbacks.SimpleHandler{}
	ex := agents.NewExecutor(ag, agents.WithMaxIterations(1))
	_, _ = chains.Run(context.Background(), ex, "compute")
	require.Equal(t, int32(0), demo.calls.Load(), "truncated tool call must not execute")
}
