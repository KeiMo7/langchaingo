package openaiclient

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/tmc/langchaingo/llms"
)

const (
	defaultChatModel = "gpt-3.5-turbo"
)

var ErrContentExclusive = errors.New("only one of Content / MultiContent allowed in message")

type StreamOptions struct {
	// If set, an additional chunk will be streamed before the data: [DONE] message.
	// The usage field on this chunk shows the token usage statistics for the entire request,
	// and the choices field will always be an empty array.
	// All other chunks will also include a usage field, but with a null value.
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// ChatRequest is a request to complete a chat completion..
type ChatRequest struct {
	Model       string         `json:"model"`
	Messages    []*ChatMessage `json:"messages"`
	Temperature float64        `json:"temperature"`
	TopP        float64        `json:"top_p,omitempty"`
	// Deprecated: Use MaxCompletionTokens
	// Note: Some OpenAI-compatible servers still require this field
	MaxTokens           int      `json:"max_tokens,omitempty"`
	MaxCompletionTokens int      `json:"max_completion_tokens,omitempty"`
	N                   int      `json:"n,omitempty"`
	StopWords           []string `json:"stop,omitempty"`
	Stream              bool     `json:"stream,omitempty"`
	FrequencyPenalty    float64  `json:"frequency_penalty,omitempty"`
	PresencePenalty     float64  `json:"presence_penalty,omitempty"`
	Seed                int      `json:"seed,omitempty"`

	// ResponseFormat is the format of the response.
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`

	// LogProbs indicates whether to return log probabilities of the output tokens or not.
	// If true, returns the log probabilities of each output token returned in the content of message.
	// This option is currently not available on the gpt-4-vision-preview model.
	LogProbs bool `json:"logprobs,omitempty"`
	// TopLogProbs is an integer between 0 and 5 specifying the number of most likely tokens to return at each
	// token position, each with an associated log probability.
	// logprobs must be set to true if this parameter is used.
	TopLogProbs int `json:"top_logprobs,omitempty"`

	Tools []Tool `json:"tools,omitempty"`
	// This can be either a string or a ToolChoice object.
	// If it is a string, it should be one of 'none', or 'auto', otherwise it should be a ToolChoice object specifying a specific tool to use.
	ToolChoice any `json:"tool_choice,omitempty"`

	// Options for streaming response. Only set this when you set stream: true.
	StreamOptions *StreamOptions `json:"stream_options,omitempty"`

	// ReasoningEffort controls thinking effort for reasoning models (o1, o3, GPT-5).
	// Valid values: "minimal" (GPT-5 only), "low", "medium", "high"
	ReasoningEffort string `json:"reasoning_effort,omitempty"`

	// StreamingFunc is a function to be called for each chunk of a streaming response.
	// Return an error to stop streaming early.
	StreamingFunc func(ctx context.Context, chunk []byte) error `json:"-"`

	// StreamingReasoningFunc is a function to be called for each reasoning and content chunk of a streaming response.
	// Return an error to stop streaming early.
	StreamingReasoningFunc func(ctx context.Context, reasoningChunk, chunk []byte) error `json:"-"`

	// StreamingEventFunc, when set, receives provider-neutral stream events:
	// text, reasoning and tool call increments, per-call lifecycle events
	// and a terminal summary. It is additive: StreamingFunc, if also set,
	// still receives the same text fragments.
	StreamingEventFunc llms.StreamEventFunc `json:"-"`
	// StreamingProvider names the adapter in warning logs.
	StreamingProvider string `json:"-"`

	// Deprecated: use Tools instead.
	Functions []FunctionDefinition `json:"functions,omitempty"`
	// Deprecated: use ToolChoice instead.
	FunctionCallBehavior FunctionCallBehavior `json:"function_call,omitempty"`

	// Metadata allows you to specify additional information that will be passed to the model.
	Metadata map[string]any `json:"metadata,omitempty"`

	// WebSearchOptions configures web search behavior for search-enabled models
	// like gpt-4o-search-preview and gpt-4o-mini-search-preview.
	WebSearchOptions *WebSearchOptions `json:"web_search_options,omitempty"`
}

// MarshalJSON ensures that only one of MaxTokens or MaxCompletionTokens is sent.
// OpenAI's API returns an error if both fields are present.
// Also omits temperature for reasoning models (GPT-5, o1, o3) that only accept default temperature.
func (r ChatRequest) MarshalJSON() ([]byte, error) {
	type Alias ChatRequest
	aux := struct {
		*Alias
		MaxTokens           *int     `json:"max_tokens,omitempty"`
		MaxCompletionTokens *int     `json:"max_completion_tokens,omitempty"`
		Temperature         *float64 `json:"temperature,omitempty"`
	}{
		Alias: (*Alias)(&r),
	}

	// Handle temperature for reasoning and search-preview models
	if isReasoningModel(r.Model) || isSearchPreviewModel(r.Model) {
		// These models reject non-default temperature values.
		// Omit temperature field to let API use its default value
		aux.Temperature = nil
	} else {
		// For regular models, always send temperature
		aux.Temperature = &r.Temperature
	}

	// Ensure only one token field is sent
	if r.MaxCompletionTokens > 0 && r.MaxTokens > 0 {
		// Both are set - this shouldn't happen with our logic,
		// but if it does, prefer MaxCompletionTokens (modern field)
		aux.MaxCompletionTokens = &r.MaxCompletionTokens
		aux.MaxTokens = nil
	} else if r.MaxCompletionTokens > 0 {
		aux.MaxCompletionTokens = &r.MaxCompletionTokens
		aux.MaxTokens = nil
	} else if r.MaxTokens > 0 {
		aux.MaxTokens = &r.MaxTokens
		aux.MaxCompletionTokens = nil
	}

	return json.Marshal(&aux)
}

// isSearchPreviewModel reports whether model is a search-preview model,
// including dated snapshots such as gpt-4o-search-preview-2025-03-11.
func isSearchPreviewModel(model string) bool {
	return strings.Contains(model, "-search-preview")
}

// isReasoningModel returns true if the model is a reasoning model that has temperature constraints.
// Reasoning models (GPT-5, o1, o3) only accept temperature=1 and reject other values.
func isReasoningModel(model string) bool {
	// o1 series: o1-preview, o1-mini
	if strings.HasPrefix(model, "o1-") {
		return true
	}
	// o3 series: o3, o3-mini (note: "o3" without suffix is also valid)
	if model == "o3" || strings.HasPrefix(model, "o3-") {
		return true
	}
	// GPT-5 series (when released)
	if strings.HasPrefix(model, "gpt-5") {
		return true
	}
	return false
}

// ToolType is the type of a tool.
type ToolType string

const (
	ToolTypeFunction ToolType = "function"
)

// WebSearchOptions configures web search behavior for OpenAI models.
// This is used with search-enabled models like gpt-4o-search-preview.
type WebSearchOptions struct {
	// SearchContextSize controls how much context is gathered from web search.
	// Valid values: "low", "medium", "high". Higher values provide more context
	// but increase latency and cost.
	SearchContextSize string `json:"search_context_size,omitempty"`

	// UserLocation provides approximate user location for localized search results.
	UserLocation *UserLocation `json:"user_location,omitempty"`
}

// UserLocation represents the user's approximate location for web search.
type UserLocation struct {
	// Type must be "approximate" for user-provided location.
	Type string `json:"type"`

	// Approximate contains the approximate location details.
	Approximate *ApproximateLocation `json:"approximate,omitempty"`
}

// ApproximateLocation contains approximate location information.
type ApproximateLocation struct {
	// Country is the two-letter ISO country code (e.g., "US", "GB").
	Country string `json:"country,omitempty"`

	// City is the city name (e.g., "San Francisco", "London").
	City string `json:"city,omitempty"`

	// Region is the region or state (e.g., "California", "London").
	Region string `json:"region,omitempty"`
}

// Tool is a tool to use in a chat request.
type Tool struct {
	Type     ToolType           `json:"type"`
	Function FunctionDefinition `json:"function,omitempty"`
}

// ToolChoice is a choice of a tool to use.
type ToolChoice struct {
	Type     ToolType     `json:"type"`
	Function ToolFunction `json:"function,omitempty"`
}

// ToolFunction is a function to be called in a tool choice.
type ToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolCall is a call to a tool.
type ToolCall struct {
	ID       string       `json:"id,omitempty"`
	Type     ToolType     `json:"type"`
	Function ToolFunction `json:"function,omitempty"`
	// Index is the position of the call within the message. Streaming
	// deltas carry it; it is the stable identity a fragmented call is
	// assembled under. It is omitted from requests.
	Index *int `json:"index,omitempty"`
}

type ResponseFormatJSONSchemaProperty struct {
	Type                 string                                       `json:"type"`
	Description          string                                       `json:"description,omitempty"`
	Enum                 []interface{}                                `json:"enum,omitempty"`
	Items                *ResponseFormatJSONSchemaProperty            `json:"items,omitempty"`
	Properties           map[string]*ResponseFormatJSONSchemaProperty `json:"properties,omitempty"`
	AdditionalProperties bool                                         `json:"additionalProperties"`
	Required             []string                                     `json:"required,omitempty"`
	Ref                  string                                       `json:"$ref,omitempty"`
}

type ResponseFormatJSONSchema struct {
	Name   string                            `json:"name"`
	Strict bool                              `json:"strict"`
	Schema *ResponseFormatJSONSchemaProperty `json:"schema"`
}

// ResponseFormat is the format of the response.
type ResponseFormat struct {
	Type       string                    `json:"type"`
	JSONSchema *ResponseFormatJSONSchema `json:"json_schema,omitempty"`
}

// ChatMessage is a message in a chat request.
type ChatMessage struct { //nolint:musttag
	// The role of the author of this message. One of system, user, assistant, function, or tool.
	Role string

	// The content of the message.
	// This field is mutually exclusive with MultiContent.
	Content string

	// MultiContent is a list of content parts to use in the message.
	MultiContent []llms.ContentPart

	// The name of the author of this message. May contain a-z, A-Z, 0-9, and underscores,
	// with a maximum length of 64 characters.
	Name string

	// ToolCalls is a list of tools that were called in the message.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	// FunctionCall represents a function call that was made in the message.
	// Deprecated: use ToolCalls instead.
	FunctionCall *FunctionCall

	// ToolCallID is the ID of the tool call this message is for.
	// Only present in tool messages.
	ToolCallID string `json:"tool_call_id,omitempty"`

	// This field is only used with the deepseek-reasoner model and represents the reasoning contents of the assistant message before the final answer.
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

func (m ChatMessage) MarshalJSON() ([]byte, error) {
	if m.Content != "" && m.MultiContent != nil {
		return nil, ErrContentExclusive
	}
	if text, ok := isSingleTextContent(m.MultiContent); ok {
		m.Content = text
		m.MultiContent = nil
	}
	if len(m.MultiContent) > 0 {
		msg := struct {
			Role         string             `json:"role"`
			Content      string             `json:"-"`
			MultiContent []llms.ContentPart `json:"content,omitempty"`
			Name         string             `json:"name,omitempty"`
			ToolCalls    []ToolCall         `json:"tool_calls,omitempty"`

			// Deprecated: use ToolCalls instead.
			FunctionCall *FunctionCall `json:"function_call,omitempty"`

			// ToolCallID is the ID of the tool call this message is for.
			// Only present in tool messages.
			ToolCallID string `json:"tool_call_id,omitempty"`

			// This field is only used with the deepseek-reasoner model and represents the reasoning contents of the assistant message before the final answer.
			ReasoningContent string `json:"reasoning_content,omitempty"`
		}(m)
		return json.Marshal(msg)
	}
	msg := struct {
		Role         string             `json:"role"`
		Content      string             `json:"content"`
		MultiContent []llms.ContentPart `json:"-"`
		Name         string             `json:"name,omitempty"`
		ToolCalls    []ToolCall         `json:"tool_calls,omitempty"`
		// Deprecated: use ToolCalls instead.
		FunctionCall *FunctionCall `json:"function_call,omitempty"`

		// ToolCallID is the ID of the tool call this message is for.
		// Only present in tool messages.
		ToolCallID string `json:"tool_call_id,omitempty"`

		// This field is only used with the deepseek-reasoner model and represents the reasoning contents of the assistant message before the final answer.
		ReasoningContent string `json:"reasoning_content,omitempty"`
	}(m)
	return json.Marshal(msg)
}

func isSingleTextContent(parts []llms.ContentPart) (string, bool) {
	if len(parts) != 1 {
		return "", false
	}
	tc, isText := parts[0].(llms.TextContent)
	return tc.Text, isText
}

func (m *ChatMessage) UnmarshalJSON(data []byte) error {
	msg := struct {
		Role         string             `json:"role"`
		Content      string             `json:"content"`
		MultiContent []llms.ContentPart `json:"-"` // not expected in response
		Name         string             `json:"name,omitempty"`
		ToolCalls    []ToolCall         `json:"tool_calls,omitempty"`
		// Deprecated: use ToolCalls instead.
		FunctionCall *FunctionCall `json:"function_call,omitempty"`

		// ToolCallID is the ID of the tool call this message is for.
		// Only present in tool messages.
		ToolCallID string `json:"tool_call_id,omitempty"`

		// This field is only used with the deepseek-reasoner model and represents the reasoning contents of the assistant message before the final answer.
		ReasoningContent string `json:"reasoning_content,omitempty"`
	}{}
	err := json.Unmarshal(data, &msg)
	if err != nil {
		return err
	}
	*m = ChatMessage(msg)
	return nil
}

type TopLogProbs struct {
	Token   string  `json:"token"`
	LogProb float64 `json:"logprob"`
	Bytes   []byte  `json:"bytes,omitempty"`
}

// LogProb represents the probability information for a token.
type LogProb struct {
	Token   string  `json:"token"`
	LogProb float64 `json:"logprob"`
	Bytes   []byte  `json:"bytes,omitempty"` // Omitting the field if it is null
	// TopLogProbs is a list of the most likely tokens and their log probability, at this token position.
	// In rare cases, there may be fewer than the number of requested top_logprobs returned.
	TopLogProbs []TopLogProbs `json:"top_logprobs"`
}

// LogProbs is the top-level structure containing the log probability information.
type LogProbs struct {
	// Content is a list of message content tokens with log probability information.
	Content []LogProb `json:"content"`
}

type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonFunctionCall  FinishReason = "function_call"
	FinishReasonToolCalls     FinishReason = "tool_calls"
	FinishReasonContentFilter FinishReason = "content_filter"
	FinishReasonNull          FinishReason = "null"
)

func (r FinishReason) MarshalJSON() ([]byte, error) {
	if r == FinishReasonNull || r == "" {
		return []byte("null"), nil
	}
	return []byte(`"` + string(r) + `"`), nil // best effort to not break future API changes
}

// ChatCompletionChoice is a choice in a chat response.
type ChatCompletionChoice struct {
	Index        int          `json:"index"`
	Message      ChatMessage  `json:"message"`
	FinishReason FinishReason `json:"finish_reason"`
	LogProbs     *LogProbs    `json:"logprobs,omitempty"`
}

// ChatUsage is the usage of a chat completion request.
type ChatUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
		AudioTokens  int `json:"audio_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails struct {
		ReasoningTokens          int `json:"reasoning_tokens"`
		AudioTokens              int `json:"audio_tokens"`
		AcceptedPredictionTokens int `json:"accepted_prediction_tokens"`
		RejectedPredictionTokens int `json:"rejected_prediction_tokens"`
	} `json:"completion_tokens_details"`
}

// ChatCompletionResponse is a response to a chat request.
type ChatCompletionResponse struct {
	ID                string                  `json:"id,omitempty"`
	Created           int64                   `json:"created,omitempty"`
	Choices           []*ChatCompletionChoice `json:"choices,omitempty"`
	Model             string                  `json:"model,omitempty"`
	Object            string                  `json:"object,omitempty"`
	Usage             ChatUsage               `json:"usage,omitempty"`
	SystemFingerprint string                  `json:"system_fingerprint"`
}

type Usage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
		AudioTokens  int `json:"audio_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails struct {
		ReasoningTokens          int `json:"reasoning_tokens"`
		AudioTokens              int `json:"audio_tokens"`
		AcceptedPredictionTokens int `json:"accepted_prediction_tokens"`
		RejectedPredictionTokens int `json:"rejected_prediction_tokens"`
	} `json:"completion_tokens_details"`
}

// StreamedChatResponsePayload is a chunk from the stream.
type StreamedChatResponsePayload struct {
	ID      string  `json:"id,omitempty"`
	Created float64 `json:"created,omitempty"`
	Model   string  `json:"model,omitempty"`
	Object  string  `json:"object,omitempty"`
	Choices []struct {
		Index float64 `json:"index,omitempty"`
		Delta struct {
			Role         string        `json:"role,omitempty"`
			Content      string        `json:"content,omitempty"`
			FunctionCall *FunctionCall `json:"function_call,omitempty"`
			// ToolCalls is a list of tools that were called in the message.
			ToolCalls []*ToolCall `json:"tool_calls,omitempty"`
			// This field is only used with the deepseek-reasoner model and represents the reasoning contents of the assistant message before the final answer.
			ReasoningContent string `json:"reasoning_content,omitempty"`
		} `json:"delta,omitempty"`
		FinishReason FinishReason `json:"finish_reason,omitempty"`
	} `json:"choices,omitempty"`
	SystemFingerprint string `json:"system_fingerprint"`
	// An optional field that will only be present when you set stream_options: {"include_usage": true} in your request.
	// When present, it contains a null value except for the last chunk which contains the token usage statistics
	// for the entire request.
	Usage *Usage `json:"usage,omitempty"`
	Error error  `json:"-"` // use for error handling only
	// Warning reports fragments the reader skipped. It carries no payload
	// bytes, only a count, so the stream can record the gap without
	// exposing content or failing.
	Warning *StreamWarning `json:"-"`
}

// StreamWarning is an internal, payload-free note that the reader skipped
// fragments it could not decode.
type StreamWarning struct {
	Count int
}

// FunctionDefinition is a definition of a function that can be called by the model.
type FunctionDefinition struct {
	// Name is the name of the function.
	Name string `json:"name"`
	// Description is a description of the function.
	Description string `json:"description,omitempty"`
	// Parameters is a list of parameters for the function.
	Parameters any `json:"parameters"`
	// Strict is a flag to enable structured output mode.
	Strict bool `json:"strict,omitempty"`
}

// FunctionCallBehavior is the behavior to use when calling functions.
type FunctionCallBehavior string

const (
	// FunctionCallBehaviorUnspecified is the empty string.
	FunctionCallBehaviorUnspecified FunctionCallBehavior = ""
	// FunctionCallBehaviorNone will not call any functions.
	FunctionCallBehaviorNone FunctionCallBehavior = "none"
	// FunctionCallBehaviorAuto will call functions automatically.
	FunctionCallBehaviorAuto FunctionCallBehavior = "auto"
)

// FunctionCall is a call to a function.
type FunctionCall struct {
	// Name is the name of the function to call.
	Name string `json:"name"`
	// Arguments is the set of arguments to pass to the function.
	Arguments string `json:"arguments"`
}

func (c *Client) createChat(ctx context.Context, payload *ChatRequest) (*ChatCompletionResponse, error) {
	// A provider-neutral event handler streams tool calls as well as text,
	// so it also turns streaming on.
	if payload.StreamingFunc != nil || payload.StreamingReasoningFunc != nil || payload.StreamingEventFunc != nil {
		payload.Stream = true
		if payload.StreamOptions == nil {
			payload.StreamOptions = &StreamOptions{IncludeUsage: true}
		}
	}
	// Build request payload

	// Filter out internal metadata that shouldn't be sent to the API
	originalMetadata := payload.Metadata
	if payload.Metadata != nil {
		filteredMetadata := make(map[string]any)
		for k, v := range payload.Metadata {
			// Skip internal openai: prefixed metadata fields
			if !strings.HasPrefix(k, "openai:") {
				filteredMetadata[k] = v
			}
		}
		if len(filteredMetadata) > 0 {
			payload.Metadata = filteredMetadata
		} else {
			payload.Metadata = nil
		}
	}

	payloadBytes, err := json.Marshal(payload)

	// Restore original metadata
	payload.Metadata = originalMetadata
	if err != nil {
		return nil, err
	}

	// Build request
	body := bytes.NewReader(payloadBytes)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.buildURL("/chat/completions", payload.Model), body)
	if err != nil {
		return nil, err
	}

	c.setHeaders(req)

	// Send request
	r, err := c.httpClient.Do(req)
	if err != nil {
		return nil, sanitizeHTTPError(err)
	}
	defer r.Body.Close()

	if r.StatusCode != http.StatusOK {
		msg := fmt.Sprintf("API returned unexpected status code: %d", r.StatusCode)

		// No need to check the error here: if it fails, we'll just return the
		// status code.
		var errResp errorMessage
		if err := json.NewDecoder(r.Body).Decode(&errResp); err != nil {
			return nil, errors.New(msg)
		}

		return nil, fmt.Errorf("%s: %s", msg, errResp.Error.Message)
	}
	if payload.StreamingFunc != nil || payload.StreamingReasoningFunc != nil || payload.StreamingEventFunc != nil {
		return parseStreamingChatResponse(ctx, r, payload)
	}
	// Parse response
	var response ChatCompletionResponse
	return &response, json.NewDecoder(r.Body).Decode(&response)
}

func parseStreamingChatResponse(ctx context.Context, r *http.Response, payload *ChatRequest) (*ChatCompletionResponse,
	error,
) { //nolint:cyclop,lll
	scanner := bufio.NewScanner(r.Body)
	responseChan := make(chan StreamedChatResponsePayload)

	// Create a context that can be cancelled to stop the goroutine
	readerCtx, cancelReader := context.WithCancel(ctx)
	defer cancelReader()

	// Close the body when the reader context is cancelled. Scan blocks in
	// Read on a slow or stalled upstream, so the context check between
	// lines never runs; closing the body is what unblocks it and lets the
	// reader goroutine exit promptly on cancel or consumer error.
	stopCloser := make(chan struct{})
	go func() {
		select {
		case <-readerCtx.Done():
			_ = r.Body.Close()
		case <-stopCloser:
		}
	}()
	defer close(stopCloser)

	go func() {
		defer close(responseChan)
		skipNonJSON := 0
		for scanner.Scan() {
			// Check if context is cancelled
			select {
			case <-readerCtx.Done():
				return
			default:
			}

			line := scanner.Text()
			if line == "" {
				continue
			}

			// Skip SSE comment lines (any line starting with ':')
			// According to SSE spec: https://www.w3.org/TR/eventsource/
			// "Lines that start with a U+003A COLON character (:) are comments"
			if strings.HasPrefix(line, ":") {
				continue
			}

			// Only process lines that start with "data:"
			if !strings.HasPrefix(line, "data:") {
				// Skip any other non-data lines (like event:, id:, retry:, etc.)
				continue
			}

			data := strings.TrimPrefix(line, "data:") // here use `data:` instead of `data: ` for compatibility
			data = strings.TrimSpace(data)
			if data == "[DONE]" {
				return
			}
			var streamPayload StreamedChatResponsePayload
			err := json.NewDecoder(bytes.NewReader([]byte(data))).Decode(&streamPayload)
			if err != nil {
				// Skip non-JSON data values that some providers might send
				// This could happen if the data field contains non-JSON content
				skipNonJSON++
				// Report the skip immediately so a warning that occurs
				// before [DONE] is still observed; later skips only bump
				// the count.
				if skipNonJSON == 1 {
					select {
					case <-readerCtx.Done():
						return
					case responseChan <- StreamedChatResponsePayload{Warning: &StreamWarning{Count: 1}}:
					}
				}
				continue
			}

			// Non-blocking send with context check
			select {
			case <-readerCtx.Done():
				return
			case responseChan <- streamPayload:
			}
		}
		if err := scanner.Err(); err != nil {
			select {
			case <-readerCtx.Done():
				return
			case responseChan <- StreamedChatResponsePayload{Error: fmt.Errorf("error reading streaming response: %w", err)}:
			}
			return
		}
	}()
	// Combine response
	return combineStreamingChatResponse(readerCtx, payload, responseChan, cancelReader)
}

// streamAccumulator accumulates OpenAI fragments into the response shape and
// drives the provider-neutral llms.Stream.
type streamAccumulator struct {
	payload  *ChatRequest
	response *ChatCompletionResponse
	stream   *llms.Stream

	// functionCall accumulates the legacy function_call deltas.
	functionCall *FunctionCall
}

func combineStreamingChatResponse(
	ctx context.Context,
	payload *ChatRequest,
	responseChan chan StreamedChatResponsePayload,
	cancelReader context.CancelFunc,
) (*ChatCompletionResponse, error) {
	acc := &streamAccumulator{
		payload: payload,
		response: &ChatCompletionResponse{
			Choices: []*ChatCompletionChoice{{}},
		},
	}
	acc.stream = llms.NewStream(payload.StreamingProvider, payload.StreamingEventFunc)

	var streamErr error
	for streamResponse := range responseChan {
		if streamResponse.Error != nil {
			streamErr = streamResponse.Error
			break
		}
		if streamResponse.Warning != nil {
			// An undecodable fragment is recorded and skipped; the rest
			// of the stream keeps flowing.
			if winErr := acc.stream.Warning(ctx, llms.StreamWarning{
				Code:   llms.StreamWarningUnknownChunk,
				Index:  -1,
				Detail: fmt.Sprintf("skipped %d undecodable fragment(s)", streamResponse.Warning.Count),
			}); winErr != nil {
				streamErr = winErr
				break
			}
			continue
		}
		if err := acc.consume(ctx, &streamResponse); err != nil {
			streamErr = err
			break
		}
	}
	// Stop the reader now, before finishing: a consumer error or
	// cancellation must not leave the producer parked on a send while the
	// summary is assembled and delivered.
	parentErr := ctx.Err()
	cancelReader()

	reason := llms.StreamStopReasonComplete
	switch {
	case streamErr != nil:
		reason = llms.StreamStopReasonUpstreamError
		if isConsumerError(streamErr) {
			reason = llms.StreamStopReasonConsumerError
		}
	case parentErr != nil:
		reason = llms.StreamStopReasonCanceled
	}
	// Deliver the terminal summary on a live context: when the stream is
	// ending because its own context was cancelled, the summary must still
	// reach the observer exactly once.
	closeCtx := ctx
	if closeCtx.Err() != nil {
		closeCtx = context.Background()
	}
	acc.stream.Close(closeCtx, reason, streamErr)

	// The assembled tool call list keeps the shape the non-streaming path
	// returns; only structurally valid calls are exposed as executable.
	acc.finish()

	if streamErr != nil {
		// When the parent context was cancelled, the read error is a
		// symptom of that cancellation, not the cause. Report the
		// cancellation so callers can tell the two apart.
		if parentErr != nil {
			return acc.response, parentErr
		}
		return acc.response, streamErr
	}
	return acc.response, parentErr
}

// errConsumer is wrapped around an error returned by one of the streaming
// callbacks, so the terminal summary can tell it apart from a transport
// failure.
type errConsumer struct{ err error }

func (e *errConsumer) Error() string { return e.err.Error() }
func (e *errConsumer) Unwrap() error { return e.err }

func isConsumerError(err error) bool {
	var ce *errConsumer
	return errors.As(err, &ce)
}

func (a *streamAccumulator) consume(ctx context.Context, sr *StreamedChatResponsePayload) error {
	if sr.Usage != nil {
		a.response.Usage.CompletionTokens = sr.Usage.CompletionTokens
		a.response.Usage.PromptTokens = sr.Usage.PromptTokens
		a.response.Usage.TotalTokens = sr.Usage.TotalTokens
		a.response.Usage.PromptTokensDetails.AudioTokens = sr.Usage.PromptTokensDetails.AudioTokens
		a.response.Usage.PromptTokensDetails.CachedTokens = sr.Usage.PromptTokensDetails.CachedTokens
		a.response.Usage.CompletionTokensDetails.AudioTokens = sr.Usage.CompletionTokensDetails.AudioTokens
		a.response.Usage.CompletionTokensDetails.AcceptedPredictionTokens = sr.Usage.CompletionTokensDetails.AcceptedPredictionTokens
		a.response.Usage.CompletionTokensDetails.RejectedPredictionTokens = sr.Usage.CompletionTokensDetails.RejectedPredictionTokens
		a.response.Usage.CompletionTokensDetails.ReasoningTokens = sr.Usage.CompletionTokensDetails.ReasoningTokens
	}

	if len(sr.Choices) == 0 {
		return nil
	}
	choice := sr.Choices[0]
	a.response.Choices[0].FinishReason = choice.FinishReason
	a.response.Choices[0].Message.ReasoningContent += choice.Delta.ReasoningContent
	a.response.Choices[0].Message.Content += choice.Delta.Content

	// Reasoning is delivered through the shared stream, then mirrored to
	// the historical reasoning callback with the same pairing it always had
	// (the reasoning fragment plus the content fragment of the delta).
	if choice.Delta.ReasoningContent != "" {
		if err := a.stream.Reasoning(ctx, choice.Delta.ReasoningContent); err != nil {
			return &errConsumer{err}
		}
		if a.payload.StreamingReasoningFunc != nil {
			if err := a.payload.StreamingReasoningFunc(ctx, []byte(choice.Delta.ReasoningContent), []byte(choice.Delta.Content)); err != nil {
				return fmt.Errorf("streaming reasoning func returned an error: %w", err)
			}
		}
	}
	// Text is delivered through the shared stream, then mirrored to the
	// historical text callback with exactly the fragment it always got.
	if choice.Delta.Content != "" {
		if err := a.stream.Text(ctx, choice.Delta.Content); err != nil {
			return &errConsumer{err}
		}
		if a.payload.StreamingFunc != nil {
			if err := a.payload.StreamingFunc(ctx, []byte(choice.Delta.Content)); err != nil {
				return fmt.Errorf("streaming func returned an error: %w", err)
			}
		}
	}

	if choice.Delta.FunctionCall != nil {
		if err := a.consumeFunctionCall(ctx, choice.Delta.FunctionCall); err != nil {
			return err
		}
	}

	for _, tc := range choice.Delta.ToolCalls {
		if tc == nil {
			continue
		}
		if err := a.stream.AddToolCall(ctx, deltaFromToolCall(tc)); err != nil {
			return &errConsumer{err}
		}
	}
	return nil
}

// deltaFromToolCall maps one OpenAI tool call fragment to the neutral form.
// The position comes from the delta's index; the type field is incidental and
// is never used to decide whether the fragment opens a new call.
func deltaFromToolCall(tc *ToolCall) llms.ToolCallDelta {
	idx := 0
	if tc.Index != nil {
		idx = *tc.Index
	}
	return llms.ToolCallDelta{
		Index:     idx,
		ID:        tc.ID,
		Name:      tc.Function.Name,
		Type:      string(tc.Type),
		Arguments: tc.Function.Arguments,
	}
}

// consumeFunctionCall handles the legacy function_call delta. It is assembled
// and validated like a tool call, under the reserved negative position, and
// still delivered to the historical callback as an incrementally grown JSON
// object so that path is unchanged.
func (a *streamAccumulator) consumeFunctionCall(ctx context.Context, fc *FunctionCall) error {
	if a.functionCall == nil {
		a.functionCall = &FunctionCall{Name: fc.Name}
	} else if fc.Name != "" {
		a.functionCall.Name = fc.Name
	}
	a.functionCall.Arguments += fc.Arguments
	if err := a.stream.AddToolCall(ctx, llms.ToolCallDelta{
		Index:     legacyFunctionCallIndex,
		Name:      a.functionCall.Name,
		Arguments: fc.Arguments,
	}); err != nil {
		return &errConsumer{err}
	}
	if a.payload.StreamingFunc != nil {
		chunk := updateFunctionCall(a.response.Choices[0].Message, fc)
		if err := a.payload.StreamingFunc(ctx, chunk); err != nil {
			return fmt.Errorf("streaming func returned an error: %w", err)
		}
	}
	return nil
}

// legacyFunctionCallIndex is the reserved position of the legacy
// function_call, which has no index on the wire.
const legacyFunctionCallIndex = -1

// finish copies the assembled tool calls into the response.
func (a *streamAccumulator) finish() {
	assembled := a.stream.Assembled()
	a.response.Choices[0].Message.ToolCalls = nil
	for _, ac := range assembled {
		c := ac.ToolCall
		if c.FunctionCall == nil {
			continue
		}
		// The legacy function_call is surfaced through Message.FunctionCall
		// only, exactly as before, not as an entry in ToolCalls.
		if ac.Status.Index == legacyFunctionCallIndex {
			continue
		}
		if ac.Status.State != llms.ToolCallStateComplete {
			continue
		}
		a.response.Choices[0].Message.ToolCalls = append(a.response.Choices[0].Message.ToolCalls, ToolCall{
			ID:   c.ID,
			Type: ToolType(c.Type),
			Function: ToolFunction{
				Name:      c.FunctionCall.Name,
				Arguments: c.FunctionCall.Arguments,
			},
		})
	}
	if a.functionCall != nil {
		a.response.Choices[0].Message.FunctionCall = a.functionCall
	}
}

// updateFunctionCall appends a legacy function_call delta to the message and
// returns the incrementally grown call as JSON, which is what the historical
// streaming callback receives on that path.
func updateFunctionCall(message ChatMessage, functionCall *FunctionCall) []byte {
	if message.FunctionCall == nil {
		message.FunctionCall = functionCall
	} else {
		message.FunctionCall.Arguments += functionCall.Arguments
	}
	chunk, _ := json.Marshal(message.FunctionCall) // nolint:errchkjson
	return chunk
}

// StreamingChatResponseTools is a helper function to append tool calls to the stack.
func StreamingChatResponseTools(tools []ToolCall, delta []*ToolCall) ([]byte, []ToolCall) {
	if len(delta) == 0 {
		return []byte{}, tools
	}
	for _, t := range delta {
		// if we have arguments append to the last Tool call
		if t.Type == `` && t.Function.Arguments != `` {
			lindex := len(tools) - 1
			if lindex < 0 {
				continue
			}

			tools[lindex].Function.Arguments += t.Function.Arguments
			continue
		}

		// Otherwise, this is a new tool call, append that to the stack
		tools = append(tools, *t)
	}

	chunk, _ := json.Marshal(delta) // nolint:errchkjson

	return chunk, tools
}
