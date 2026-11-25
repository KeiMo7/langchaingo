package llms

import (
	"context"
	"fmt"
)

// StreamEventKind identifies one increment of a streaming model response.
//
// A single stream carries text (and reasoning) increments and tool call
// increments with the same delivery semantics: every event is produced in
// arrival order, exactly once, and never with an empty payload.
type StreamEventKind int

const (
	// StreamEventText is a fragment of assistant text. Text is set.
	StreamEventText StreamEventKind = iota + 1
	// StreamEventReasoning is a fragment of reasoning/thinking content.
	StreamEventReasoning
	// StreamEventToolCallStart announces a tool call. Index, ToolCallID
	// and (when the provider supplies it) ToolName identify it.
	StreamEventToolCallStart
	// StreamEventToolCallDelta adds ArgumentsFragment to the call at Index.
	StreamEventToolCallDelta
	// StreamEventToolCallDone reports that a call reached its terminal
	// state. Status carries the outcome; ToolCall carries the assembled
	// call when Status.State is ToolCallStateComplete.
	StreamEventToolCallDone
	// StreamEventWarning reports a fragment that was ignored or repaired.
	// A warning never terminates the stream.
	StreamEventWarning
	// StreamEventEnd is the terminal summary. It is delivered at most once
	// per stream and never after the context is cancelled or the handler
	// returned an error.
	StreamEventEnd
)

// String returns the stable name of the event kind.
func (k StreamEventKind) String() string {
	switch k {
	case StreamEventText:
		return "text"
	case StreamEventReasoning:
		return "reasoning"
	case StreamEventToolCallStart:
		return "tool_call_start"
	case StreamEventToolCallDelta:
		return "tool_call_delta"
	case StreamEventToolCallDone:
		return "tool_call_done"
	case StreamEventWarning:
		return "warning"
	case StreamEventEnd:
		return "end"
	default:
		return fmt.Sprintf("StreamEventKind(%d)", int(k))
	}
}

// ToolCallState is the terminal state of one assembled tool call.
type ToolCallState string

const (
	// ToolCallStateIncomplete means the stream ended or failed while the
	// call was still being assembled, or its arguments never formed a
	// complete JSON object. The call must not be executed.
	ToolCallStateIncomplete ToolCallState = "incomplete"
	// ToolCallStateComplete means the arguments parsed as a JSON object
	// (or array) and the call may be executed.
	ToolCallStateComplete ToolCallState = "complete"
)

// String returns the state name.
func (s ToolCallState) String() string { return string(s) }

// StreamStopReason explains why a stream ended.
type StreamStopReason string

const (
	// StreamStopReasonComplete means the upstream stream ended normally.
	StreamStopReasonComplete StreamStopReason = "complete"
	// StreamStopReasonUpstreamError means the provider reported an error
	// inside the stream.
	StreamStopReasonUpstreamError StreamStopReason = "upstream_error"
	// StreamStopReasonCanceled means the context was cancelled.
	StreamStopReasonCanceled StreamStopReason = "canceled"
	// StreamStopReasonConsumerError means the event or text handler
	// returned an error.
	StreamStopReasonConsumerError StreamStopReason = "consumer_error"
)

// String returns the reason name.
func (r StreamStopReason) String() string { return string(r) }

// StreamEvent is one increment of a streaming response.
//
// Only the fields named by Kind are meaningful; the rest are zero. An event
// always carries one increment - the assembler never emits an empty text or
// tool delta.
type StreamEvent struct {
	// Kind selects which of the remaining fields are set.
	Kind StreamEventKind
	// Text is the fragment for StreamEventText and StreamEventReasoning.
	Text string
	// Index is the position of the tool call within the response.
	Index int
	// ToolCallID is the stable identifier of the tool call. It is empty
	// only when the provider never supplied one (position-only calls).
	ToolCallID string
	// ToolName is the function name of the tool call, when known.
	ToolName string
	// ArgumentsFragment is the raw, undecoded arguments text added by a
	// StreamEventToolCallDelta. It is never echoed into logs by default.
	ArgumentsFragment string
	// ArgumentsLength is the total length of the accumulated arguments
	// after this event. It lets an observer gauge progress without
	// seeing the arguments body.
	ArgumentsLength int
	// ToolCall is the assembled call, set on StreamEventToolCallDone only
	// when Status.State is ToolCallStateComplete.
	ToolCall *ToolCall
	// Status is the per-call outcome, set on StreamEventToolCallDone only.
	Status *ToolCallStatus
	// Summary is the terminal summary, set on StreamEventEnd only.
	Summary *StreamSummary
	// Warning describes an ignored or repaired fragment, set on
	// StreamEventWarning only.
	Warning *StreamWarning
}

// String renders the event without exposing tool argument bodies.
func (e StreamEvent) String() string {
	switch e.Kind {
	case StreamEventText:
		return fmt.Sprintf("text(%d bytes)", len(e.Text))
	case StreamEventReasoning:
		return fmt.Sprintf("reasoning(%d bytes)", len(e.Text))
	case StreamEventToolCallStart:
		return fmt.Sprintf("tool_call_start(index=%d id=%q name=%q)", e.Index, e.ToolCallID, e.ToolName)
	case StreamEventToolCallDelta:
		return fmt.Sprintf("tool_call_delta(index=%d id=%q args=%d/%d)", e.Index, e.ToolCallID,
			len(e.ArgumentsFragment), e.ArgumentsLength)
	case StreamEventToolCallDone:
		state := ToolCallState("")
		if e.Status != nil {
			state = e.Status.State
		}
		return fmt.Sprintf("tool_call_done(index=%d id=%q state=%s)", e.Index, e.ToolCallID, state)
	case StreamEventWarning:
		code := ""
		if e.Warning != nil {
			code = e.Warning.Code
		}
		return fmt.Sprintf("warning(code=%s index=%d)", code, e.Index)
	case StreamEventEnd:
		if e.Summary == nil {
			return "end(<nil>)"
		}
		return "end(" + e.Summary.String() + ")"
	default:
		return "event(<unknown>)"
	}
}

// ToolCallStatus is the outcome of one tool call.
type ToolCallStatus struct {
	// Index is the position of the call within the response.
	Index int
	// ID is the call identifier, empty for position-only calls.
	ID string
	// Name is the tool name, when known.
	Name string
	// State is the terminal state of the call.
	State ToolCallState
	// ErrorOffset is the byte offset in the accumulated arguments at
	// which structural parsing failed. It is set only when State is
	// ToolCallStateIncomplete because the arguments did not parse.
	ErrorOffset int
	// Error describes why the call is not complete, if it is not.
	Error string
	// ArgumentsLength is the length of the accumulated arguments.
	ArgumentsLength int
}

// String renders the status without exposing the arguments body.
func (s ToolCallStatus) String() string {
	return fmt.Sprintf("tool_call(index=%d id=%q name=%q state=%s args=%d)",
		s.Index, s.ID, s.Name, s.State, s.ArgumentsLength)
}

// Executable reports whether the call may be handed to a tool.
func (s ToolCallStatus) Executable() bool { return s.State == ToolCallStateComplete }

// StreamSummary is the terminal summary of one stream. It is produced once.
type StreamSummary struct {
	// StopReason says why the stream ended.
	StopReason StreamStopReason
	// TextLength is the number of text bytes delivered.
	TextLength int
	// ReasoningLength is the number of reasoning bytes delivered.
	ReasoningLength int
	// Calls is the terminal status of every tool call the stream saw, in
	// position order.
	Calls []ToolCallStatus
	// Error is the error that ended the stream, if any.
	Error string
}

// String renders the summary without exposing tool argument bodies.
func (s StreamSummary) String() string {
	complete, incomplete := 0, 0
	for _, c := range s.Calls {
		switch c.State {
		case ToolCallStateComplete:
			complete++
		default:
			incomplete++
		}
	}
	return fmt.Sprintf("reason=%s text=%d calls=%d complete=%d incomplete=%d",
		s.StopReason, s.TextLength, len(s.Calls), complete, incomplete)
}

// CompleteCalls returns the statuses of the calls that may be executed, in
// position order.
func (s StreamSummary) CompleteCalls() []ToolCallStatus {
	out := make([]ToolCallStatus, 0, len(s.Calls))
	for _, c := range s.Calls {
		if c.State == ToolCallStateComplete {
			out = append(out, c)
		}
	}
	return out
}

// IncompleteCalls returns the statuses of the calls that must not be
// executed, in position order.
func (s StreamSummary) IncompleteCalls() []ToolCallStatus {
	out := make([]ToolCallStatus, 0)
	for _, c := range s.Calls {
		if c.State != ToolCallStateComplete {
			out = append(out, c)
		}
	}
	return out
}

// StreamWarning describes a fragment the assembler ignored or repaired.
type StreamWarning struct {
	// Code is a stable identifier for the warning class.
	Code string
	// Index is the affected tool call position, or -1 if not applicable.
	Index int
	// Detail is a human-readable description that never contains the
	// arguments body.
	Detail string
}

const (
	// StreamWarningDuplicateFragment reports a fragment that repeated an
	// already accumulated fragment and was dropped.
	StreamWarningDuplicateFragment = "duplicate_fragment"
	// StreamWarningCumulativeArguments reports a fragment that re-sent
	// the accumulated arguments and was merged instead of appended.
	StreamWarningCumulativeArguments = "cumulative_arguments"
	// StreamWarningUnknownFragment reports a fragment that matched no
	// known tool call and was ignored.
	StreamWarningUnknownFragment = "unknown_fragment"
	// StreamWarningIdentityConflict reports a fragment whose position or
	// identifier disagreed with the call it would have joined.
	StreamWarningIdentityConflict = "identity_conflict"
	// StreamWarningUnknownChunk reports provider chunk content the
	// adapter did not understand.
	StreamWarningUnknownChunk = "unknown_chunk"
)

// StreamEventFunc receives every event of a stream in arrival order. It must
// not retain the event's string fields beyond the call. Returning an error
// stops the stream and is returned from the model call together with the
// partial response.
type StreamEventFunc func(ctx context.Context, event StreamEvent) error

// ToolCallDelta is one provider-neutral tool call fragment.
//
// Identity is positional plus identifier: Index locates the call, ID (when
// non-empty) must agree with the call already at that position. A delta with
// only Arguments set extends the call at Index. A delta with a new ID at a
// new Index opens a call.
type ToolCallDelta struct {
	// Index is the position of the call within the response.
	Index int
	// ID is the call identifier, when the provider supplies one.
	ID string
	// Name is the function name, when the provider supplies it.
	Name string
	// Type is the provider's tool type ("function"). It is informational
	// only: merge rules never depend on it.
	Type string
	// Arguments is the raw, undecoded arguments fragment.
	Arguments string
}

// StreamTextFunc is the text-only streaming callback. It is the historical
// StreamingFunc type and keeps its exact behavior.
type StreamTextFunc = func(ctx context.Context, chunk []byte) error

// AssembledCall is a completed tool call together with its status.
type AssembledCall struct {
	ToolCall ToolCall
	Status   ToolCallStatus
}
