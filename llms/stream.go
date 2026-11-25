package llms

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"
)

// Stream assembles model response fragments and reports them as
// provider-neutral events.
//
// A provider adapter creates one Stream per model call and feeds it the
// fragments decoded from that provider's wire format. The adapter keeps
// owning the transport (reading the socket, the SSE framing, cancellation);
// the Stream owns the semantics that must be identical across providers: text
// and tool call increments from one stream, per-call identity, deduplication,
// executability, lifecycle and the terminal summary.
//
// A Stream is single-use and is not safe for concurrent use. It holds no
// package-level state, so concurrent model calls are independent.
type Stream struct {
	provider string
	emit     StreamEventFunc

	text       strings.Builder
	reasoning  strings.Builder
	calls      []*streamCall
	byID       map[string]*streamCall
	ended      bool
	finishOnce bool
	emitErr    error
}

type streamCall struct {
	index int
	id    string
	name  string
	args  strings.Builder

	done   bool
	status ToolCallStatus
}

// NewStream returns a Stream that reports events through emit. provider names
// the adapter for warning logs; it may be empty. emit may be nil when the
// caller only needs the assembled result.
func NewStream(provider string, emit StreamEventFunc) *Stream {
	return &Stream{provider: provider, emit: emit, byID: make(map[string]*streamCall)}
}

// Text records and emits one text fragment. Empty fragments are dropped, so a
// stream never delivers an empty text event.
func (s *Stream) Text(ctx context.Context, fragment string) error {
	if fragment == "" || s.failed() {
		return nil
	}
	if err := s.send(ctx, StreamEvent{Kind: StreamEventText, Text: fragment}); err != nil {
		return err
	}
	s.text.WriteString(fragment)
	return nil
}

// Reasoning records and emits one reasoning fragment. Empty fragments are
// dropped. This is the separate channel reasoning models already use; the
// adapter decides whether reasoning is also mirrored to StreamingFunc.
func (s *Stream) Reasoning(ctx context.Context, fragment string) error {
	if fragment == "" || s.failed() {
		return nil
	}
	if err := s.send(ctx, StreamEvent{Kind: StreamEventReasoning, Text: fragment}); err != nil {
		return err
	}
	s.reasoning.WriteString(fragment)
	return nil
}

// AddToolCall routes one provider-neutral tool call fragment.
//
// A fragment opens a call when its position is new; otherwise it extends the
// call already there. An identifier, once seen, is authoritative. A fragment
// that repeats or re-sends arguments already accumulated is merged rather
// than appended, so the assembled arguments are valid by construction.
func (s *Stream) AddToolCall(ctx context.Context, d ToolCallDelta) error {
	if s.failed() {
		return nil
	}
	c := s.resolve(ctx, d)
	if c == nil {
		return nil
	}
	if c.done {
		// A completed call never reopens and never completes twice.
		if d.Arguments == "" || strings.HasPrefix(c.args.String(), d.Arguments) {
			return nil
		}
		return s.warn(ctx, StreamWarning{
			Code:   StreamWarningIdentityConflict,
			Index:  c.index,
			Detail: "fragment for a completed call was dropped",
		})
	}

	suffix, kind := mergeArguments(c.args.String(), d.Arguments)
	switch kind {
	case mergeDuplicate:
		return s.warn(ctx, StreamWarning{Code: StreamWarningDuplicateFragment, Index: c.index})
	case mergeCumulative:
		if suffix == "" {
			return s.warn(ctx, StreamWarning{Code: StreamWarningCumulativeArguments, Index: c.index})
		}
		if err := s.warn(ctx, StreamWarning{Code: StreamWarningCumulativeArguments, Index: c.index}); err != nil {
			return err
		}
	case mergeConflict:
		return s.warn(ctx, StreamWarning{
			Code:   StreamWarningIdentityConflict,
			Index:  c.index,
			Detail: "fragment did not form valid arguments and was dropped",
		})
	case mergeAppend:
	}
	if suffix == "" {
		return nil
	}
	// Deliver first, commit second: a consumer error must leave the
	// assembled state exactly as the consumer last observed it, so a
	// partial result never contains bytes no event announced.
	if err := s.send(ctx, StreamEvent{
		Kind:              StreamEventToolCallDelta,
		Index:             c.index,
		ToolCallID:        c.id,
		ToolName:          c.name,
		ArgumentsFragment: suffix,
		ArgumentsLength:   c.args.Len() + len(suffix),
	}); err != nil {
		return err
	}
	c.args.WriteString(suffix)
	return nil
}

// Warning emits a warning about a fragment the adapter ignored or repaired.
// Warnings never terminate the stream.
func (s *Stream) Warning(ctx context.Context, w StreamWarning) error {
	if s.failed() {
		return nil
	}
	return s.warn(ctx, w)
}

func (s *Stream) warn(ctx context.Context, w StreamWarning) error {
	if w.Code == "" {
		return nil
	}
	logStreamWarning(s.provider, w)
	return s.send(ctx, StreamEvent{Kind: StreamEventWarning, Index: w.Index, Warning: &w})
}

func (s *Stream) resolve(ctx context.Context, d ToolCallDelta) *streamCall {
	// An identifier already seen is authoritative even if the reported
	// position slipped: identity wins, and routing by ID keeps one call
	// from being split.
	if d.ID != "" {
		if existing, ok := s.byID[d.ID]; ok {
			if existing.index != d.Index {
				_ = s.warn(ctx, StreamWarning{
					Code:   StreamWarningIdentityConflict,
					Index:  existing.index,
					Detail: "fragment position disagreed with its identifier",
				})
			}
			s.mergeIdentity(existing, d)
			return existing
		}
	}

	c := s.at(d.Index)
	switch {
	case c == nil:
		if d.ID == "" && d.Name == "" && d.Arguments == "" {
			_ = s.warn(ctx, StreamWarning{
				Code:   StreamWarningUnknownFragment,
				Index:  d.Index,
				Detail: "fragment carried no identifier, name or arguments",
			})
			return nil
		}
		return s.open(ctx, d)
	case d.ID != "" && c.id == "":
		// The identifier arrived after the opening fragment. Bind it to
		// the call already open at this position.
		c.id = d.ID
		s.byID[d.ID] = c
	case d.ID != "" && c.id != d.ID:
		_ = s.warn(ctx, StreamWarning{
			Code:   StreamWarningIdentityConflict,
			Index:  c.index,
			Detail: fmt.Sprintf("fragment id %q conflicts with %q", d.ID, c.id),
		})
		return nil
	}
	s.mergeIdentity(c, d)
	return c
}

func (s *Stream) mergeIdentity(c *streamCall, d ToolCallDelta) {
	if c.name == "" && d.Name != "" {
		c.name = d.Name
	}
	if c.id == "" && d.ID != "" {
		c.id = d.ID
		s.byID[d.ID] = c
	}
}

func (s *Stream) at(index int) *streamCall {
	for _, c := range s.calls {
		if c.index == index {
			return c
		}
	}
	return nil
}

// open announces a new call and emits its Start event. It does not consume the
// opening fragment's arguments; AddToolCall appends those through the normal
// merge path so every fragment follows one code path.
func (s *Stream) open(ctx context.Context, d ToolCallDelta) *streamCall {
	c := &streamCall{index: d.Index, id: d.ID, name: d.Name}
	s.calls = append(s.calls, c)
	if c.id != "" {
		s.byID[c.id] = c
	}
	if err := s.send(ctx, StreamEvent{
		Kind:       StreamEventToolCallStart,
		Index:      c.index,
		ToolCallID: c.id,
		ToolName:   c.name,
	}); err != nil {
		return nil
	}
	return c
}

// Finish finalizes every open call and emits its Done event exactly once.
// A call is complete when it has a name and its accumulated arguments parse
// as a JSON object or array; everything else is incomplete and must not be
// executed.
func (s *Stream) Finish(ctx context.Context) error {
	if s.finishOnce {
		return nil
	}
	for _, c := range s.calls {
		if c.done {
			continue
		}
		status := c.evaluate()
		ev := StreamEvent{
			Kind:       StreamEventToolCallDone,
			Index:      c.index,
			ToolCallID: c.id,
			ToolName:   c.name,
			Status:     &status,
		}
		if status.State == ToolCallStateComplete {
			call := c.toolCall()
			ev.ToolCall = &call
		}
		if err := s.send(ctx, ev); err != nil {
			// Leave finishOnce unset so a later Close can finish the
			// calls that were not reached; delivery is already latched
			// off, so no further event escapes.
			return err
		}
		// Commit the terminal state only after the observer has seen it,
		// so the summary always matches what was delivered.
		c.done = true
		c.status = status
	}
	s.finishOnce = true
	return nil
}

func (c *streamCall) evaluate() ToolCallStatus {
	st := ToolCallStatus{
		Index:           c.index,
		ID:              c.stableID(),
		Name:            c.name,
		ArgumentsLength: c.args.Len(),
	}
	if c.name == "" {
		st.State = ToolCallStateIncomplete
		st.Error = "tool call has no name"
		return st
	}
	args := c.args.String()
	if strings.TrimSpace(args) == "" {
		st.State = ToolCallStateIncomplete
		st.Error = "tool call has no arguments"
		return st
	}
	var probe any
	if err := json.Unmarshal([]byte(args), &probe); err != nil {
		st.State = ToolCallStateIncomplete
		st.Error = err.Error()
		st.ErrorOffset = jsonErrorOffset(err, len(args))
		return st
	}
	switch probe.(type) {
	case map[string]any, []any:
		st.State = ToolCallStateComplete
	default:
		st.State = ToolCallStateIncomplete
		st.Error = "arguments are not a JSON object or array"
	}
	return st
}

// stableID returns the call's identifier, synthesizing a deterministic one from
// its position when the provider never supplied an identifier. It is the same
// value reported by events, statuses and the assembled call.
func (c *streamCall) stableID() string {
	if c.id != "" {
		return c.id
	}
	// The legacy function_call has no position and no identity of its own.
	if c.index < 0 {
		return ""
	}
	return fmt.Sprintf("call_%d", c.index)
}

func (c *streamCall) toolCall() ToolCall {
	return ToolCall{
		ID:   c.stableID(),
		Type: "function",
		FunctionCall: &FunctionCall{
			Name:      c.name,
			Arguments: c.args.String(),
		},
	}
}

func (s *Stream) statusOf(c *streamCall) ToolCallStatus {
	if c.done {
		return c.status
	}
	return c.evaluate()
}

// Assembled returns every call seen so far with its current status, in
// position order. It emits no events, so it is safe to call after a consumer
// error stopped delivery.
func (s *Stream) Assembled() []AssembledCall {
	out := make([]AssembledCall, 0, len(s.calls))
	for _, c := range s.calls {
		out = append(out, AssembledCall{ToolCall: c.toolCall(), Status: s.statusOf(c)})
	}
	sortAssembled(out)
	return out
}

// CompleteCalls returns the calls whose arguments are structurally valid, in
// position order. Incomplete calls are omitted, so the result is safe to hand
// to tools.
func (s *Stream) CompleteCalls() []ToolCall {
	assembled := s.Assembled()
	out := make([]ToolCall, 0, len(assembled))
	for _, c := range assembled {
		if c.Status.State == ToolCallStateComplete {
			out = append(out, c.ToolCall)
		}
	}
	return out
}

// IncompleteCalls returns the calls that must not be executed, in position
// order.
func (s *Stream) IncompleteCalls() []ToolCallStatus {
	assembled := s.Assembled()
	out := make([]ToolCallStatus, 0)
	for _, c := range assembled {
		if c.Status.State != ToolCallStateComplete {
			out = append(out, c.Status)
		}
	}
	return out
}

// AccumulatedText returns the text accumulated so far.
func (s *Stream) AccumulatedText() string { return s.text.String() }

// AccumulatedReasoning returns the reasoning accumulated so far.
func (s *Stream) AccumulatedReasoning() string { return s.reasoning.String() }

// Summary returns the terminal summary for the given reason. It is
// deterministic for a given sequence of fragments and never includes
// argument bodies.
func (s *Stream) Summary(reason StreamStopReason, err error) StreamSummary {
	sum := StreamSummary{
		StopReason:      reason,
		TextLength:      s.text.Len(),
		ReasoningLength: s.reasoning.Len(),
	}
	if err != nil {
		sum.Error = err.Error()
	}
	assembled := s.Assembled()
	sum.Calls = make([]ToolCallStatus, len(assembled))
	for i, c := range assembled {
		sum.Calls[i] = c.Status
	}
	return sum
}

// Close finishes any open calls and emits the terminal summary exactly once,
// under the given reason. It is safe to call more than once and from a
// cancellation path: only the first call delivers anything, so the summary
// cannot appear twice.
//
// closeCtx bounds best-effort delivery of the remaining events. When the
// stream is ending because its own context was cancelled, pass a context that
// is still live if the caller wants the summary delivered; the summary is
// returned either way.
func (s *Stream) Close(closeCtx context.Context, reason StreamStopReason, err error) StreamSummary {
	// The summary is computed from the assembled state, independent of
	// whether any remaining events can still be delivered.
	if s.ended {
		return s.Summary(reason, err)
	}
	s.ended = true
	// A consumer error already stopped delivery; do not push further
	// events at it. The summary is still returned, and the assembler's
	// state is complete, so the caller keeps the partial result.
	if s.emit != nil && s.emitErr == nil && closeCtx.Err() == nil {
		_ = s.Finish(closeCtx)
		sum := s.Summary(reason, err)
		_ = s.emit(closeCtx, StreamEvent{Kind: StreamEventEnd, Summary: &sum})
		return sum
	}
	return s.Summary(reason, err)
}

func (s *Stream) failed() bool { return s.ended || s.emitErr != nil }

func (s *Stream) send(ctx context.Context, ev StreamEvent) error {
	if s.emitErr != nil {
		// A consumer error already stopped delivery. Later fragments are
		// ignored rather than re-delivered, and the recorded error is
		// what the model call reports.
		return s.emitErr
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if s.emit == nil {
		return nil
	}
	if err := s.emit(ctx, ev); err != nil {
		s.emitErr = err
		return err
	}
	return nil
}

type mergeKind int

const (
	mergeAppend mergeKind = iota
	mergeDuplicate
	mergeCumulative
	mergeConflict
)

// mergeArguments decides how a new fragment combines with what has been
// accumulated, returning the bytes to append.
//
// A fragment that re-sends or repeats the accumulated arguments is merged, not
// appended. The remaining fragments are appended verbatim when doing so leaves
// a valid JSON prefix; that keeps every byte the provider meant to send
// (including a legitimately repeated fragment such as the "aa" in
// {"a":"aa"}) and makes the assembled arguments valid by construction.
func mergeArguments(acc, frag string) (string, mergeKind) {
	if frag == "" {
		return "", mergeDuplicate
	}
	if acc == "" {
		return frag, mergeAppend
	}
	if frag == acc {
		return "", mergeDuplicate
	}
	if strings.HasPrefix(frag, acc) && len(frag) > len(acc) {
		// The upstream re-sent the accumulated arguments with more on
		// top. Keep only the bytes beyond what is already there.
		return frag[len(acc):], mergeCumulative
	}
	if isJSONObjectPrefix(acc + frag) {
		return frag, mergeAppend
	}
	if strings.HasSuffix(acc, frag) {
		return "", mergeDuplicate
	}
	// The fragment alone may be a valid prefix, but appending it would
	// produce invalid arguments. Dropping it keeps the assembled arguments
	// valid; the caller's retry policy decides whether to ask again.
	return "", mergeConflict
}

// isJSONObjectPrefix reports whether s is a syntactically valid prefix of a
// single JSON value: every token so far is well formed and the only error is
// that the input ends early. Byte splits inside a multibyte character land in
// the middle of a string token, which the tokenizer accepts, so fragments may
// be cut at any byte boundary.
func isJSONObjectPrefix(s string) bool {
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()
	depth := 0
	seenRoot := false
	for {
		tok, err := dec.Token()
		if errors.Is(err, io.EOF) {
			return true
		}
		if err != nil {
			return isJSONTruncation(err)
		}
		if d, ok := tok.(json.Delim); ok {
			switch d {
			case '{', '[':
				depth++
				if depth == 1 {
					seenRoot = true
				}
			case '}', ']':
				depth--
			}
		}
		if seenRoot && depth == 0 {
			if _, err := dec.Token(); errors.Is(err, io.EOF) {
				return true
			}
			return false
		}
	}
}

func isJSONTruncation(err error) bool {
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	var se *json.SyntaxError
	return errors.As(err, &se) && strings.Contains(se.Error(), "unexpected end of JSON input")
}

func jsonErrorOffset(err error, total int) int {
	var se *json.SyntaxError
	if errors.As(err, &se) {
		if int(se.Offset) > total {
			return total
		}
		return int(se.Offset)
	}
	var te *json.UnmarshalTypeError
	if errors.As(err, &te) {
		if int(te.Offset) > total {
			return total
		}
		return int(te.Offset)
	}
	return total
}

func sortAssembled(calls []AssembledCall) {
	// Insertion sort keeps the helper dependency-free and stable.
	for i := 1; i < len(calls); i++ {
		for j := i; j > 0 && calls[j-1].Status.Index > calls[j].Status.Index; j-- {
			calls[j-1], calls[j] = calls[j], calls[j-1]
		}
	}
}

// logStreamWarning records a warning without exposing argument bodies. The
// message carries only codes, positions, identifiers and lengths.
func logStreamWarning(provider string, w StreamWarning) {
	if w.Code == "" {
		return
	}
	attrs := []any{"code", w.Code, "index", w.Index}
	if provider != "" {
		attrs = append(attrs, "provider", provider)
	}
	if w.Detail != "" {
		attrs = append(attrs, "detail", w.Detail)
	}
	slog.Warn("stream: fragment ignored", attrs...)
}
