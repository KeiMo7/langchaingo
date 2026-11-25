package llms

import "context"

// WithStreamingHandler installs a provider-neutral event handler. It is
// additive and optional: StreamingFunc, if also set, still receives text
// fragments with unchanged semantics.
func WithStreamingHandler(handler StreamEventFunc) CallOption {
	return func(o *CallOptions) {
		o.StreamingEventFunc = handler
	}
}

// WithStreamingEventFunc is the explicit form of WithStreamingHandler.
func WithStreamingEventFunc(handler StreamEventFunc) CallOption {
	return WithStreamingHandler(handler)
}

// StreamEventFuncFromText adapts a historical text-only callback to an event
// handler. Tool call events are ignored, so text delivered through the
// adapter is byte-identical to what StreamingFunc would have received.
func StreamEventFuncFromText(fn StreamTextFunc) StreamEventFunc {
	if fn == nil {
		return nil
	}
	return func(ctx context.Context, ev StreamEvent) error {
		if ev.Kind != StreamEventText {
			return nil
		}
		return fn(ctx, []byte(ev.Text))
	}
}
