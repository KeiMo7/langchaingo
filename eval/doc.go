// Package eval evaluates a chain or model against a fixed set of samples,
// so that changes to prompts, models, or retrieval parameters can be judged
// by a reproducible report instead of by spot-checking a handful of answers.
//
// The evaluator is deliberately independent of the types it evaluates. It
// consumes the existing [llms.Model], [chains.Chain], and [schema.Retriever]
// interfaces through small adapters ([ModelTarget], [ChainTarget], and
// [FuncTarget]) and never changes their signatures or behavior.
//
// # Workflow
//
// Load a sample file with [LoadSamples], wrap the system under test in a
// [Target], and call [Evaluate]:
//
//	samples, err := eval.LoadSamples("evaldata/qa.json")
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	target := &eval.ModelTarget{Model: llm}
//	result, err := eval.Evaluate(ctx, samples, target, eval.DefaultConfig())
//	if err != nil {
//	    log.Fatal(err)
//	}
//	if err := result.Report.WriteText(os.Stdout); err != nil {
//	    log.Fatal(err)
//	}
//
// # Capturing usage and retrieval from chains
//
// A model target reads usage and the stop reason straight from the model's
// response. A chain target only receives the chain's output map, so to record
// usage, stop reason, and retriever documents for a chain, attach the
// callback recorder to the handler the chain or model already has:
//
//	llm, _ := openai.New(openai.WithCallback(eval.NewRecorder()))
//	retriever := vectorstores.ToRetriever(store, 4)
//	retriever.CallbacksHandler = eval.NewRecorder()
//
// To keep an application's own handler, combine the two:
//
//	handler := callbacks.CombiningHandler{Callbacks: []callbacks.Handler{
//	    appHandler, eval.NewRecorder(),
//	}}
//
// Without a recorder the chain still runs and is scored; usage and stop
// reason are simply reported as unknown. For retrieval QA, enabling
// chains.RetrievalQA.ReturnSourceDocuments is an alternative that needs no
// callbacks at all.
//
// # Sample files
//
// Samples can be stored as a JSON array, a JSON object with a "samples"
// array, JSON Lines, or YAML. Each record has an "id", "question", and
// "reference_answer"; "reference_materials", "key_points", and "category"
// are optional:
//
//	[
//	  {
//	    "id": "q01",
//	    "question": "How do I reset my password?",
//	    "reference_answer": "Open Settings, choose Security, and click Reset password.",
//	    "reference_materials": ["Settings > Security > Reset password"],
//	    "key_points": ["Settings", "Security", "Reset password"],
//	    "category": "account"
//	  }
//	]
//
// A malformed file fails before any sample is run. The returned [LoadError]
// identifies the 1-based record number (and line, for line-oriented files)
// and, when possible, the offending field.
//
// If "key_points" is omitted, the reference answer itself is the single
// required point. If "category" is omitted, the sample is grouped under
// "uncategorized".
//
// # Metrics
//
// Each successful sample is scored on four axes:
//
//   - exact_match: normalized answers are identical after lowercasing,
//     removing punctuation and English articles, and collapsing whitespace.
//   - token_f1: word-level F1 over normalized tokens (CJK characters are
//     individual tokens).
//   - key_point_coverage: fraction of key points contained in the answer.
//   - retrieval_recall / retrieval_precision: how many reference materials
//     were found in the retrieved documents, and how many retrieved
//     documents matched a reference material. These are only reported when
//     retrieval ran and the sample has reference materials.
//
// Reports aggregate these by category and overall, and always report sample,
// success, failure, and cancellation counts. Metrics that do not apply (for
// example retrieval metrics without a retriever) are reported as unknown and
// excluded from means rather than counted as zero. The same applies to token
// usage that a provider did not report.
//
// # Determinism and caching
//
// Results are collected with bounded concurrency but reported in a fixed
// order: [Evaluation.Results] follows the input order, while the report's
// sample list is sorted by sample ID and its category list by category name.
// Metric means are rounded to four decimal places. [Report.CanonicalBytes]
// omits the volatile section (timestamp, duration, cache counters), so two
// runs over the same inputs are byte-for-byte identical and can be compared
// in CI. [Report.Compare] turns two canonical reports into metric deltas and
// per-sample fixed/regressed outcomes.
//
// When [Config.Cache] is set, responses are cached under a key derived from
// the sample content, the configured model ID, and the prompt version, so a
// prompt change can never hit a response produced by the old prompt.
// [Config.CacheSalt] folds additional configuration (for example retrieval
// parameters) into the key.
//
// # Failures and cancellation
//
// A sample that times out or fails upstream is recorded as failed with an
// error class and the run continues. Retries apply only to transient errors
// (timeouts, rate limits, provider unavailability) and are bounded by
// [Config.MaxRetries]. If the parent context is canceled, no new samples are
// dispatched, in-flight samples are allowed to return, and the report is
// marked incomplete.
package eval
