# eval-example

Runs the [`eval`](../../eval) package against a sample file and prints a
report. This is the "did the change make it worse?" check meant for CI: run it
on the same samples before and after a prompt, model, or retrieval change.

## Usage

```console
$ go run . -samples samples.json
$ go run . -samples samples.json -json > report.json
$ go run . -samples samples.json -baseline baseline.json
```

`-baseline` compares the new run against a previously saved canonical report
and exits non-zero when the exact-match rate dropped. That is the gate to put
in a deploy pipeline.

## Sample file

A JSON array; JSON Lines and YAML are also accepted.

```json
[
  {
    "id": "acct-01",
    "question": "How do I reset my password?",
    "reference_answer": "Open Settings, choose Security, and click Reset password.",
    "key_points": ["Settings", "Security", "Reset password"],
    "reference_materials": ["Settings > Security > Reset password"],
    "category": "account"
  }
]
```

## Notes

- `-prompt-version` is part of the response-cache key. Bump it whenever the
  prompt changes, so an -cache run cannot reuse an answer from the old prompt.
- Per-sample failures (timeouts, provider errors) are recorded and the run
  continues.
- Requires `OPENAI_API_KEY`; this example calls the live API.
