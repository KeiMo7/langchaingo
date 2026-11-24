package eval

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

const goodJSON = `[
  {
    "id": "q1",
    "question": "How do I reset my password?",
    "reference_answer": "Open Settings, choose Security, and click Reset password.",
    "reference_materials": ["Settings > Security > Reset password"],
    "key_points": ["Settings", "Security", "Reset password"],
    "category": "account"
  },
  {
    "id": "q2",
    "question": "What is the refund window?",
    "reference_answer": "Refunds are available within 30 days.",
    "category": "billing"
  }
]`

func TestLoadSamplesJSONArray(t *testing.T) {
	t.Parallel()
	samples, err := LoadSamplesFromBytes([]byte(goodJSON), FormatAuto)
	require.NoError(t, err)
	require.Len(t, samples, 2)
	require.Equal(t, "q1", samples[0].ID)
	require.Equal(t, "account", samples[0].Category)
	require.Equal(t, []string{"Settings", "Security", "Reset password"}, samples[0].KeyPoints)
	require.Equal(t, "billing", samples[1].EffectiveCategory())
}

func TestLoadSamplesWrapperAndSingleObject(t *testing.T) {
	t.Parallel()
	wrapped := `{"samples": [{"id":"a","question":"q","reference_answer":"r"}]}`
	samples, err := LoadSamplesFromBytes([]byte(wrapped), FormatJSON)
	require.NoError(t, err)
	require.Len(t, samples, 1)

	single := `{"id":"a","question":"q","reference_answer":"r"}`
	samples, err = LoadSamplesFromBytes([]byte(single), FormatJSON)
	require.NoError(t, err)
	require.Equal(t, "a", samples[0].ID)
}

func TestLoadSamplesJSONL(t *testing.T) {
	t.Parallel()
	data := `{"id":"a","question":"q1","reference_answer":"r1"}
{"id":"b","question":"q2","reference_answer":"r2"}

{"id":"c","question":"q3","reference_answer":"r3"}
`
	samples, err := LoadSamplesFromBytes([]byte(data), FormatJSONL)
	require.NoError(t, err)
	require.Len(t, samples, 3)
}

func TestLoadSamplesYAML(t *testing.T) {
	t.Parallel()
	data := `- id: a
  question: q1
  reference_answer: r1
  category: cat
- id: b
  question: q2
  reference_answer: r2
`
	samples, err := LoadSamplesFromBytes([]byte(data), FormatAuto)
	require.NoError(t, err)
	require.Len(t, samples, 2)
	require.Equal(t, "cat", samples[0].Category)
}

func TestLoadSamplesReportsRecordAndField(t *testing.T) {
	t.Parallel()
	data := `[
  {"id":"a","question":"q","reference_answer":"r"},
  {"id":"b","question":"q","reference_answer":123}
]`
	_, err := LoadSamplesFromBytes([]byte(data), FormatJSON)
	require.Error(t, err)
	var le *LoadError
	require.ErrorAs(t, err, &le)
	require.Equal(t, 2, le.Record)
	require.Equal(t, "reference_answer", le.Field)
	require.Contains(t, err.Error(), "record 2")
	require.Contains(t, err.Error(), "reference_answer")

	missing := `[
  {"id":"a","question":"q","reference_answer":"r"},
  {"id":"b","reference_answer":"r"}
]`
	_, err = LoadSamplesFromBytes([]byte(missing), FormatJSON)
	require.ErrorAs(t, err, &le)
	require.Equal(t, 2, le.Record)
	require.Equal(t, "question", le.Field)
}

func TestLoadSamplesJSONLReportsLine(t *testing.T) {
	t.Parallel()
	data := `{"id":"a","question":"q","reference_answer":"r"}
{"id":"b","question":"q"}
`
	_, err := LoadSamplesFromBytes([]byte(data), FormatJSONL)
	require.Error(t, err)
	var le *LoadError
	require.ErrorAs(t, err, &le)
	require.Equal(t, 2, le.Record)
	require.Equal(t, 2, le.Line)
	require.Equal(t, "reference_answer", le.Field)
}

func TestLoadSamplesRejectsEmptyAndDuplicates(t *testing.T) {
	t.Parallel()
	_, err := LoadSamplesFromBytes([]byte(`[]`), FormatJSON)
	require.Error(t, err)

	dup := `[{"id":"a","question":"q","reference_answer":"r"},{"id":"a","question":"q2","reference_answer":"r2"}]`
	_, err = LoadSamplesFromBytes([]byte(dup), FormatJSON)
	require.Error(t, err)
	require.Contains(t, err.Error(), "duplicate")

	garbage := `not a sample file`
	_, err = LoadSamplesFromBytes([]byte(garbage), FormatAuto)
	require.Error(t, err)
}

func TestLoadSamplesFromFile(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "samples.json")
	require.NoError(t, os.WriteFile(path, []byte(goodJSON), 0o600))
	samples, err := LoadSamples(path)
	require.NoError(t, err)
	require.Len(t, samples, 2)

	badPath := filepath.Join(dir, "bad.jsonl")
	require.NoError(t, os.WriteFile(badPath, []byte("{\"id\":\"x\"}\n"), 0o600))
	_, err = LoadSamples(badPath)
	require.Error(t, err)
	require.Contains(t, err.Error(), badPath)
}

func TestNormalizeAndMetrics(t *testing.T) {
	t.Parallel()
	cases := []struct {
		ref, answer string
		exact       bool
	}{
		{"The quick brown fox.", "quick, brown fox", true},
		{"Open the Settings, and reset.", "open Settings and reset", true},
		{"thirty days", "30 days", false},
	}
	for _, tc := range cases {
		m := Score(
			Sample{ReferenceAnswer: tc.ref},
			&Output{Answer: tc.answer},
		)
		require.Equal(t, tc.exact, m.ExactMatch, "%q vs %q", tc.ref, tc.answer)
	}
}

func TestTokenF1AndKeyPointsCJK(t *testing.T) {
	t.Parallel()
	sample := Sample{
		ReferenceAnswer: "重置密码需要进入安全设置",
		KeyPoints:       []string{"安全设置", "重置密码"},
	}
	out := &Output{Answer: "请进入安全设置完成重置密码"}
	m := Score(sample, out)
	require.Equal(t, 1.0, m.KeyPointCoverage)
	require.Greater(t, m.TokenF1, 0.0)
}

func TestTokenF1Partial(t *testing.T) {
	t.Parallel()
	sample := Sample{ReferenceAnswer: "the cat sat on the mat"}
	m := Score(sample, &Output{Answer: "the cat sat on a chair"})
	// Normalization drops the articles, leaving 4 tokens on each side with 3
	// in common: precision and recall are both 3/4, so F1 is 0.75.
	require.InDelta(t, 0.75, m.TokenF1, 0.0001)
	require.False(t, m.ExactMatch)
}
