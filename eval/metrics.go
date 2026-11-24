package eval

import (
	"math"
	"strings"
	"unicode"

	"github.com/tmc/langchaingo/schema"
)

// Uncategorized is the category used for samples without a category label.
const Uncategorized = "uncategorized"

// Score computes the per-sample metrics for an output.
func Score(sample Sample, out *Output) Metrics {
	answer := out.Answer
	normalizedAnswer := normalize(answer)
	m := Metrics{
		ExactMatch:       normalizedAnswer == normalize(sample.ReferenceAnswer),
		TokenF1:          round4(tokenF1(normalize(sample.ReferenceAnswer), normalizedAnswer)),
		KeyPointCoverage: round4(keyPointCoverage(sample.EffectiveKeyPoints(), normalizedAnswer)),
	}
	if out.Retrieved && len(sample.ReferenceMaterials) > 0 {
		m.Retrieval = scoreRetrieval(sample.ReferenceMaterials, out.Documents)
	}
	return m
}

// normalize lowercases text, drops punctuation and English articles, and
// collapses whitespace. It keeps letters and digits of all scripts.
func normalize(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	space := false
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			if space && b.Len() > 0 {
				b.WriteByte(' ')
			}
			space = false
			b.WriteRune(r)
			continue
		}
		// Everything else, including punctuation, symbols, whitespace, and
		// characters this package does not classify, separates tokens. Doing
		// this by exclusion keeps normalization stable across Unicode versions.
		space = true
	}
	return dropArticles(b.String())
}

// dropArticles removes the English indefinite and definite articles when they
// are whole tokens.
func dropArticles(s string) string {
	if s == "" {
		return s
	}
	fields := strings.Fields(s)
	kept := fields[:0]
	for _, f := range fields {
		switch f {
		case "a", "an", "the":
			continue
		}
		kept = append(kept, f)
	}
	return strings.Join(kept, " ")
}

// tokenize splits normalized text into word tokens. CJK runs yield one token
// per character; other scripts group by letters and digits.
func tokenize(s string) []string {
	var tokens []string
	var word strings.Builder
	flush := func() {
		if word.Len() > 0 {
			tokens = append(tokens, word.String())
			word.Reset()
		}
	}
	for _, r := range s {
		switch {
		case isCJK(r):
			flush()
			tokens = append(tokens, string(r))
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			word.WriteRune(r)
		default:
			flush()
		}
	}
	flush()
	return tokens
}

func isCJK(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) ||
		unicode.Is(unicode.Hangul, r)
}

// tokenF1 is the word-level F1 between a reference and an answer.
func tokenF1(normalizedRef, normalizedAnswer string) float64 {
	refTokens := tokenize(normalizedRef)
	answerTokens := tokenize(normalizedAnswer)
	if len(refTokens) == 0 || len(answerTokens) == 0 {
		if len(refTokens) == 0 && len(answerTokens) == 0 {
			return 1
		}
		return 0
	}
	counts := make(map[string]int, len(refTokens))
	for _, t := range refTokens {
		counts[t]++
	}
	common := 0
	for _, t := range answerTokens {
		if counts[t] > 0 {
			counts[t]--
			common++
		}
	}
	if common == 0 {
		return 0
	}
	precision := float64(common) / float64(len(answerTokens))
	recall := float64(common) / float64(len(refTokens))
	return 2 * precision * recall / (precision + recall)
}

// keyPointCoverage is the fraction of key points present in the answer. A
// point is present when its normalized form is a substring of the normalized
// answer, which works for both whitespace-delimited and CJK text.
func keyPointCoverage(keyPoints []string, normalizedAnswer string) float64 {
	points := make([]string, 0, len(keyPoints))
	for _, p := range keyPoints {
		np := normalize(p)
		if np == "" {
			continue
		}
		points = append(points, np)
	}
	if len(points) == 0 {
		return 1
	}
	hit := 0
	for _, p := range points {
		if strings.Contains(normalizedAnswer, p) {
			hit++
		}
	}
	return float64(hit) / float64(len(points))
}

// scoreRetrieval measures how many reference materials the retrieved
// documents contained and how many retrieved documents were usable.
func scoreRetrieval(references []string, docs []schema.Document) *RetrievalMetrics {
	refs := make([]string, 0, len(references))
	for _, r := range references {
		if n := normalize(r); n != "" {
			refs = append(refs, n)
		}
	}
	if len(refs) == 0 {
		return nil
	}
	normalizedDocs := make([]string, len(docs))
	for i, d := range docs {
		normalizedDocs[i] = normalize(d.PageContent)
	}

	matched := make([]bool, len(refs))
	usedDocs := 0
	for _, d := range normalizedDocs {
		used := false
		for i, r := range refs {
			if documentsMatch(d, r) {
				matched[i] = true
				used = true
			}
		}
		if used {
			usedDocs++
		}
	}
	matchedRefs := 0
	for _, ok := range matched {
		if ok {
			matchedRefs++
		}
	}
	m := &RetrievalMetrics{
		MatchedRefs:    matchedRefs,
		ReferenceCount: len(refs),
		RetrievedCount: len(normalizedDocs),
		Recall:         round4(float64(matchedRefs) / float64(len(refs))),
	}
	if len(normalizedDocs) > 0 {
		m.Precision = round4(float64(usedDocs) / float64(len(normalizedDocs)))
	}
	return m
}

// documentsMatch reports whether a document contains the reference material,
// or is near-identical to it. Containment covers a document that quotes the
// reference passage; F1 is a fallback for reworded documents.
func documentsMatch(normalizedDoc, normalizedRef string) bool {
	if normalizedDoc == "" || normalizedRef == "" {
		return false
	}
	if strings.Contains(normalizedDoc, normalizedRef) || strings.Contains(normalizedRef, normalizedDoc) {
		return true
	}
	return tokenF1(normalizedRef, normalizedDoc) >= 0.8
}

// round4 rounds a mean to four decimal places so reports are stable.
func round4(v float64) float64 {
	return math.Round(v*1e4) / 1e4
}

// mean returns the mean of the values, or nil when there are none.
func mean(values []float64) *float64 {
	if len(values) == 0 {
		return nil
	}
	var sum float64
	for _, v := range values {
		sum += v
	}
	m := round4(sum / float64(len(values)))
	return &m
}
