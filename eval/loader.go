package eval

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"sigs.k8s.io/yaml"
)

// Format selects how sample bytes are decoded.
type Format string

const (
	// FormatAuto infers the format from the file extension and content.
	FormatAuto Format = ""
	// FormatJSON is a JSON array of samples, or an object with a "samples"
	// array, or a single sample object.
	FormatJSON Format = "json"
	// FormatJSONL is one JSON sample object per line.
	FormatJSONL Format = "jsonl"
	// FormatYAML is YAML in the same shapes as FormatJSON.
	FormatYAML Format = "yaml"
)

// LoadError describes why a sample file could not be used. It identifies the
// 1-based record number, the line for line-oriented formats, and the field
// when the problem is attributable to one.
type LoadError struct {
	Record int    // 1-based record number; 0 when the file as a whole is bad.
	Line   int    // 1-based line, or 0 when not applicable.
	Field  string // field name, when known.
	Err    error
}

// Error implements error.
func (e *LoadError) Error() string {
	var b strings.Builder
	b.WriteString("eval: sample file: ")
	if e.Record > 0 {
		fmt.Fprintf(&b, "record %d", e.Record)
		if e.Line > 0 {
			fmt.Fprintf(&b, " (line %d)", e.Line)
		}
	} else if e.Line > 0 {
		fmt.Fprintf(&b, "line %d", e.Line)
	} else {
		b.WriteString("invalid file")
	}
	if e.Field != "" {
		fmt.Fprintf(&b, ": field %q", e.Field)
	}
	if e.Err != nil {
		b.WriteString(": ")
		b.WriteString(e.Err.Error())
	}
	return b.String()
}

// Unwrap returns the underlying cause.
func (e *LoadError) Unwrap() error { return e.Err }

// LoadSamples reads and validates samples from a file. The file format is
// inferred from the extension and content. A malformed file is rejected as a
// whole, before any evaluation starts.
func LoadSamples(path string) ([]Sample, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("eval: read sample file %s: %w", path, err)
	}
	format := formatForPath(path)
	samples, err := LoadSamplesFromBytes(data, format)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return samples, nil
}

// LoadSamplesFromBytes decodes samples from data.
func LoadSamplesFromBytes(data []byte, format Format) ([]Sample, error) {
	if format == FormatAuto {
		format = detectFormat(data)
	}
	var (
		samples []Sample
		err     error
	)
	switch format {
	case FormatJSON:
		samples, err = loadJSON(data)
	case FormatJSONL:
		samples, err = loadJSONL(data)
	case FormatYAML:
		samples, err = loadYAML(data)
	default:
		return nil, &LoadError{Err: fmt.Errorf("unsupported format %q", format)}
	}
	if err != nil {
		return nil, err
	}
	if err := validateLoaded(samples); err != nil {
		return nil, err
	}
	return samples, nil
}

func formatForPath(path string) Format {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".json":
		return FormatJSON
	case ".jsonl", ".ndjson":
		return FormatJSONL
	case ".yaml", ".yml":
		return FormatYAML
	default:
		return FormatAuto
	}
}

// detectFormat distinguishes a single JSON value from JSON Lines and YAML. A
// file is JSON when it holds one complete JSON value; it is JSON Lines when
// every non-empty line is a complete JSON value; otherwise it is tried as
// YAML.
func detectFormat(data []byte) Format {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return FormatJSON
	}
	if json.Valid(trimmed) {
		return FormatJSON
	}
	if linesAreJSON(data) {
		return FormatJSONL
	}
	return FormatYAML
}

func linesAreJSON(data []byte) bool {
	seen := false
	for _, line := range bytes.Split(data, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 || bytes.HasPrefix(line, []byte("#")) {
			continue
		}
		seen = true
		if !json.Valid(line) {
			return false
		}
	}
	return seen
}

// loadJSON accepts an array, a wrapper object with "samples", or a single
// sample object.
func loadJSON(data []byte) ([]Sample, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return nil, &LoadError{Err: errors.New("empty file")}
	}
	switch trimmed[0] {
	case '[':
		var records []json.RawMessage
		if err := json.Unmarshal(trimmed, &records); err != nil {
			return nil, &LoadError{Err: fmt.Errorf("invalid JSON array: %w", err)}
		}
		return decodeRecords(records)
	case '{':
		var wrapper struct {
			Samples []json.RawMessage `json:"samples"`
		}
		if err := json.Unmarshal(trimmed, &wrapper); err != nil {
			return nil, &LoadError{Err: fmt.Errorf("invalid JSON object: %w", err)}
		}
		if wrapper.Samples != nil {
			return decodeRecords(wrapper.Samples)
		}
		return decodeRecords([]json.RawMessage{trimmed})
	default:
		return nil, &LoadError{Err: errors.New("expected a JSON array or object")}
	}
}

// loadYAML converts YAML to JSON and reuses the JSON decoding paths, so both
// formats share validation and error reporting.
func loadYAML(data []byte) ([]Sample, error) {
	jsonData, err := yaml.YAMLToJSON(data)
	if err != nil {
		return nil, &LoadError{Err: fmt.Errorf("invalid YAML: %w", err)}
	}
	return loadJSON(jsonData)
}

// loadJSONL decodes one sample per non-empty line. It validates each record
// as it goes, so errors can name the physical line.
func loadJSONL(data []byte) ([]Sample, error) {
	var (
		samples []Sample
		record  int
	)
	for lineNo, line := range bytes.Split(data, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 || bytes.HasPrefix(line, []byte("#")) {
			continue
		}
		record++
		var s Sample
		if err := unmarshalSample(line, &s); err != nil {
			return nil, recordError(record, lineNo+1, err)
		}
		if err := validateRecord(record, s); err != nil {
			err.Line = lineNo + 1
			return nil, err
		}
		samples = append(samples, s)
	}
	if len(samples) == 0 {
		return nil, &LoadError{Err: errors.New("no samples found")}
	}
	return samples, nil
}

// decodeRecords decodes raw records, attributing errors to a record number.
func decodeRecords(records []json.RawMessage) ([]Sample, error) {
	if len(records) == 0 {
		return nil, &LoadError{Err: errors.New("no samples found")}
	}
	samples := make([]Sample, 0, len(records))
	for i, raw := range records {
		var s Sample
		if err := unmarshalSample(raw, &s); err != nil {
			return nil, recordError(i+1, 0, err)
		}
		samples = append(samples, s)
	}
	return samples, nil
}

func unmarshalSample(raw []byte, s *Sample) error {
	if len(bytes.TrimSpace(raw)) == 0 {
		return errors.New("empty record")
	}
	return json.Unmarshal(raw, s)
}

// recordError converts a decoding error into a LoadError. It names the field
// when the decoder identified one.
func recordError(record, line int, err error) *LoadError {
	e := &LoadError{Record: record, Line: line, Err: err}
	var typeErr *json.UnmarshalTypeError
	if errors.As(err, &typeErr) && typeErr.Field != "" {
		e.Field = typeErr.Field
		e.Err = fmt.Errorf("expected %s, got %s", typeErr.Type, typeErr.Value)
	}
	return e
}

// validateRecord checks one sample's required fields. JSON Lines records go
// through this per line, so the caller can add the line number.
func validateRecord(record int, s Sample) *LoadError {
	switch {
	case s.ID == "":
		return &LoadError{Record: record, Field: "id", Err: errors.New("required")}
	case s.Question == "":
		return &LoadError{Record: record, Field: "question", Err: errors.New("required")}
	case s.ReferenceAnswer == "":
		return &LoadError{Record: record, Field: "reference_answer", Err: errors.New("required")}
	}
	return nil
}

// validateLoaded checks loaded samples for problems that make a report
// ambiguous. The record number in the error matches the file order.
func validateLoaded(samples []Sample) error {
	if len(samples) == 0 {
		return &LoadError{Err: errors.New("no samples found")}
	}
	seen := make(map[string]int, len(samples))
	for i, s := range samples {
		record := i + 1
		if err := validateRecord(record, s); err != nil {
			return err
		}
		if prev, ok := seen[s.ID]; ok {
			return &LoadError{
				Record: record,
				Field:  "id",
				Err:    fmt.Errorf("duplicate of record %d (%q)", prev, s.ID),
			}
		}
		seen[s.ID] = record
	}
	return nil
}
