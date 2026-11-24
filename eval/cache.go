package eval

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sync"
)

// Cache stores outputs by cache key. Implementations must be safe for
// concurrent use and must distinguish a miss from a stored zero value.
type Cache interface {
	Get(ctx context.Context, key string) (*Output, bool)
	Put(ctx context.Context, key string, output *Output)
}

// MemoryCache is a process-local Cache. It is safe for concurrent use.
type MemoryCache struct {
	mu      sync.RWMutex
	entries map[string]*Output
}

// NewMemoryCache returns an empty in-process cache.
func NewMemoryCache() *MemoryCache {
	return &MemoryCache{entries: make(map[string]*Output)}
}

// Get returns the cached output for key.
func (c *MemoryCache) Get(_ context.Context, key string) (*Output, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out, ok := c.entries[key]
	if !ok {
		return nil, false
	}
	return out.clone(), true
}

// Put stores output under key.
func (c *MemoryCache) Put(_ context.Context, key string, output *Output) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries[key] = output.clone()
}

// Len returns the number of cached entries.
func (c *MemoryCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.entries)
}

// cacheInput is the canonical material that a cache key is derived from. It
// intentionally includes the prompt version, so a prompt change cannot hit a
// response produced by the old prompt.
type cacheInput struct {
	Sample        Sample `json:"sample"`
	ModelID       string `json:"model_id"`
	PromptVersion string `json:"prompt_version"`
	Salt          string `json:"salt,omitempty"`
	Schema        int    `json:"schema"`
}

// cacheKey returns a stable key for one sample under one configuration.
func cacheKey(cfg Config, sample Sample) string {
	in := cacheInput{
		Sample:        sample,
		ModelID:       cfg.ModelID,
		PromptVersion: cfg.PromptVersion,
		Salt:          cfg.CacheSalt,
		Schema:        SchemaVersion,
	}
	// json.Marshal on structs with only string/slice fields cannot fail.
	b, err := json.Marshal(in)
	if err != nil {
		// Fall back to a key that still includes every input; this branch is
		// unreachable for the types involved.
		b = []byte(sample.ID + "\x00" + cfg.ModelID + "\x00" + cfg.PromptVersion + "\x00" + cfg.CacheSalt)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
