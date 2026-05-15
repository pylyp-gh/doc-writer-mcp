package tools

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/google/uuid"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// ============================================================================
// Registration — auto-runs at package import (same pattern as echo.go).
// ============================================================================

func init() {
	registerTool(AddDocument())
}

// ============================================================================
// Config — all knobs through env vars (12-factor app, Principle III).
// Defaults point at in-cluster Service DNS for typical Lab 3 deployment.
// ============================================================================

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

var (
	ollamaURL        = envOr("OLLAMA_URL", "http://ollama.agentgateway-system.svc.cluster.local:11434/v1")
	ollamaModel      = envOr("OLLAMA_MODEL", "nomic-embed-text")
	qdrantURL        = envOr("QDRANT_URL", "http://qdrant.qdrant.svc.cluster.local:6333")
	qdrantCollection = envOr("QDRANT_COLLECTION", "doc-writer")
	vectorDim        = 768  // matches nomic-embed-text dimension
	dupThreshold     = 0.95 // cosine similarity — paraphrase-tolerant duplicate detection
)

// Single package-level http.Client — TCP connection reuse, avoid per-call thrash.
var httpClient = &http.Client{Timeout: 30 * time.Second}

// ============================================================================
// Schema types — input/output of the tool.
// ============================================================================

type AddDocumentParams struct {
	Text      string `json:"text" description:"The document text to embed and store in the vector database."`
	SourceURL string `json:"sourceUrl,omitempty" description:"Optional URL identifying where the document came from."`
}

type AddDocumentResult struct {
	Success        bool   `json:"success" description:"Whether the document was successfully stored."`
	PointID        string `json:"pointId" description:"Qdrant point UUID for the inserted document."`
	CollectionName string `json:"collectionName" description:"Qdrant collection that received the point."`
	Action         string `json:"action" description:"What happened: inserted | replaced | versioned | added_variant."`
	Message        string `json:"message,omitempty" description:"Human-readable explanation when IsError-style soft failure occurs."`
}

// ============================================================================
// Tool — mirrors Echo() pattern (go-sdk v1.6.0 handler signature).
// ============================================================================

func AddDocument() MCPTool[AddDocumentParams, AddDocumentResult] {
	return MCPTool[AddDocumentParams, AddDocumentResult]{
		Name: "add_document",
		Description: "Embed a text document and store it in the Qdrant vector " +
			"database. Uses elicitation to ask the user before creating a " +
			"missing collection, and again when a duplicate is detected.",
		Handler: handleAddDocument,
	}
}

// ============================================================================
// Main handler — orchestrates embed, collection bootstrap, dup detection, upsert.
//
// Hard errors (`return nil, _, err`) for system/protocol failures — network
// down, invalid response shapes, things the agent cannot recover from.
//
// Soft errors (`return &result{IsError:true}, output, nil`) for expected
// business outcomes — user declines, validation rules, missing prerequisites.
// The agent reads the message and decides what to do next.
// ============================================================================

func handleAddDocument(
	ctx context.Context,
	req *mcp.CallToolRequest,
	params AddDocumentParams,
) (*mcp.CallToolResult, AddDocumentResult, error) {

	text := params.Text
	sourceURL := params.SourceURL

	// --- Guard: empty text — soft error so the agent can re-prompt the user
	if text == "" {
		return softErr("text parameter is required and cannot be empty")
	}

	// --- STEP 1: Embed text via Ollama (OpenAI-compatible embeddings endpoint)
	vector, err := embedText(ctx, text)
	if err != nil {
		return nil, AddDocumentResult{}, fmt.Errorf("embed failed: %w", err)
	}

	// --- STEP 2: Check collection existence
	exists, err := checkCollection(ctx)
	if err != nil {
		return nil, AddDocumentResult{}, fmt.Errorf("collection check failed: %w", err)
	}

	// --- STEP 2.5: Elicit bootstrap if missing
	if !exists {
		approved, err := elicitBootstrap(ctx, req.Session)
		if err != nil {
			return nil, AddDocumentResult{}, fmt.Errorf("elicit bootstrap failed: %w", err)
		}
		if !approved {
			return softErr(fmt.Sprintf(
				"Cannot add document: collection '%s' is missing and user did "+
					"not approve creation. Provide an existing collection name via "+
					"the QDRANT_COLLECTION env var, or accept the create prompt next time.",
				qdrantCollection,
			))
		}
		if err := createCollection(ctx); err != nil {
			return nil, AddDocumentResult{}, fmt.Errorf("collection create failed: %w", err)
		}
	}

	// --- STEP 3: Duplicate detection — cosine similarity search
	dup, err := searchSimilar(ctx, vector)
	if err != nil {
		return nil, AddDocumentResult{}, fmt.Errorf("search failed: %w", err)
	}

	action := "inserted"
	pointID := uuid.NewString()

	// --- STEP 3.5: Elicit dup-handling if a match was found
	if dup != nil {
		choice, err := elicitDupAction(ctx, req.Session, dup)
		if err != nil {
			return nil, AddDocumentResult{}, fmt.Errorf("elicit dup action failed: %w", err)
		}
		switch choice {
		case "decline", "":
			return softErr("Duplicate detected; user declined insertion.")
		case "replace":
			if err := deletePoint(ctx, dup.ID); err != nil {
				return nil, AddDocumentResult{}, fmt.Errorf("delete old point: %w", err)
			}
			action = "replaced"
		case "new_version":
			if err := deletePoint(ctx, dup.ID); err != nil {
				return nil, AddDocumentResult{}, fmt.Errorf("delete old point: %w", err)
			}
			action = "versioned"
		case "add_anyway":
			action = "added_variant"
		}
	}

	// --- STEP 4: Build payload + upsert into Qdrant
	hash := sha256.Sum256([]byte(text))
	payload := map[string]any{
		"text":      text,
		"sourceUrl": sourceURL,
		"hash":      hex.EncodeToString(hash[:]),
		"addedAt":   time.Now().UTC().Format(time.RFC3339),
		"action":    action,
	}
	if err := upsertPoint(ctx, pointID, vector, payload); err != nil {
		return nil, AddDocumentResult{}, fmt.Errorf("upsert failed: %w", err)
	}

	// --- STEP 5: Return success
	result := &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{
				Text: fmt.Sprintf("Document %s into collection '%s' as point %s",
					action, qdrantCollection, pointID),
			},
		},
	}
	output := AddDocumentResult{
		Success:        true,
		PointID:        pointID,
		CollectionName: qdrantCollection,
		Action:         action,
	}
	return result, output, nil
}

// ============================================================================
// Elicitation helpers — wrap ServerSession.Elicit (go-sdk v1.6.0).
// ============================================================================

func elicitBootstrap(ctx context.Context, ss *mcp.ServerSession) (bool, error) {
	resp, err := ss.Elicit(ctx, &mcp.ElicitParams{
		Message: fmt.Sprintf(
			"Collection '%s' doesn't exist. Create it with %d-dimension Cosine vectors?",
			qdrantCollection, vectorDim,
		),
		RequestedSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"create": map[string]any{
					"type":        "boolean",
					"description": "Create the missing collection",
					"default":     true,
				},
			},
			"required": []string{"create"},
		},
	})
	if err != nil {
		return false, err
	}
	if resp.Action != "accept" {
		return false, nil
	}
	create, _ := resp.Content["create"].(bool)
	return create, nil
}

func elicitDupAction(ctx context.Context, ss *mcp.ServerSession, dup *dupHit) (string, error) {
	resp, err := ss.Elicit(ctx, &mcp.ElicitParams{
		Message: fmt.Sprintf(
			"Similar document found (cosine score=%.3f):\n%s\n\nHow to handle?",
			dup.Score, previewFromPayload(dup.Payload),
		),
		RequestedSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"choice": map[string]any{
					"type": "string",
					"enum": []string{"new_version", "replace", "decline", "add_anyway"},
					"enumNames": []string{
						"New version (replace + bump version)",
						"Replace (exact swap)",
						"Decline (cancel)",
						"Add as variant (keep both)",
					},
					"description": "How to handle the duplicate",
				},
			},
			"required": []string{"choice"},
		},
	})
	if err != nil {
		return "", err
	}
	if resp.Action != "accept" {
		return "decline", nil
	}
	choice, _ := resp.Content["choice"].(string)
	return choice, nil
}

func previewFromPayload(p map[string]any) string {
	if text, ok := p["text"].(string); ok {
		if len(text) > 200 {
			return text[:200] + "..."
		}
		return text
	}
	if src, ok := p["sourceUrl"].(string); ok {
		return "Source: " + src
	}
	return "(no preview available)"
}

// ============================================================================
// HTTP helpers — Ollama embed + Qdrant CRUD.
// All return errors with context; the caller decides hard/soft semantics.
// ============================================================================

func embedText(ctx context.Context, text string) ([]float32, error) {
	body, _ := json.Marshal(map[string]any{
		"model": ollamaModel,
		"input": text,
	})
	req, _ := http.NewRequestWithContext(ctx, "POST", ollamaURL+"/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer ollama") // OpenAI SDK requires non-empty

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama returned HTTP %d", resp.StatusCode)
	}
	var out struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode ollama response: %w", err)
	}
	if len(out.Data) == 0 || len(out.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("ollama returned empty embedding")
	}
	return out.Data[0].Embedding, nil
}

func checkCollection(ctx context.Context) (bool, error) {
	req, _ := http.NewRequestWithContext(ctx, "GET",
		qdrantURL+"/collections/"+qdrantCollection, nil)
	resp, err := httpClient.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusOK:
		return true, nil
	case http.StatusNotFound:
		return false, nil
	default:
		return false, fmt.Errorf("qdrant check unexpected HTTP %d", resp.StatusCode)
	}
}

func createCollection(ctx context.Context) error {
	body, _ := json.Marshal(map[string]any{
		"vectors": map[string]any{
			"size":     vectorDim,
			"distance": "Cosine",
		},
	})
	req, _ := http.NewRequestWithContext(ctx, "PUT",
		qdrantURL+"/collections/"+qdrantCollection, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("qdrant create returned HTTP %d", resp.StatusCode)
	}
	return nil
}

type dupHit struct {
	ID      string         `json:"id"`
	Score   float64        `json:"score"`
	Payload map[string]any `json:"payload"`
}

func searchSimilar(ctx context.Context, vector []float32) (*dupHit, error) {
	body, _ := json.Marshal(map[string]any{
		"vector":          vector,
		"limit":           1,
		"score_threshold": dupThreshold,
		"with_payload":    true,
	})
	req, _ := http.NewRequestWithContext(ctx, "POST",
		qdrantURL+"/collections/"+qdrantCollection+"/points/search", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("qdrant search HTTP %d", resp.StatusCode)
	}
	var out struct {
		Result []dupHit `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode qdrant search: %w", err)
	}
	if len(out.Result) == 0 {
		return nil, nil
	}
	return &out.Result[0], nil
}

func upsertPoint(ctx context.Context, id string, vector []float32, payload map[string]any) error {
	body, _ := json.Marshal(map[string]any{
		"points": []map[string]any{
			{"id": id, "vector": vector, "payload": payload},
		},
	})
	req, _ := http.NewRequestWithContext(ctx, "PUT",
		qdrantURL+"/collections/"+qdrantCollection+"/points", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("qdrant upsert HTTP %d", resp.StatusCode)
	}
	return nil
}

func deletePoint(ctx context.Context, id string) error {
	body, _ := json.Marshal(map[string]any{
		"points": []string{id},
	})
	req, _ := http.NewRequestWithContext(ctx, "POST",
		qdrantURL+"/collections/"+qdrantCollection+"/points/delete", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("qdrant delete HTTP %d", resp.StatusCode)
	}
	return nil
}

// ============================================================================
// Soft-error helper — returns a tool result with IsError=true and a message.
// Use for expected business outcomes (user decline, validation rejection)
// so the LLM agent reads the text and reasons (instead of seeing JSON-RPC error).
// ============================================================================

func softErr(msg string) (*mcp.CallToolResult, AddDocumentResult, error) {
	result := &mcp.CallToolResult{
		IsError: true,
		Content: []mcp.Content{
			&mcp.TextContent{Text: msg},
		},
	}
	output := AddDocumentResult{
		Success: false,
		Message: msg,
	}
	return result, output, nil
}
