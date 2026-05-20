// tracing.go — package-scoped tracer + helpers for OpenInference-aware
// span emission. Tracer name matches OTel convention (lowercased package
// import path) so Phoenix UI groups spans by their owning code.
//
// Two convention layers are stacked on each span:
//   - OTel SemConv (gen_ai.*, db.*, server.*)  — cross-vendor portability,
//     consumed by Tempo/Jaeger/Datadog/X-Ray with their respective views.
//   - OpenInference (openinference.span.kind, llm.*, embedding.*, retrieval.*)
//     — Phoenix-native overlay that drives the "Kind" facet and specialised
//     drilldown panels (LLM token usage, embedding vector stats, etc.).
//
// Both are plain OTel attributes — adding the OpenInference layer does not
// break OTel readers; it just enriches the data for Phoenix.
package tools

import (
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

var tracer = otel.Tracer("github.com/pylyp-gh/doc-writer-mcp/internal/tools")

// OpenInference span kinds — enum values consumed by Phoenix UI.
// See https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md
const (
	kindChain     = "CHAIN"     // multi-step orchestration (root tool handler)
	kindLLM       = "LLM"       // LLM completion / sampling call
	kindEmbedding = "EMBEDDING" // text → vector
	kindRetriever = "RETRIEVER" // vector DB query / write
	kindGuardrail = "GUARDRAIL" // validation / safety gate before LLM
	kindTool      = "TOOL"      // generic tool / human-in-loop interaction
)

// setKind attaches the OpenInference span.kind attribute. Phoenix uses
// it both as a UI label badge and as the bucket key для specialised
// drilldown views (e.g. LLM kind unlocks token-usage panels).
func setKind(span trace.Span, kind string) {
	span.SetAttributes(attribute.String("openinference.span.kind", kind))
}

// endSpanWithErr finalises a span, marking status=Error when err != nil
// and attaching error message як event. Caller uses у defer:
//
//	ctx, span := tracer.Start(ctx, "step.name")
//	defer func() { endSpanWithErr(span, err) }()
func endSpanWithErr(span trace.Span, err error) {
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		span.RecordError(err)
	}
	span.End()
}
