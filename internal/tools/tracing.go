// tracing.go — package-scoped tracer + small helper для setting span
// status from error. Tracer name matches OTel convention (lowercased
// package import path) so Phoenix UI groups spans by their owning code.
package tools

import (
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

var tracer = otel.Tracer("github.com/pylyp-gh/doc-writer-mcp/internal/tools")

// endSpanWithErr finalises a span, marking status=Error when err != nil
// and attaching error message як event. Caller uses в defer:
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
