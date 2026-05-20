// tracing.go — OpenTelemetry SDK initialization для doc-writer-mcp.
//
// Sends spans via OTLP gRPC to Phoenix (or any OTLP-compatible
// collector) when OTEL_EXPORTER_OTLP_ENDPOINT is set. Defaults: empty
// endpoint disables tracing (no-op provider) so dev/test runs don't
// hard-fail when no collector is reachable.
//
// Environment variables:
//   OTEL_EXPORTER_OTLP_ENDPOINT — host:port для OTLP gRPC server
//     (e.g. "phoenix-svc.phoenix.svc.cluster.local:4317")
//   OTEL_SERVICE_NAME           — override the default "doc-writer-mcp"
//                                  resource attribute
//   OTEL_SERVICE_VERSION        — override the default "0.1.0"
//   OTEL_EXPORTER_OTLP_INSECURE — "true" → use insecure gRPC
//                                  (default true; in-cluster traffic)
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func initTracing(ctx context.Context) (func(context.Context) error, error) {
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint == "" {
		log.Println("OTEL_EXPORTER_OTLP_ENDPOINT not set — tracing disabled (no-op provider)")
		// Install no-op tracer provider so calls to otel.Tracer(...) don't panic.
		otel.SetTracerProvider(sdktrace.NewTracerProvider())
		return func(context.Context) error { return nil }, nil
	}

	serviceName := os.Getenv("OTEL_SERVICE_NAME")
	if serviceName == "" {
		serviceName = "doc-writer-mcp"
	}
	serviceVersion := os.Getenv("OTEL_SERVICE_VERSION")
	if serviceVersion == "" {
		serviceVersion = "0.1.0"
	}

	conn, err := grpc.NewClient(
		endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("otel: grpc.NewClient %s: %w", endpoint, err)
	}

	exporter, err := otlptracegrpc.New(ctx, otlptracegrpc.WithGRPCConn(conn))
	if err != nil {
		return nil, fmt.Errorf("otel: otlptracegrpc.New: %w", err)
	}

	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName(serviceName),
			semconv.ServiceVersion(serviceVersion),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("otel: resource.New: %w", err)
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
	)

	otel.SetTracerProvider(tp)
	// W3C TraceContext propagation — let upstream callers (agentgateway,
	// agent runtimes) link their spans to ours.
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	log.Printf("OTel tracing initialized — service=%q version=%q endpoint=%q",
		serviceName, serviceVersion, endpoint)

	// Return shutdown closure for graceful drain on stop.
	return tp.Shutdown, nil
}
