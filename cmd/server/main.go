package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/pylyp-gh/doc-writer-mcp/internal/tools"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

var (
	httpAddr = flag.String("http", "", "if set, use streamable HTTP to serve MCP (on this address), instead of stdin/stdout")
)

func main() {
	flag.Parse()

	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	// Initialize OTel tracing (no-op when OTEL_EXPORTER_OTLP_ENDPOINT unset).
	ctx := context.Background()
	shutdownTracer, err := initTracing(ctx)
	if err != nil {
		log.Printf("WARN: tracing init failed (%v) — continuing without telemetry", err)
		shutdownTracer = func(context.Context) error { return nil }
	}
	defer func() {
		_ = shutdownTracer(context.Background())
	}()

	// Create the MCP server
	server := mcp.NewServer(&mcp.Implementation{Name: "doc-writer-mcp", Version: "0.1.0"}, nil)

	// Register tools
	tools.AddToolsToServer(server)

	// Start server with appropriate transport. Tracing instrumentation
	// lives у the tool handlers themselves (semantic spans per
	// validate/sampling/qdrant operation) — HTTP-level otelhttp wrap
	// was dropped because per-request DELETE 204 session-teardown +
	// healthcheck noise drowned the actual LLM operation signal у Phoenix.
	if *httpAddr != "" {
		handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server {
			return server
		}, nil)
		log.Printf("MCP server listening at %s", *httpAddr)
		return http.ListenAndServe(*httpAddr, handler)
	} else {
		// v1.6.0: NewStdioTransport / NewLoggingTransport functions removed —
		// transports are now plain structs with public fields, initialized
		// via composite literal.
		t := &mcp.LoggingTransport{
			Transport: &mcp.StdioTransport{},
			Writer:    os.Stderr,
		}
		return server.Run(context.Background(), t)
	}
}
