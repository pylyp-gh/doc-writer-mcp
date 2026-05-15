package tools

import (
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// AddToolsToServer iterates the toolsToAdd slice (populated by each tool's
// init() through registerTool) and attaches every registered tool to the
// given MCP server.
func AddToolsToServer(server *mcp.Server) {
	for _, addToolFunc := range toolsToAdd {
		addToolFunc(server)
	}
}

var toolsToAdd []func(server *mcp.Server)

// registerTool appends a closure that calls mcp.AddTool[In, Out] when the
// server is being built up. Each tool file's init() invokes this with its
// own typed handler.
func registerTool[In, Out any](tool MCPTool[In, Out]) {
	toolsToAdd = append(toolsToAdd, func(server *mcp.Server) {
		mcp.AddTool(server, &mcp.Tool{Name: tool.Name, Description: tool.Description}, tool.Handler)
	})
}

// MCPTool wraps a tool's metadata + typed handler.
// Handler signature matches go-sdk v1.6.0 ToolHandlerFor[In, Out]:
//
//	func(ctx, *CallToolRequest, input In) (*CallToolResult, output Out, error)
//
// Returning a non-nil output Out auto-populates result.StructuredContent.
// The middle Out value is *typed* structured output for programmatic agents;
// Content (text/image/etc.) is for human-readable display.
type MCPTool[In, Out any] struct {
	Name        string
	Description string
	Handler     mcp.ToolHandlerFor[In, Out]
}
