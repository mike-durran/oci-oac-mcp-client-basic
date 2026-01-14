# OCI OAC MCP Client Basic

A basic Model Context Protocol (MCP) client implementation using Oracle Cloud Infrastructure (OCI) Generative AI as the LLM backend. This educational example demonstrates how to build an MCP client that can connect to any MCP-compliant server, such as Oracle Analytics Cloud (OAC).

## Overview

This project provides a Python-based MCP client that:

- Connects to MCP servers via **Streamable HTTP** transport (the current recommended transport for remote MCP servers)
- Uses **OCI Generative AI** (e.g., Cohere Command R+, Meta Llama, etc.) for natural language understanding
- Implements **tool calling via prompt engineering** since OCI GenAI doesn't have native tool use support
- Provides an interactive chat interface for querying data through MCP tools

## What is MCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open standard created by Anthropic that allows AI applications to connect to external data sources and tools in a standardized way. Key concepts:

| Concept       | Description                                                         |
|---------------|---------------------------------------------------------------------|
| **Client**    | The application that connects to MCP servers (this code)            |
| **Server**    | Exposes tools, resources, and prompts to clients                    |
| **Tools**     | Functions that can be called (e.g., `execute_sql`, `describe_data`) |
| **Resources** | Data that can be read (files, database records)                     |
| **Transport** | How client and server communicate (stdio, HTTP, SSE)                |

## Prerequisites

- Python 3.10+
- Oracle Cloud Infrastructure account with:
  - Access to OCI Generative AI service
  - A compartment with GenAI enabled
  - OCI CLI configured (`~/.oci/config`)
- Access to Oracle Analytics Cloud MCP endpoint

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/oci-oac-mcp-client-basic.git
cd oci-oac-mcp-client-basic
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables:**

Create a `.env` file in the project root:

```env
# MCP Server Configuration
MCP_SERVER_URL=https://<oac-instance>.analytics.ocp.oraclecloud.com/api/mcp
MCP_ACCESS_TOKEN=your-bearer-token-here

# OCI Generative AI Configuration
OCI_COMPARTMENT_ID=ocid1.compartment.oc1..aaaa...
OCI_GENAI_MODEL_ID=ocid1.generativeaimodel.oc1..aaaa...
OCI_GENAI_ENDPOINT=https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com
OCI_CONFIG_FILE=~/.oci/config
OCI_CONFIG_PROFILE=DEFAULT
OCI_AUTH_TYPE=security_token
```

## OCI Authentication

This client supports two authentication methods:

### Security Token (Recommended for Development)

```bash
oci session authenticate --profile DEFAULT
```

This creates a temporary session token (usually valid for 1 hour). Set `OCI_AUTH_TYPE=security_token` in your `.env` file.

### API Key (For Production)

Configure your `~/.oci/config` with a permanent API key. Set `OCI_AUTH_TYPE=api_key` in your `.env` file.

## Usage

Run the client:

```bash
python mcp_client.py
```

You'll see output like:

```
============================================================
  Initializing MCP Client with OCI GenAI
============================================================

Step 1: Connecting to OCI Generative AI...
✓ Authenticated using security token
✓ Model: ocid1.generativeaimodel.oc1...
✓ Endpoint: https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com

Step 2: Connecting to MCP Server...
Connecting to MCP server at: https://your-oac.analytics.ocp.oraclecloud.com/api/mcp

✓ Connected to: Oracle Analytics MCP Server v1.0.0
✓ Protocol version: 2024-11-05

✓ Available tools (3):
  • oracle_analytics-discover_data: Discover available subject areas and datasets
  • oracle_analytics-describe_data: Get metadata for columns and measures
  • oracle_analytics-execute_logical_sql: Execute Logical SQL queries

============================================================
  MCP Client (OCI GenAI) Ready!
  Type 'quit' to exit.
============================================================

You: 
```

### Example Queries

```
You: What data sources are available?

You: Show me the top 10 products by revenue

You: What's the year-over-year growth for each region?
```

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   User Query    │────▶│   OCI GenAI     │────▶│   MCP Server    │
│                 │     │   (LLM)         │     │   (OAC)         │
│                 │◀────│                 │◀────│                 │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │                       │                       │
        ▼                       ▼                       ▼
   Natural Language      Tool Call Parsing       Tool Execution
   Input/Output          & Prompt Engineering    via MCP Protocol
```

1. **User asks a question** in natural language
2. **MCP Client sends query to OCI GenAI** with descriptions of available tools
3. **LLM decides** whether to use a tool or answer directly
4. **If tool needed**, client parses the tool call and executes via MCP
5. **Tool results** are sent back to the LLM
6. **LLM provides final answer** to the user

## Project Structure

```
oci-oac-mcp-client-basic/
├── mcp_client.py       # Main client implementation
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment configuration
├── .gitignore          # Git ignore patterns
└── README.md           # This file
```

## Key Classes

### `OCIGenAIClient`

Wrapper for Oracle Cloud Infrastructure Generative AI service. Handles:
- OCI authentication (security token or API key)
- Message format conversion (OpenAI-style to OCI format)
- Chat completions with configurable parameters

### `MCPClient`

Model Context Protocol client implementation. Handles:
- Streamable HTTP connection to MCP servers
- Tool discovery and caching
- Agentic loop for multi-turn tool calling
- Response parsing and tool result handling

## Troubleshooting

### "security_token_file not found"

Run `oci session authenticate` to create a new session token.

### Connection timeout

- Verify the MCP server URL is correct
- Check network connectivity
- Ensure your bearer token is valid

### Tool calls not parsing

The LLM may not always format tool calls correctly. The client includes fallback parsing, but you may need to rephrase your query.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see https://opensource.org/license/mit for details.

Copyright (c) 2026 Mike Durran

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Resources

- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [OCI Generative AI Documentation](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)
- [Oracle Analytics Cloud Documentation](https://docs.oracle.com/en/cloud/paas/analytics-cloud/)
- [Oracle Analytics Cloud MCP Server Documentation]((https://docs.oracle.com/en/cloud/paas/analytics-cloud/acsdv/access-oracle-analytics-cloud-mcp-server.html)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
