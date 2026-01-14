"""
================================================================================
MCP CLIENT FOR ORACLE OCI GENERATIVE AI
================================================================================

This is an educational implementation of a Model Context Protocol (MCP) client
that uses Oracle Cloud Infrastructure (OCI) Generative AI as the LLM backend.

WHAT IS MCP?
------------
The Model Context Protocol (MCP) is an open standard created by Anthropic that
allows AI applications to connect to external data sources and tools in a 
standardized way. Any MCP-compliant client can connect to any MCP-compliant server.

Key MCP concepts:
- CLIENT: The application that connects to MCP servers (this code)
- SERVER: Exposes tools, resources, and prompts to clients
- TOOLS: Functions that can be called (like "execute_sql", "get_weather")
- RESOURCES: Data that can be read (like files, database records)
- TRANSPORT: How client and server communicate (stdio, HTTP, SSE)

This client uses "Streamable HTTP" transport - the current recommended 
transport for remote MCP servers (replaced SSE in March 2025).

WHAT IS OCI GENAI?
------------------
Oracle Cloud Infrastructure Generative AI is Oracle's managed LLM service.
It provides access to models like Cohere Command R+ and Meta Llama.
Unlike e.g., Anthropic Claude, OCI GenAI doesn't have native "tool use" support,
so this client implements tool calling via prompt engineering.

Authenticate to OCI 
-------------------
oci session authenticate --profile DEFAULT

HOW THIS CLIENT WORKS:
----------------------
1. Connect to MCP server via Streamable HTTP
2. Discover available tools from the server
3. When user asks a question:
   a. Send question + tool descriptions to OCI GenAI
   b. If LLM wants to use a tool, parse the response
   c. Execute the tool via MCP
   d. Send results back to LLM
   e. Repeat until LLM gives final answer

================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import asyncio          # For async/await support (MCP uses async operations)
import json             # For parsing tool calls from LLM responses
import os               # For environment variables
import re               # For regex parsing of tool calls
from typing import Optional
from contextlib import AsyncExitStack  # For managing async resources

# OCI SDK - Oracle's Python SDK for cloud services
import oci

# python-dotenv - Loads environment variables from .env file
from dotenv import load_dotenv

# MCP SDK - The official Model Context Protocol Python SDK
from mcp import ClientSession                                 # Manages MCP session
from mcp.client.streamable_http import streamablehttp_client  # HTTP transport

# Load environment variables from .env file
# This must be called before accessing os.getenv() for .env variables
load_dotenv()

# .env file needs to include:
#
# MCP_SERVER_URL=https://<oac-instance>.analytics.ocp.oraclecloud.com/api/mcp
# MCP_ACCESS_TOKEN=your-bearer-token-here
#
# OCI_COMPARTMENT_ID=<ocid>
# OCI_GENAI_MODEL_ID=<ocid>
# OCI_GENAI_ENDPOINT=<uri>
# OCI_CONFIG_FILE=<path-to>/.oci/config
# OCI_CONFIG_PROFILE=DEFAULT
# OCI_AUTH_TYPE=security_token


# =============================================================================
# OCI GENERATIVE AI CLIENT
# =============================================================================

class OCIGenAIClient:
    """
    Wrapper class for Oracle Cloud Infrastructure Generative AI service.
    
    This class handles:
    - Authentication with OCI (supports security token and API key auth)
    - Sending chat requests to OCI GenAI models
    - Parsing responses from the model
    
    OCI GenAI uses a different API structure than OpenAI/Anthropic, so this
    class abstracts those differences into a simple chat() method.
    """
    
    def __init__(
        self,
        compartment_id: str,
        model_id: str,
        endpoint: str,
        config_file: str,
        config_profile: str,
        auth_type: str
    ):
        """
        Initialize the OCI GenAI client.
        
        Args:
            compartment_id: OCI compartment OCID where GenAI is enabled
            model_id: OCID of the specific model to use (e.g., Cohere Command R+)
            endpoint: OCI GenAI endpoint URL (region-specific)
            config_file: Path to OCI config file (usually ~/.oci/config)
            config_profile: Profile name in config file (usually DEFAULT)
            auth_type: 'security_token' or 'api_key'
        """
        self.compartment_id = compartment_id
        self.model_id = model_id
        
        # Expand ~ to full home directory path
        config_file = os.path.expanduser(config_file)
        
        # Load OCI configuration from file
        # This file contains tenancy, user, region, and key information
        config = oci.config.from_file(config_file, config_profile)
        
        # =================================================================
        # AUTHENTICATION SETUP
        # =================================================================
        # OCI supports multiple authentication methods:
        # 1. API Key: Traditional method using a private key file
        # 2. Security Token: Session-based auth from 'oci session authenticate'
        # 3. Instance Principal: For code running on OCI compute
        # 4. Resource Principal: For OCI Functions
        
        if auth_type == "security_token":
            # Security token auth is used when you run 'oci session authenticate'
            # It creates a temporary token that expires (usually 1 hour)
            
            # The token file path is stored in the OCI config
            security_token_file = config.get("security_token_file")
            
            if not security_token_file:
                raise ValueError(
                    "security_token_file not found in OCI config. "
                    "Run 'oci session authenticate' first."
                )
            
            # Read the security token from file
            token_path = os.path.expanduser(security_token_file)
            with open(token_path, 'r') as f:
                token = f.read().strip()
            
            # Load the private key for signing requests
            key_file = os.path.expanduser(config["key_file"])
            private_key = oci.signer.load_private_key_from_file(key_file)
            
            # Create a signer that uses the security token
            signer = oci.auth.signers.SecurityTokenSigner(token, private_key)
            
            # Create the GenAI client with security token authentication
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config={"region": config.get("region")},
                signer=signer,
                service_endpoint=endpoint,
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240)  # (connect_timeout, read_timeout) in seconds
            )
            print("✓ Authenticated using security token")
            
        else:
            # Standard API key authentication
            # Uses the key_file specified in ~/.oci/config
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=config,
                service_endpoint=endpoint,
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240)
            )
            print("✓ Authenticated using API key")
        
        print(f"✓ Model: {model_id}")
        print(f"✓ Endpoint: {endpoint}")
    
    def chat(
        self, 
        messages: list, 
        max_tokens: int = 4096, 
        temperature: float = 0.7
    ) -> str:
        """
        Send a chat request to OCI GenAI and get a response.
        
        This method converts our simple message format to OCI's format,
        sends the request, and extracts the response text.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
                      Example: [{"role": "user", "content": "Hello!"}]
            max_tokens: Maximum tokens in the response
            temperature: Controls randomness (0=deterministic, 1=creative)
            
        Returns:
            The model's response as a string
        """
        # =================================================================
        # CONVERT MESSAGES TO OCI FORMAT
        # =================================================================
        # OCI uses a different message structure than OpenAI/Anthropic:
        # - Messages contain a list of "content" objects
        # - Roles are uppercase: "USER", "ASSISTANT" (not "user", "assistant")
        
        oci_messages = []
        for msg in messages:
            # Create a TextContent object for the message text
            content = oci.generative_ai_inference.models.TextContent()
            content.text = msg["content"]
            
            # Create the Message object
            message = oci.generative_ai_inference.models.Message()
            message.role = msg["role"].upper()  # OCI requires uppercase
            message.content = [content]  # Content is a list
            oci_messages.append(message)
        
        # =================================================================
        # BUILD THE CHAT REQUEST
        # =================================================================
        # OCI has a nested structure for chat requests:
        # ChatDetails -> ChatRequest -> Messages
        
        chat_request = oci.generative_ai_inference.models.GenericChatRequest()
        chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
        chat_request.messages = oci_messages
        chat_request.max_tokens = max_tokens
        chat_request.temperature = temperature
        chat_request.top_p = 0.9    # Nucleus sampling threshold
        chat_request.top_k = 40     # Top-k sampling (must be < 41 for OCI)
        
        # Create the outer ChatDetails wrapper
        chat_detail = oci.generative_ai_inference.models.ChatDetails()
        
        # Specify the model to use
        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=self.model_id
        )
        chat_detail.chat_request = chat_request
        chat_detail.compartment_id = self.compartment_id
        
        # =================================================================
        # SEND REQUEST AND PARSE RESPONSE
        # =================================================================
        response = self.client.chat(chat_detail)
        
        # Extract the text from the response
        # Response structure: response.data.chat_response.choices[0].message.content[0].text
        chat_response = response.data.chat_response
        
        if hasattr(chat_response, 'choices') and chat_response.choices:
            return chat_response.choices[0].message.content[0].text
        elif hasattr(chat_response, 'text'):
            return chat_response.text
        else:
            return str(chat_response)


# =============================================================================
# MCP CLIENT
# =============================================================================

class MCPClient:
    """
    Model Context Protocol (MCP) client that connects to MCP servers
    and uses OCI GenAI for natural language understanding.
    
    This class handles:
    - Connecting to MCP servers via Streamable HTTP transport
    - Discovering available tools from the server
    - Processing user queries with tool calling
    - Managing the conversation loop
    
    The key innovation of MCP is that it standardizes how AI applications
    discover and use external tools. Instead of hardcoding tool integrations,
    the client can connect to any MCP server and automatically learn what
    tools are available.
    """
    
    def __init__(self, genai_client: OCIGenAIClient):
        """
        Initialize the MCP client.
        
        Args:
            genai_client: An initialized OCIGenAIClient for LLM calls
        """
        # The MCP session - handles protocol communication
        self.session: Optional[ClientSession] = None
        
        # AsyncExitStack manages cleanup of async resources
        # When we exit the stack, all resources are properly closed
        self.exit_stack = AsyncExitStack()
        
        # Our LLM client
        self.genai = genai_client
        
        # Server URL (set when connecting)
        self.server_url: Optional[str] = None
        
        # Cached list of tools from the server
        # We cache these to avoid fetching on every query
        self._cached_tools = None
        
        # Pre-built prompt describing available tools
        self._tools_prompt = None
    
    async def connect_to_server(self, server_url: str, access_token: str):
        """
        Connect to a remote MCP server using Streamable HTTP transport.
        
        The connection process:
        1. Open HTTP connection with authentication
        2. Initialize MCP session (protocol handshake)
        3. Discover available tools
        4. Cache tool information for later use
        
        Args:
            server_url: Full URL to the MCP endpoint (e.g., https://server.com/mcp)
            access_token: Bearer token for authentication
        """
        self.server_url = server_url
        
        # =================================================================
        # SETUP AUTHENTICATION HEADERS
        # =================================================================
        # MCP servers typically use Bearer token authentication
        # The token is sent in the Authorization header
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        
        print(f"Connecting to MCP server at: {server_url}")
        
        # =================================================================
        # ESTABLISH STREAMABLE HTTP CONNECTION
        # =================================================================
        # streamablehttp_client creates a connection to the MCP server
        # It returns:
        # - read_stream: For receiving messages from the server
        # - write_stream: For sending messages to the server  
        # - session_id: Optional server-assigned session ID
        #
        # We use enter_async_context to ensure proper cleanup when done
        
        streams = await self.exit_stack.enter_async_context(
            streamablehttp_client(
                url=server_url,
                headers=headers,
                timeout=30.0  # Connection timeout in seconds
            )
        )
        read_stream, write_stream, _ = streams
        
        # =================================================================
        # CREATE MCP SESSION
        # =================================================================
        # ClientSession manages the MCP protocol communication
        # It handles JSON-RPC message formatting, request/response matching, etc.
        
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        
        # =================================================================
        # INITIALIZE THE MCP CONNECTION
        # =================================================================
        # The initialize() call performs the MCP handshake:
        # 1. Client sends: protocol version, capabilities
        # 2. Server responds: server info, capabilities, protocol version
        #
        # This is where we learn what the server can do
        
        init_result = await self.session.initialize()
        
        # Display server information
        # Note: MCP SDK uses camelCase (serverInfo, protocolVersion)
        print(f"\n✓ Connected to: {init_result.serverInfo.name} v{init_result.serverInfo.version}")
        print(f"✓ Protocol version: {init_result.protocolVersion}")
        
        # =================================================================
        # DISCOVER AVAILABLE TOOLS
        # =================================================================
        # list_tools() asks the server what tools are available
        # Each tool has:
        # - name: Unique identifier (e.g., "execute_sql")
        # - description: Human-readable explanation
        # - inputSchema: JSON Schema defining the parameters
        
        response = await self.session.list_tools()
        tools = response.tools
        
        # Cache the tools for later use
        self._cached_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema
            }
            for tool in tools
        ]
        
        # Build the tools prompt for the LLM
        self._tools_prompt = self._build_tools_prompt()
        
        # Display available tools
        print(f"\n✓ Available tools ({len(tools)}):")
        for tool in tools:
            print(f"  • {tool.name}: {tool.description}")
    
    def _build_tools_prompt(self) -> str:
        """
        Build a text description of available tools for the LLM.
        
        Since OCI GenAI doesn't have native tool calling, we need to
        describe the tools in the prompt and ask the LLM to format
        tool calls in a specific way we can parse.
        
        Returns:
            Formatted string describing all available tools
        """
        tools_desc = []
        
        for tool in self._cached_tools:
            # Get the JSON Schema for parameters
            schema = tool.get("inputSchema", {})
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            # Build parameter descriptions
            params = []
            for param_name, param_def in properties.items():
                param_type = param_def.get("type", "any")
                param_desc = param_def.get("description", "")
                req_marker = "(required)" if param_name in required else "(optional)"
                params.append(f"    - {param_name}: {param_type} {req_marker} - {param_desc}")
            
            params_str = "\n".join(params) if params else "    (no parameters)"
            tools_desc.append(
                f"• {tool['name']}: {tool['description']}\n"
                f"  Parameters:\n{params_str}"
            )
        
        return "\n\n".join(tools_desc)
    
    def _parse_tool_call(self, response: str) -> tuple[Optional[str], Optional[dict]]:
        """
        Parse the LLM response to extract tool calls.
        
        We instruct the LLM to format tool calls like:
        <tool_call>{"tool": "tool_name", "arguments": {...}}</tool_call>
        
        This method extracts that JSON and returns the tool name and arguments.
        
        Args:
            response: The raw text response from the LLM
            
        Returns:
            Tuple of (tool_name, arguments) or (None, None) if no tool call found
        """
        # Try to find tool call in our expected format
        tool_call_match = re.search(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>', 
            response, 
            re.DOTALL
        )
        
        if tool_call_match:
            try:
                tool_call = json.loads(tool_call_match.group(1))
                return tool_call.get("tool"), tool_call.get("arguments", {})
            except json.JSONDecodeError:
                pass
        
        # Fallback: Try to find raw JSON that looks like a tool call
        json_match = re.search(
            r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}', 
            response
        )
        if json_match:
            try:
                tool_call = json.loads(json_match.group(0))
                return tool_call.get("tool"), tool_call.get("arguments", {})
            except json.JSONDecodeError:
                pass
        
        return None, None
    
    async def process_query(self, query: str) -> str:
        """
        Process a user query using the LLM and available MCP tools.
        
        This implements an "agentic loop":
        1. Send query to LLM with tool descriptions
        2. If LLM wants to use a tool, execute it via MCP
        3. Send tool results back to LLM
        4. Repeat until LLM gives a final answer
        
        Args:
            query: The user's question or request
            
        Returns:
            The final response from the LLM
        """
        # =================================================================
        # BUILD THE SYSTEM PROMPT
        # =================================================================
        # We need to tell the LLM:
        # 1. What tools are available
        # 2. How to format tool calls
        # 3. When to use tools vs answer directly
        
        system_prompt = f"""You are a helpful assistant with access to the following tools:

{self._tools_prompt}

When you need to use a tool, respond with ONLY a tool call in this exact format:
<tool_call>{{"tool": "tool_name", "arguments": {{"param1": "value1"}}}}</tool_call>

After receiving tool results, provide a helpful response to the user.
If you don't need to use a tool, just respond directly to the user.

Important: Only call one tool at a time. Wait for the result before calling another tool."""

        # Start the conversation with the system prompt + user query
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\nUser query: {query}"}
        ]
        
        # =================================================================
        # AGENTIC LOOP
        # =================================================================
        # Keep processing until the LLM gives a final answer (no tool call)
        # or we hit the iteration limit
        
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get response from OCI GenAI
            response = self.genai.chat(messages)
            
            # Check if the LLM wants to call a tool
            tool_name, tool_args = self._parse_tool_call(response)
            
            if tool_name and tool_args is not None:
                # =============================================================
                # EXECUTE TOOL VIA MCP
                # =============================================================
                print(f"\n[Calling tool: {tool_name}]")
                print(f"  Arguments: {tool_args}")
                
                try:
                    # call_tool sends an MCP request to execute the tool
                    tool_result = await self.session.call_tool(tool_name, tool_args)
                    
                    # Extract text from the tool result
                    result_content = []
                    for content in tool_result.content:
                        if hasattr(content, 'text'):
                            result_content.append(content.text)
                        elif hasattr(content, 'data'):
                            mime_type = getattr(content, 'mimeType', 'unknown')
                            result_content.append(f"[Binary data: {mime_type}]")
                    
                    result_text = "\n".join(result_content) if result_content else str(tool_result.content)
                    
                    # Show truncated result
                    display_result = result_text[:200] + "..." if len(result_text) > 200 else result_text
                    print(f"  Result: {display_result}")
                    
                    # Add the assistant's response and tool result to the conversation
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": f"Tool result for {tool_name}:\n{result_text}\n\n"
                                   f"Please continue helping the user based on this result."
                    })
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": f"Tool call failed with error: {e}\n\n"
                                   f"Please try a different approach or explain the issue."
                    })
            else:
                # =============================================================
                # FINAL RESPONSE - NO TOOL CALL
                # =============================================================
                # Clean up any remaining tool call formatting
                final_response = re.sub(
                    r'<tool_call>.*?</tool_call>', 
                    '', 
                    response, 
                    flags=re.DOTALL
                )
                return final_response.strip()
        
        return "I apologize, but I reached the maximum number of tool calls. Please try a simpler query."
    
    async def chat_loop(self):
        """
        Run an interactive chat loop.
        
        This provides a simple REPL (Read-Eval-Print Loop) interface
        where users can type queries and see responses.
        """
        print("\n" + "=" * 60)
        print("  MCP Client (OCI GenAI) Ready!")
        print("  Type 'quit' to exit.")
        print("=" * 60 + "\n")
        
        while True:
            try:
                # Get user input
                query = input("You: ").strip()
                
                # Check for exit commands
                if query.lower() in ('quit', 'exit', 'q'):
                    print("Goodbye!")
                    break
                
                # Skip empty input
                if not query:
                    continue
                
                # Process the query
                print("\nProcessing...")
                response = await self.process_query(query)
                print(f"\nAssistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
    
    async def cleanup(self):
        """
        Clean up all resources.
        
        This closes the MCP connection and any other async resources.
        Called automatically when using 'async with' or in finally block.
        """
        await self.exit_stack.aclose()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """
    Main entry point - loads configuration and runs the client.
    """
    # =================================================================
    # LOAD CONFIGURATION
    # =================================================================
    # All configuration is loaded from environment variables
    # These can be set in a .env file in the same directory
    
    server_url = os.getenv("MCP_SERVER_URL")
    access_token = os.getenv("MCP_ACCESS_TOKEN")
    compartment_id = os.getenv("OCI_COMPARTMENT_ID")
    model_id = os.getenv("OCI_GENAI_MODEL_ID")
    endpoint = os.getenv(
        "OCI_GENAI_ENDPOINT", 
        "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com"
    )
    config_file = os.getenv("OCI_CONFIG_FILE", "~/.oci/config")
    config_profile = os.getenv("OCI_CONFIG_PROFILE", "DEFAULT")
    auth_type = os.getenv("OCI_AUTH_TYPE", "security_token")
    
    # =================================================================
    # VALIDATE CONFIGURATION
    # =================================================================
    missing = []
    if not server_url:
        missing.append("MCP_SERVER_URL")
    if not access_token:
        missing.append("MCP_ACCESS_TOKEN")
    if not compartment_id:
        missing.append("OCI_COMPARTMENT_ID")
    if not model_id:
        missing.append("OCI_GENAI_MODEL_ID")
    
    if missing:
        print("=" * 60)
        print("ERROR: Missing required environment variables:")
        print("=" * 60)
        for var in missing:
            print(f"  • {var}")
        print("\nCreate a .env file with the following content:")
        print("-" * 60)
        print("MCP_SERVER_URL=https://your-mcp-server.com/api/mcp")
        print("MCP_ACCESS_TOKEN=your-bearer-token")
        print("OCI_COMPARTMENT_ID=ocid1.compartment.oc1..aaaa...")
        print("OCI_GENAI_MODEL_ID=ocid1.generativeaimodel.oc1..aaaa...")
        print("OCI_GENAI_ENDPOINT=https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com")
        print("OCI_CONFIG_FILE=~/.oci/config")
        print("OCI_CONFIG_PROFILE=DEFAULT")
        print("OCI_AUTH_TYPE=security_token")
        print("-" * 60)
        return
    
    # =================================================================
    # INITIALIZE AND RUN
    # =================================================================
    print("=" * 60)
    print("  Initializing MCP Client with OCI GenAI")
    print("=" * 60 + "\n")
    
    try:
        # Create OCI GenAI client
        print("Step 1: Connecting to OCI Generative AI...")
        genai_client = OCIGenAIClient(
            compartment_id=compartment_id,
            model_id=model_id,
            endpoint=endpoint,
            config_file=config_file,
            config_profile=config_profile,
            auth_type=auth_type
        )
        
        # Create MCP client
        print("\nStep 2: Connecting to MCP Server...")
        client = MCPClient(genai_client)
        
        # Connect to server and run chat loop
        await client.connect_to_server(server_url, access_token)
        await client.chat_loop()
        
    except Exception as e:
        print(f"\nFailed to initialize: {e}")
        print("\nTroubleshooting:")
        print("  1. If using security_token auth, run: oci session authenticate")
        print("  2. Verify your OCI config file exists: ~/.oci/config")
        print("  3. Check that the MCP server URL is correct and accessible")
        print("  4. Verify your MCP access token is valid")
        
    finally:
        # Always cleanup, even if there's an error
        if 'client' in locals():
            await client.cleanup()


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
# This block runs when the script is executed directly (not imported)

if __name__ == "__main__":
    # asyncio.run() creates an event loop and runs our async main function
    asyncio.run(main())