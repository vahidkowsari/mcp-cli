# AI MCP Host

A Python application that hosts AI assistants (OpenAI GPT or Claude) with Model Context Protocol (MCP) integration, allowing AI to use external tools and services. **Now with library-like architecture for easy integration into other applications!**

## Features

- 🤖 **Multi-AI Support**: Works with both OpenAI GPT and Claude
- 🔧 **MCP Integration**: Connect to various MCP servers for extended functionality
- 🎯 **Tool Calling**: AI can automatically use available MCP tools
- ⚙️ **Configurable**: Easy JSON-based configuration
- 💬 **Interactive CLI**: Chat interface with command support
- 🔌 **Extensible**: Add new MCP servers easily
- 📚 **Library-like**: Core chat functionality can be used programmatically in other applications

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-mcp-host

# Quick setup with Makefile (recommended)
make install

# Or manual setup:
# pip install -r requirements.txt
# Install Node.js (required for MCP servers)
# On macOS: brew install node
# On Ubuntu: sudo apt install nodejs npm
```

### 2. Configuration

**Note**: Configuration files are automatically created when you run `make install`.

#### API Keys Setup

Set up your API keys by creating a `.env` file:

```bash
# Create .env file with your API keys
cat > .env << EOF
# Choose one or both AI providers
ANTHROPIC_API_KEY=your-anthropic-api-key-here
OPENAI_API_KEY=your-openai-api-key-here

# Optional: MCP server tokens (configure as needed)
# SLACK_MCP_XOXP_TOKEN=your-slack-bot-token
# SLACK_TEAM_ID=your-slack-team-id
# NOTION_API_KEY=your-notion-api-key
# SUPABASE_ACCESS_TOKEN=your-supabase-token
EOF
```

**Important**: Never commit `.env` files to git. They are automatically ignored by `.gitignore`.

#### MCP Servers Configuration

Configure MCP servers by editing `mcp_config.json`:

```bash
# Copy the example file (done automatically by make install)
cp mcp_config.example.json mcp_config.json

# Edit mcp_config.json to configure your desired MCP servers
```

Example configuration:

```json
{
  "ai": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key_env": "OPENAI_API_KEY"
  },
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### 3. Run the Application

```bash
# Using Makefile (recommended)
make run

# Or directly
python mcp-use-cli.py

# Or using the run script
./run_mcp_use.sh
```

## 🛠️ Makefile Commands

The project includes a comprehensive Makefile for easy development:

```bash
# Setup and installation
make install          # Complete setup (dependencies, venv, env)
make setup           # Alias for install

# Development
make run             # Start the application
make dev             # Alias for run
make test            # Run tests
make lint            # Run linting checks
make format          # Format code with black
make check           # Run lint + test

# Utilities
make clean           # Clean cache files
make clean-all       # Clean everything including venv
make status          # Show system status
make logs            # Show recent logs
make update          # Update dependencies

# Quick shortcuts
make i               # Quick install
make r               # Quick run
make c               # Quick clean
make t               # Quick test
make l               # Quick lint
make f               # Quick format

# Help
make help            # Show all available commands
```

## Available MCP Servers

The configuration includes several pre-configured MCP servers:

- **Filesystem**: File system operations
- **Brave Search**: Web search capabilities
- **SQLite**: Database operations
- **GitHub**: GitHub repository access
- **PostgreSQL**: PostgreSQL database operations

Enable them by setting `"disabled": false` and providing required API keys.

## Usage

### Chat Commands

- `help` - Show available commands
- `clear` - Clear conversation history
- `tools` - List available MCP tools
- `status` - Show MCP connection status
- `quit` - Exit application

### Example Interaction

```
You: Can you search for information about Python async programming?

🤖 Assistant: I'll search for information about Python async programming for you.

🔧 Calling tool: brave_search
✅ Tool result: {...}

Based on my search, here's what I found about Python async programming...
```

## Configuration Options

### AI Provider Settings

```json
{
  "ai": {
    "provider": "openai",     // "openai" or "claude"
    "model": "gpt-4",         // Model name
    "api_key_env": "OPENAI_API_KEY"  // Environment variable name
  }
}
```

### MCP Server Configuration

```json
{
  "mcpServers": {
    "server_name": {
      "command": "npx",                    // Command to run
      "args": ["-y", "package-name"],      // Command arguments
      "env": {                             // Environment variables
        "API_KEY": "your-key"
      },
      "disabled": false,                   // Enable/disable server
      "autoApprove": ["tool_name"]         // Auto-approve specific tools
    }
  }
}
```

## Adding New MCP Servers

1. Find an MCP server package (usually npm packages)
2. Add configuration to `mcp_config.json`
3. Set required environment variables
4. Enable the server and restart

Example:
```json
{
  "my_custom_mcp": {
    "command": "python",
    "args": ["-m", "my_mcp_server"],
    "env": {
      "CUSTOM_API_KEY": "key-here"
    },
    "disabled": false,
    "autoApprove": ["safe_tool_name"]
  }
}
```

## Troubleshooting

### Common Issues

1. **MCP Server Not Starting**
   - Check if Node.js is installed
   - Verify command and arguments in config
   - Check environment variables

2. **AI API Errors**
   - Verify API key is set correctly
   - Check internet connection
   - Ensure sufficient API credits

3. **Tool Calls Failing**
   - Check MCP server logs
   - Verify tool parameters
   - Ensure MCP server is connected

### Debug Mode

Enable debug logging by setting:
```bash
export PYTHONPATH=.
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python mcp-use-cli.py
```

## 🏗️ Architecture

### **Library-Like Design**

The application now features a library-like architecture where core chat functionality is separated from interface-specific code, making it easy to integrate into other applications.

#### **Before Refactoring:**
```
┌─────────────────┐
│   CLI Interface │ ← All chat logic embedded here
│                 │ ← Tightly coupled to console
│  • User input   │ ← Hard to reuse elsewhere
│  • AI calls     │
│  • Tool calls   │
│  • History mgmt │
└─────────────────┘
```

#### **After Refactoring:**
```
┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │───▶│   ChatService   │ ← Core library
│                 │    │    (Library)    │ ← Reusable
└─────────────────┘    └─────────────────┘
                              ▲
┌─────────────────┐           │
│   API Interface │───────────┘
│                 │
└─────────────────┘
                              ▲
┌─────────────────┐           │
│  Custom App     │───────────┘
│                 │
└─────────────────┘
```

### **Core Components**

#### **1. ChatService Class** (`chat_service.py`)
The main library that handles all chat functionality:

```python
class ChatService:
    def __init__(self, ai_client, mcp_manager, verbose=False, output_handler=None):
        # Core dependencies
        self.ai_client = ai_client          # AI/LLM client
        self.mcp_manager = mcp_manager      # MCP tools manager
        self.verbose = verbose              # Detail level
        self.output_handler = output_handler # Custom output handling
```

#### **2. Data Structures**
```python
@dataclass
class ChatMessage:
    role: str           # 'user', 'assistant', 'system'
    content: str        # Message content
    tool_calls: List    # Any tool calls made
    usage: Dict         # Token usage info

@dataclass
class ChatResponse:
    message: ChatMessage           # AI's response
    tool_calls_executed: List     # Tools that were executed
    success: bool                 # Whether chat succeeded
    error: Optional[str]          # Any error message
```

#### **3. Output Handler System**
Allows different interfaces to handle output differently:

```python
def custom_output_handler(message_type: str, content: str, data=None):
    if message_type == "assistant_response":
        # CLI: print to console
        # API: add to JSON response
        # GUI: update chat window
    elif message_type == "tool_call":
        # Handle tool execution notifications
```

### **Chat Flow Process**

Here's what happens when you send a message:

```
1. User Message
   ↓
2. ChatService.chat(message)
   ↓
3. Format available MCP tools for AI
   ↓
4. Call AI client with message + tools
   ↓
5. AI responds with content ± tool calls
   ↓
6. If tool calls exist:
   ├─ Execute tools via MCP manager
   ├─ Get tool results
   ├─ Send results back to AI
   └─ Get follow-up AI response
   ↓
7. Return structured ChatResponse
   ↓
8. Output handler displays results
```

### **Usage Examples**

#### **1. CLI Usage (Current)**
```python
# In cli_interface.py
self.chat_service = ChatService(ai_client, mcp_manager, verbose=self.verbose)
response = await self.chat_service.chat(user_input)
# Output automatically handled by default console handler
```

#### **2. Programmatic Usage**
```python
# In your own code
from chat_service import ChatService

# Initialize
chat_service = ChatService(ai_client, mcp_manager, verbose=True)

# Use directly
response = await chat_service.chat("What tools do you have?")
print(f"AI said: {response.message.content}")
print(f"Tools used: {len(response.tool_calls_executed)}")
```

#### **3. API Usage**
```python
# In a web API
async def handle_chat_request(message: str):
    # Custom output handler for API responses
    api_output = []
    
    def api_handler(msg_type, content, data=None):
        api_output.append({"type": msg_type, "content": content})
    
    chat_service = ChatService(ai_client, mcp_manager, 
                              output_handler=api_handler)
    
    response = await chat_service.chat(message)
    
    return {
        "response": response.message.content,
        "tools_used": response.tool_calls_executed,
        "output_log": api_output
    }
```

### **Key Benefits**

#### **1. Separation of Concerns**
- **ChatService**: Core chat logic, AI interactions, tool management
- **CLIInterface**: User interface, command parsing, console output
- **Other interfaces**: Can focus on their specific UI/API concerns

#### **2. Reusability**
- Same chat logic works in CLI, web APIs, desktop apps, etc.
- No code duplication
- Consistent behavior across interfaces

#### **3. Customizable Output**
- CLI: Prints to console with colors/emojis
- API: Collects output for JSON responses  
- GUI: Updates chat windows/notifications
- Custom: Whatever you need

#### **4. Easy Testing**
- ChatService can be tested independently
- Mock output handlers for testing
- Clear separation makes debugging easier

### **How to Extend**

To add a new interface (e.g., web API):

1. **Create your interface class**
2. **Initialize ChatService with custom output handler**
3. **Call `chat_service.chat(message)` for interactions**
4. **Handle the structured response as needed**

The core chat functionality remains the same - you just change how input comes in and output goes out!

### **File Structure**

- `mcp-use-cli.py` - Application entry point using mcp-use library
- `chat_service.py` - **Core chat library (NEW)** - reusable chat functionality
- `ai_client.py` - AI provider abstraction (OpenAI/Claude)
- `mcp_manager.py` - MCP connection and tool management
- `cli_interface.py` - Interactive command-line interface (now uses ChatService)
- `mcp_config.json` - Configuration file

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
