# AI MCP Host

A Python application that hosts AI assistants (OpenAI GPT or Claude) with Model Context Protocol (MCP) integration, allowing AI to use external tools and services.

## Features

- 🤖 **Multi-AI Support**: Works with both OpenAI GPT and Claude
- 🔧 **MCP Integration**: Connect to various MCP servers for extended functionality
- 🎯 **Tool Calling**: AI can automatically use available MCP tools
- ⚙️ **Configurable**: Easy JSON-based configuration
- 💬 **Interactive CLI**: Chat interface with command support
- 🔌 **Extensible**: Add new MCP servers easily

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
# Copy the example file (done automatically by make install)
cp .env.example .env

# Edit .env and add your API keys
# ANTHROPIC_API_KEY=your-anthropic-api-key
# OPENAI_API_KEY=your-openai-api-key
```

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
python main.py

# Or using the run script
./run.sh
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
python main.py
```

## Architecture

- `main.py` - Application entry point and orchestration
- `ai_client.py` - AI provider abstraction (OpenAI/Claude)
- `mcp_manager.py` - MCP connection and tool management
- `cli_interface.py` - Interactive command-line interface
- `mcp_config.json` - Configuration file

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
