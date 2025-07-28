#!/usr/bin/env python3
"""
MCP-Use CLI - A simplified MCP client using the mcp-use library
Now refactored to use the MCP Library for clean separation of concerns
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Import our MCP library
from mcp_library import MCPLibrary, MCPConfig, MCPResponse


class MCPUseCLI:
    """CLI interface for MCP Library"""
    
    def __init__(self, config: MCPConfig, output_format: str = "text", quiet: bool = False):
        self.config = config
        self.output_format = output_format  # text, json, markdown
        self.quiet = quiet
        self.library = MCPLibrary(config)
        
    async def initialize(self):
        """Initialize the MCP library"""
        if not self.quiet:
            print("🚀 Initializing MCP connections...")
        
        result = await self.library.initialize()
        
        if not result.success:
            raise Exception(f"Failed to initialize MCP library: {result.error}")
        
        if not self.quiet:
            provider_info = result.metadata.get('llm_provider', 'unknown')
            server_count = result.metadata.get('server_count', 0)
            servers = result.metadata.get('servers', [])
            
            print(f"🤖 Using {provider_info.title()} LLM")
            print(f"📋 Connected to {server_count} MCP servers:")
            for server in servers:
                print(f"  • {server}: Connected")
            print()
    
    def format_output(self, content: str, is_response: bool = True) -> str:
        """Format output based on selected format"""
        if self.output_format == "json":
            return json.dumps({
                "type": "response" if is_response else "info",
                "content": content,
                "timestamp": asyncio.get_event_loop().time()
            }, indent=2)
        elif self.output_format == "markdown":
            if is_response:
                return f"## Assistant Response\n\n{content}\n"
            else:
                return f"*{content}*\n"
        else:  # text format
            return content
    
    def print_output(self, content: str, is_response: bool = True, force: bool = False):
        """Print formatted output respecting quiet mode"""
        if self.quiet and not force:
            return
        
        formatted = self.format_output(content, is_response)
        print(formatted)
    
    async def run_batch_mode(self, commands: list):
        """Run commands in batch mode using the library"""
        if not self.quiet:
            self.print_output(f"🔄 Running {len(commands)} commands in batch mode...", False)
        
        # Use library's batch query method
        results = await self.library.batch_query(commands)
        
        # Display results
        for i, result in enumerate(results, 1):
            if not self.quiet:
                command = result.metadata.get('prompt', f'Command {i}') if result.metadata else f'Command {i}'
                self.print_output(f"[{i}/{len(results)}] Processed: {command}", False)
            
            # Display response based on format
            if self.output_format == "json":
                self.print_output(result.to_json(), True, True)
            else:
                if result.success:
                    self.print_output(f"\n🤖 Assistant: {result.content}\n", True, True)
                else:
                    self.print_output(f"❌ Error: {result.error}\n", False, True)
        
        return [result.to_dict() for result in results]
    
    async def run_interactive(self):
        """Run the interactive CLI loop using the library"""
        if not self.quiet:
            self.print_output("✅ MCP-Use CLI ready!", False)
            self.print_output("Type your requests or 'quit' to exit", False)
            print()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if user_input.lower() in ['help', 'h']:
                    self.show_help()
                    continue
                
                # Show thinking indicator
                if not self.quiet:
                    print("🤔 Thinking...")
                
                # Use library to process the query
                result = await self.library.query(user_input)
                
                # Display response
                if result.success:
                    self.print_output(f"\n🤖 Assistant: {result.content}\n", True)
                else:
                    self.print_output(f"❌ Error: {result.error}\n", False)
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                if self.config.verbose:
                    import traceback
                    traceback.print_exc()
    
    def show_help(self):
        """Show help information"""
        print("""
📖 MCP-Use CLI Help:

Commands:
  help, h     - Show this help message
  quit, q     - Exit the CLI
  
Usage:
  Simply type your requests in natural language!
  
Examples:
  • "list slack channels"
  • "send a message to #general saying hello"
  • "show my recent Teams messages"
  • "create a new Notion page"
  • "browse to google.com"
  
The AI will automatically use the appropriate MCP tools to fulfill your requests.
        """)
    
    async def cleanup(self):
        """Clean up resources using the library"""
        if not self.quiet:
            print("🧹 Cleaning up... (cleanup warnings are normal and can be ignored)")
        
        result = await self.library.cleanup()
        if not result.success and self.config.verbose:
            print(f"Note: Some cleanup warnings during shutdown (non-critical): {result.error}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MCP-Use CLI - Interactive assistant with MCP tool integration"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "-c", "--config",
        default="mcp_config.json",
        help="Path to MCP configuration file (default: mcp_config.json)"
    )
    parser.add_argument(
        "--batch",
        help="Run commands from file (one command per line)"
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read commands from stdin"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only responses)"
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4-turbo-preview",
        help="OpenAI model to use (default: gpt-4-turbo-preview)"
    )
    parser.add_argument(
        "--anthropic-model",
        default="claude-3-5-sonnet-20241022",
        help="Anthropic model to use (default: claude-3-5-sonnet-20241022)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)"
    )
    parser.add_argument(
        "--preferred-provider",
        choices=["openai", "anthropic"],
        help="Preferred LLM provider (auto-detect if not specified)"
    )
    parser.add_argument(
        "--prompt",
        help="Execute a single prompt and exit"
    )
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🚀 Starting MCP-Use CLI...")
    
    # Create configuration
    config = MCPConfig(
        config_path=args.config,
        openai_model=args.openai_model,
        anthropic_model=args.anthropic_model,
        temperature=args.temperature,
        verbose=args.verbose,
        preferred_provider=args.preferred_provider
    )
    
    cli = MCPUseCLI(
        config=config,
        output_format=args.format,
        quiet=args.quiet
    )
    
    try:
        # Initialize MCP connections (includes LLM setup)
        await cli.initialize()
        
        # Determine mode: prompt, batch, stdin, or interactive
        if args.prompt:
            # Execute single prompt and exit
            response = await cli.library.query(args.prompt)
            if response.success:
                print(cli.format_output(response.content))
            else:
                print(f"❌ Error: {response.error}")
                return 1
                
        elif args.batch:
            # Read commands from file
            batch_file = Path(args.batch)
            if not batch_file.exists():
                print(f"❌ Error: Batch file '{args.batch}' not found")
                return
            
            commands = batch_file.read_text().strip().split('\n')
            await cli.run_batch_mode(commands)
            
        elif args.stdin:
            # Read commands from stdin
            commands = []
            try:
                import sys
                if sys.stdin.isatty():
                    print("Error: No input provided via stdin")
                    return
                
                for line in sys.stdin:
                    line = line.strip()
                    if line:  # Only add non-empty lines
                        commands.append(line)
            except (KeyboardInterrupt, EOFError):
                pass
            
            if commands:
                await cli.run_batch_mode(commands)
            else:
                if not args.quiet:
                    print("No commands provided via stdin")
            
        else:
            # Run interactive loop
            await cli.run_interactive()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    finally:
        if cli.library:
            await cli.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
