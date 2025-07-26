"""
Command Line Interface for the AI MCP Host
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from ai_client import AIClient
from mcp_manager import MCPManager
from chat_service import ChatService

logger = logging.getLogger(__name__)


class CLIInterface:
    """Command line interface for interacting with AI and MCPs"""
    
    def __init__(self, verbose: bool = False):
        self.running = False
        self.verbose = verbose
        self.chat_service = None
    
    async def start(self, ai_client: Optional[AIClient], mcp_manager: MCPManager):
        """Start the CLI interface"""
        self.running = True
        
        # Initialize chat service
        self.chat_service = ChatService(ai_client, mcp_manager, verbose=self.verbose)
        
        print("\n🤖 AI MCP Host - Interactive Assistant")
        print("=" * 50)
        
        if ai_client is None:
            print("⚠️  AI client not available. Please set your API key and restart.")
            print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        else:
            print("✅ AI client ready")
        
        print("Type 'help' for commands, 'quit' to exit")
        print()
        
        # Show available tools grouped by MCP
        tools = mcp_manager.get_available_tools()
        if tools:
            print(f"📋 Available tools from MCPs: {len(tools)}")
            
            # Group tools by MCP server
            tools_by_mcp = {}
            for tool in tools:
                mcp_name = tool.get('mcp_name', 'Unknown MCP')
                tool_name = tool.get('name', 'Unknown')
                if mcp_name not in tools_by_mcp:
                    tools_by_mcp[mcp_name] = []
                tools_by_mcp[mcp_name].append(tool_name)
            
            # Display one line per MCP with all its tools
            for mcp_name, tool_names in tools_by_mcp.items():
                tools_str = ", ".join(tool_names)
                print(f"  • {mcp_name} ({len(tool_names)}): {tools_str}")
        else:
            print("⚠️  No MCP tools available. Check your mcp_config.json")
        print()
        
        while self.running:
            try:
                user_input = await self._get_user_input()
                
                if not user_input.strip():
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                elif user_input.lower() == 'clear':
                    if self.chat_service:
                        self.chat_service.clear_conversation_history()
                        print("🧹 Conversation history cleared")
                    else:
                        print("⚠️  Chat service not available")
                elif user_input.lower() == 'tools':
                    self._show_tools(mcp_manager)
                elif user_input.lower() == 'status':
                    self._show_status(mcp_manager)
                else:
                    await self._handle_chat(user_input)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"CLI error: {e}")
                print(f"❌ Error: {e}")
        
        print("\n👋 Goodbye!")
    
    async def _get_user_input(self) -> str:
        """Get user input asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, input, "You: ")
    
    def _show_help(self):
        """Show help information"""
        print("\n📖 Available Commands:")
        print("  help    - Show this help message")
        print("  clear   - Clear conversation history")
        print("  tools   - List available MCP tools")
        print("  status  - Show MCP connection status")
        print("  quit    - Exit the application")
        print("\n📝 Chat with the AI by typing your message")
        print("   The AI can use available MCP tools automatically")
        print(f"\n🔍 Verbose mode: {'ON' if self.verbose else 'OFF'}")
        if self.verbose:
            print("   - Shows tool arguments and full results")
        else:
            print("   - Shows only tool names and success status")
            print("   - Use --verbose flag to see full communication")
        print()
    
    def _show_tools(self, mcp_manager: MCPManager):
        """Show available MCP tools"""
        tools = mcp_manager.get_available_tools()
        
        print(f"\n🔧 Available Tools ({len(tools)}):")
        if not tools:
            print("  No tools available")
            return
        
        # Group tools by MCP server
        tools_by_mcp = {}
        for tool in tools:
            mcp_name = tool.get('mcp_name', 'Unknown MCP')
            tool_name = tool.get('name', 'Unknown')
            if mcp_name not in tools_by_mcp:
                tools_by_mcp[mcp_name] = []
            tools_by_mcp[mcp_name].append(tool_name)
        
        # Display one line per MCP with all its tools
        for mcp_name, tool_names in tools_by_mcp.items():
            tools_str = ", ".join(tool_names)
            print(f"  • {mcp_name} ({len(tool_names)} tools): {tools_str}")
        print()
    
    def _show_status(self, mcp_manager: MCPManager):
        """Show MCP connection status"""
        print("\n📊 MCP Connection Status:")
        
        if not mcp_manager.connections:
            print("  No MCPs configured")
            return
        
        for name, connection in mcp_manager.connections.items():
            status = "🟢 Connected" if connection.is_connected else "🔴 Disconnected"
            tool_count = len(connection.tools) if connection.is_connected else 0
            
            print(f"  • {name}: {status}")
            if connection.is_connected:
                print(f"    Tools: {tool_count}, Resources: {len(connection.resources)}")
        print()
    
    async def _handle_chat(self, user_message: str):
        """Handle chat interaction with AI using chat service"""
        if not self.chat_service:
            print("⚠️  Chat service not available.")
            return
        
        # Use chat service to handle the conversation
        response = await self.chat_service.chat(user_message)
        
        # Print a newline after the response for better formatting
        print()
    

