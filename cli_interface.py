"""
Command Line Interface for the AI MCP Host
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from ai_client import AIClient
from mcp_manager import MCPManager

logger = logging.getLogger(__name__)


class CLIInterface:
    """Command line interface for interacting with AI and MCPs"""
    
    def __init__(self):
        self.running = False
    
    async def start(self, ai_client: Optional[AIClient], mcp_manager: MCPManager):
        """Start the CLI interface"""
        self.running = True
        
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
                    if ai_client:
                        ai_client.clear_history()
                        print("🧹 Conversation history cleared")
                    else:
                        print("⚠️  AI client not available")
                elif user_input.lower() == 'tools':
                    self._show_tools(mcp_manager)
                elif user_input.lower() == 'status':
                    self._show_status(mcp_manager)
                else:
                    await self._handle_chat(user_input, ai_client, mcp_manager)
                    
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
        print("\n💬 Chat with the AI by typing your message")
        print("   The AI can use available MCP tools automatically")
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
    
    async def _handle_chat(self, user_message: str, ai_client: Optional[AIClient], mcp_manager: MCPManager):
        """Handle chat interaction with AI"""
        if ai_client is None:
            print("⚠️  AI client not available. Please set your API key and restart.")
            return
        
        try:
            print("🤔 Thinking...")
            
            # Get available tools for AI
            available_tools = self._format_tools_for_ai(mcp_manager.get_available_tools())
            
            # Get AI response
            response = await ai_client.chat(user_message, available_tools)
            
            # Handle tool calls if present
            if response.get("tool_calls"):
                # Add the assistant's response with tool calls to history first
                assistant_message = {
                    "role": "assistant",
                    "content": []
                }
                
                # Add text content if present
                if response.get("content"):
                    assistant_message["content"].append({
                        "type": "text",
                        "text": response["content"]
                    })
                
                # Add tool use blocks
                for tool_call in response["tool_calls"]:
                    assistant_message["content"].append({
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "input": tool_call["input"]
                    })
                
                ai_client.conversation_history.append(assistant_message)
                
                # Handle tool calls and get follow-up response
                await self._handle_tool_calls_with_followup(response["tool_calls"], ai_client, mcp_manager)
            
            # Show AI response
            if response.get("content"):
                print(f"\n🤖 Assistant: {response['content']}")
            
            # Show usage info if available
            if response.get("usage"):
                usage = response["usage"]
                if isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
                    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
                    print(f"📊 Tokens used: {input_tokens} input, {output_tokens} output")
            
            print()
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            print(f"❌ Error: {e}\n")
    
    def _format_tools_for_ai(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format MCP tools for AI consumption"""
        formatted_tools = []
        
        for tool in tools:
            # Format for Claude API
            formatted_tool = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema", {})
            }
            formatted_tools.append(formatted_tool)
        
        return formatted_tools
    
    async def _handle_tool_calls_with_followup(self, tool_calls: List[Dict[str, Any]], ai_client: AIClient, mcp_manager: MCPManager):
        """Handle tool calls and get follow-up response from AI."""
        tool_results = []
        
        for tool_call in tool_calls:
            try:
                # Extract tool information based on format (OpenAI vs Claude)
                if hasattr(tool_call, 'function'):
                    # OpenAI format
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    call_id = tool_call.id
                else:
                    # Claude format
                    tool_name = tool_call.get("name", "")
                    arguments = tool_call.get("input", {})
                    call_id = tool_call.get("id", "")
                
                print(f"\n🔧 Calling tool: {tool_name}")
                print(f"📝 Arguments: {json.dumps(arguments, indent=2)}")
                
                # Execute the tool
                result = await mcp_manager.call_tool(tool_name, arguments)
                
                print(f"✅ Tool result: {json.dumps(result, indent=2)}")
                
                # Store result for adding to conversation
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": json.dumps(result, indent=2)
                })
                
            except Exception as e:
                print(f"❌ Tool call failed: {str(e)}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": f"Error: {str(e)}",
                    "is_error": True
                })
        
        # Add all tool results to conversation as a single user message
        if tool_results:
            tool_result_message = {
                "role": "user",
                "content": tool_results
            }
            ai_client.conversation_history.append(tool_result_message)
            
            # Get follow-up response from AI
            available_tools = self._format_tools_for_ai(mcp_manager.get_available_tools())
            follow_up_response = await ai_client.chat("", available_tools)
            
            if follow_up_response.get("content"):
                print(f"\n🤖 Assistant: {follow_up_response['content']}")
            
            # Handle any additional tool calls in the follow-up
            if follow_up_response.get("tool_calls"):
                # Add the follow-up assistant message with tool calls to history
                followup_assistant_message = {
                    "role": "assistant",
                    "content": []
                }
                
                # Add text content if present
                if follow_up_response.get("content"):
                    followup_assistant_message["content"].append({
                        "type": "text",
                        "text": follow_up_response["content"]
                    })
                
                # Add tool use blocks
                for tool_call in follow_up_response["tool_calls"]:
                    followup_assistant_message["content"].append({
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "input": tool_call["input"]
                    })
                
                ai_client.conversation_history.append(followup_assistant_message)
                
                # Handle the follow-up tool calls
                await self._handle_tool_calls_with_followup(follow_up_response["tool_calls"], ai_client, mcp_manager)

    async def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]], ai_client: AIClient, mcp_manager: MCPManager):
        """Handle tool calls from AI"""
        for tool_call in tool_calls:
            try:
                # Extract tool call information
                if hasattr(tool_call, 'function'):
                    # OpenAI format
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    call_id = tool_call.id
                else:
                    # Claude format (simplified)
                    tool_name = tool_call.get("name", "")
                    arguments = tool_call.get("input", {})
                    call_id = tool_call.get("id", "")
                
                print(f"🔧 Calling tool: {tool_name}")
                
                # Execute tool
                result = await mcp_manager.call_tool(tool_name, arguments)
                
                # Add tool result to conversation (Claude format)
                tool_result = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": json.dumps(result, indent=2)
                        }
                    ]
                }
                ai_client.conversation_history.append(tool_result)
                
                print(f"✅ Tool result: {json.dumps(result, indent=2)[:200]}...")
                
            except Exception as e:
                logger.error(f"Tool call error: {e}")
                print(f"❌ Tool call failed: {e}")
                
                # Add error to conversation (Claude format)
                error_result = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call.get('id', ''),
                            "content": f"Error: {str(e)}",
                            "is_error": True
                        }
                    ]
                }
                ai_client.conversation_history.append(error_result)
