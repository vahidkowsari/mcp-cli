"""
Chat Service - Core chat functionality that can be used by CLI, API, or other interfaces
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from ai_client import AIClient
from mcp_manager import MCPManager

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, Any]] = None


@dataclass
class ChatResponse:
    """Response from chat service"""
    message: ChatMessage
    tool_calls_executed: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None


class ChatService:
    """
    Core chat service that handles AI interactions and tool calls.
    Can be used by CLI, API, or other interfaces.
    """
    
    def __init__(self, ai_client: Optional[AIClient], mcp_manager: MCPManager, 
                 verbose: bool = False, output_handler: Optional[Callable] = None):
        """
        Initialize chat service
        
        Args:
            ai_client: AI client for chat interactions
            mcp_manager: MCP manager for tool access
            verbose: Whether to show detailed output
            output_handler: Optional callback for handling output (for custom UIs)
        """
        self.ai_client = ai_client
        self.mcp_manager = mcp_manager
        self.verbose = verbose
        self.output_handler = output_handler or self._default_output_handler
        
    def _default_output_handler(self, message_type: str, content: str, data: Optional[Dict] = None):
        """Default output handler that prints to console"""
        if message_type == "thinking":
            print("🤔 Thinking...", flush=True)
        elif message_type == "assistant_response":
            print(f"\n🤖 Assistant: {content}")
        elif message_type == "tool_call":
            print(f"\n🔧 Calling tool: {content}")
            if self.verbose and data:
                print(f"📝 Arguments: {json.dumps(data, indent=2)}")
        elif message_type == "tool_result":
            if self.verbose and data:
                print(f"✅ Tool result: {json.dumps(data, indent=2)}")
            else:
                print(f"✅ Tool executed successfully")
        elif message_type == "tool_error":
            print(f"❌ Tool call failed: {content}")
        elif message_type == "usage":
            if data:
                input_tokens = data.get("input_tokens", data.get("prompt_tokens", 0))
                output_tokens = data.get("output_tokens", data.get("completion_tokens", 0))
                print(f"📊 Tokens used: {input_tokens} input, {output_tokens} output")
        elif message_type == "error":
            print(f"❌ Error: {content}")
    
    async def chat(self, user_message: str) -> ChatResponse:
        """
        Process a chat message and return response
        
        Args:
            user_message: User's message
            
        Returns:
            ChatResponse with AI response and any tool calls executed
        """
        if self.ai_client is None:
            error_msg = "AI client not available. Please set your API key and restart."
            self.output_handler("error", error_msg)
            return ChatResponse(
                message=ChatMessage(role="system", content=error_msg),
                tool_calls_executed=[],
                success=False,
                error=error_msg
            )
        
        try:
            # Get available tools for AI
            available_tools = self._format_tools_for_ai(self.mcp_manager.get_available_tools())
            
            self.output_handler("thinking", "")
            
            # Get AI response
            response = await self.ai_client.chat(user_message, available_tools)
            
            # Create response message
            chat_message = ChatMessage(
                role="assistant",
                content=response.get("content", ""),
                tool_calls=response.get("tool_calls"),
                usage=response.get("usage")
            )
            
            # Show AI response first (before tool calls)
            if response.get("content"):
                self.output_handler("assistant_response", response["content"])
            
            tool_calls_executed = []
            
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
                
                self.ai_client.conversation_history.append(assistant_message)
                
                # Handle tool calls and get follow-up response
                tool_calls_executed = await self._handle_tool_calls_with_followup(response["tool_calls"])
                
            elif response.get("content"):
                # If no tool calls, add simple text response to history
                self.ai_client.add_message("assistant", response["content"])
            
            # Show usage info if available
            if response.get("usage"):
                self.output_handler("usage", "", response["usage"])
            
            return ChatResponse(
                message=chat_message,
                tool_calls_executed=tool_calls_executed,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            error_msg = str(e)
            self.output_handler("error", error_msg)
            return ChatResponse(
                message=ChatMessage(role="system", content=f"Error: {error_msg}"),
                tool_calls_executed=[],
                success=False,
                error=error_msg
            )
    
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
    
    async def _handle_tool_calls_with_followup(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle tool calls and get follow-up response from AI."""
        tool_results = []
        tool_calls_executed = []
        
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
                
                # Show tool call info
                self.output_handler("tool_call", tool_name, arguments)
                
                # Execute the tool
                result = await self.mcp_manager.call_tool(tool_name, arguments)
                
                # Show result
                self.output_handler("tool_result", "", result)
                
                # Store result for adding to conversation
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": json.dumps(result, indent=2)
                })
                
                # Store executed tool call info
                tool_calls_executed.append({
                    "name": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "success": True
                })
                
            except Exception as e:
                error_msg = str(e)
                self.output_handler("tool_error", error_msg)
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": f"Error: {error_msg}",
                    "is_error": True
                })
                
                tool_calls_executed.append({
                    "name": tool_name,
                    "arguments": arguments,
                    "error": error_msg,
                    "success": False
                })
        
        # Add all tool results to conversation as a single user message
        if tool_results:
            tool_result_message = {
                "role": "user",
                "content": tool_results
            }
            self.ai_client.conversation_history.append(tool_result_message)
            
            # Get follow-up response from AI
            available_tools = self._format_tools_for_ai(self.mcp_manager.get_available_tools())
            follow_up_response = await self.ai_client.chat("", available_tools)
            
            if follow_up_response.get("content"):
                self.output_handler("assistant_response", follow_up_response["content"])
            
            # Handle any additional tool calls in the follow-up
            if follow_up_response.get("tool_calls"):
                # Add follow-up assistant message to history
                follow_up_assistant_message = {
                    "role": "assistant",
                    "content": []
                }
                
                if follow_up_response.get("content"):
                    follow_up_assistant_message["content"].append({
                        "type": "text",
                        "text": follow_up_response["content"]
                    })
                
                for tool_call in follow_up_response["tool_calls"]:
                    follow_up_assistant_message["content"].append({
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "input": tool_call["input"]
                    })
                
                self.ai_client.conversation_history.append(follow_up_assistant_message)
                
                # Recursively handle additional tool calls
                additional_tool_calls = await self._handle_tool_calls_with_followup(follow_up_response["tool_calls"])
                tool_calls_executed.extend(additional_tool_calls)
            else:
                # Add simple text response to history
                if follow_up_response.get("content"):
                    self.ai_client.add_message("assistant", follow_up_response["content"])
        
        return tool_calls_executed
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history"""
        if self.ai_client:
            return self.ai_client.conversation_history
        return []
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        if self.ai_client:
            self.ai_client.conversation_history = []
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available MCP tools"""
        return self.mcp_manager.get_available_tools()
    
    def get_mcp_status(self) -> Dict[str, Any]:
        """Get MCP connection status"""
        status = {}
        for name, connection in self.mcp_manager.connections.items():
            status[name] = {
                "connected": connection.is_connected,
                "tool_count": len(connection.tools) if connection.is_connected else 0,
                "resource_count": len(connection.resources) if connection.is_connected else 0
            }
        return status
