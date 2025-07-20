"""
MCP Manager for handling Model Context Protocol connections and tool execution
"""

import asyncio
import json
import logging
import os
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MCPConnection:
    """Represents a connection to an MCP server"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.tools: List[Dict[str, Any]] = []
        self.resources: List[Dict[str, Any]] = []
        self.is_connected = False
        self.auto_approve_tools = set(config.get("autoApprove", []))
    
    async def start(self):
        """Start the MCP server process"""
        # Check if server is disabled (following Windsurf/Cursor schema)
        if self.config.get("disabled", False):
            logger.info(f"MCP {self.name} is disabled, skipping")
            return
        
        try:
            command = self.config.get("command", "")
            args = self.config.get("args", [])
            env = self.config.get("env", {})
            
            if not command:
                logger.error(f"No command specified for MCP {self.name}")
                return
            
            # Prepare environment
            full_env = dict(os.environ)
            full_env.update(env)
            
            # Start process
            full_command = [command] + args
            logger.info(f"Starting MCP {self.name}: {' '.join(full_command)}")
            
            self.process = subprocess.Popen(
                full_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=full_env,
                text=True
            )
            
            # Initialize MCP protocol
            await self._initialize_protocol()
            
            self.is_connected = True
            logger.info(f"MCP {self.name} connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to start MCP {self.name}: {e}")
            await self.stop()
    
    async def _initialize_protocol(self):
        """Initialize MCP protocol handshake"""
        try:
            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {}
                    },
                    "clientInfo": {
                        "name": "ai-mcp-host",
                        "version": "1.0.0"
                    }
                }
            }
            
            await self._send_request(init_request)
            response = await self._read_response()
            
            if response and response.get("result"):
                # Get available tools
                await self._list_tools()
                # Get available resources
                await self._list_resources()
                
        except Exception as e:
            logger.error(f"MCP protocol initialization failed for {self.name}: {e}")
            raise
    
    async def _send_request(self, request: Dict[str, Any]):
        """Send JSON-RPC request to MCP server"""
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP process not available")
        
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str)
        self.process.stdin.flush()
    
    async def _read_response(self) -> Optional[Dict[str, Any]]:
        """Read JSON-RPC response from MCP server"""
        if not self.process or not self.process.stdout:
            return None
        
        try:
            # Read line with timeout
            line = await asyncio.wait_for(
                asyncio.create_task(self._read_line()),
                timeout=5.0
            )
            
            if line:
                return json.loads(line.strip())
        except (asyncio.TimeoutError, json.JSONDecodeError) as e:
            logger.error(f"Error reading response from {self.name}: {e}")
        
        return None
    
    async def _read_line(self) -> str:
        """Read a line from stdout"""
        return self.process.stdout.readline()
    
    async def _list_tools(self):
        """List available tools from MCP server"""
        try:
            request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }
            
            await self._send_request(request)
            response = await self._read_response()
            
            if response and response.get("result"):
                self.tools = response["result"].get("tools", [])
                logger.info(f"MCP {self.name} loaded {len(self.tools)} tools")
                
        except Exception as e:
            logger.error(f"Failed to list tools for {self.name}: {e}")
    
    async def _list_resources(self):
        """List available resources from MCP server"""
        try:
            request = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "resources/list"
            }
            
            await self._send_request(request)
            response = await self._read_response()
            
            if response and response.get("result"):
                self.resources = response["result"].get("resources", [])
                logger.info(f"MCP {self.name} loaded {len(self.resources)} resources")
                
        except Exception as e:
            logger.error(f"Failed to list resources for {self.name}: {e}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        if not self.is_connected:
            raise RuntimeError(f"MCP {self.name} is not connected")
        
        try:
            request = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            await self._send_request(request)
            response = await self._read_response()
            
            if response and response.get("result"):
                return response["result"]
            elif response and response.get("error"):
                raise RuntimeError(f"Tool call error: {response['error']}")
            else:
                raise RuntimeError("No response from tool call")
                
        except Exception as e:
            logger.error(f"Tool call failed for {self.name}.{tool_name}: {e}")
            raise
    
    async def stop(self):
        """Stop the MCP server process"""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.sleep(1)
                if self.process.poll() is None:
                    self.process.kill()
            except Exception as e:
                logger.error(f"Error stopping MCP {self.name}: {e}")
            finally:
                self.process = None
                self.is_connected = False


class MCPManager:
    """Manages multiple MCP connections"""
    
    def __init__(self, mcp_configs: Dict[str, Dict[str, Any]]):
        self.connections: Dict[str, MCPConnection] = {}
        
        for name, config in mcp_configs.items():
            self.connections[name] = MCPConnection(name, config)
    
    async def initialize(self):
        """Initialize all MCP connections"""
        logger.info("Initializing MCP connections...")
        
        tasks = []
        for connection in self.connections.values():
            tasks.append(connection.start())
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        active_connections = [name for name, conn in self.connections.items() if conn.is_connected]
        logger.info(f"Active MCP connections: {active_connections}")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools from connected MCPs"""
        tools = []
        
        for name, connection in self.connections.items():
            if connection.is_connected:
                for tool in connection.tools:
                    # Add MCP name to tool for identification
                    tool_copy = tool.copy()
                    tool_copy["mcp_name"] = name
                    tools.append(tool_copy)
        
        return tools
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], mcp_name: Optional[str] = None) -> Dict[str, Any]:
        """Call a tool on the appropriate MCP server"""
        if mcp_name:
            # Call tool on specific MCP
            if mcp_name not in self.connections:
                raise ValueError(f"MCP {mcp_name} not found")
            
            return await self.connections[mcp_name].call_tool(tool_name, arguments)
        else:
            # Find MCP that has this tool
            for connection in self.connections.values():
                if connection.is_connected:
                    for tool in connection.tools:
                        if tool.get("name") == tool_name:
                            return await connection.call_tool(tool_name, arguments)
            
            raise ValueError(f"Tool {tool_name} not found in any connected MCP")
    
    async def cleanup(self):
        """Clean up all MCP connections"""
        logger.info("Cleaning up MCP connections...")
        
        tasks = []
        for connection in self.connections.values():
            tasks.append(connection.stop())
        
        await asyncio.gather(*tasks, return_exceptions=True)
