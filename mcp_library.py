#!/usr/bin/env python3
"""
MCP Library - Core functionality for MCP-Use integration
Provides a clean API for using MCP tools from other applications
"""

import asyncio
import json
import os
from contextlib import redirect_stderr
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

from mcp_use import MCPClient, MCPAgent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


@dataclass
class MCPResponse:
    """Response from MCP operations"""
    success: bool
    content: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class MCPConfig:
    """Configuration for MCP Library"""
    config_path: str = "mcp_config.json"
    openai_model: str = "gpt-4-turbo-preview"
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7
    verbose: bool = False
    preferred_provider: Optional[str] = None  # "openai" or "anthropic"


class MCPLibrary:
    """
    Core MCP Library for programmatic access to MCP tools
    
    This class provides a clean API for integrating MCP functionality
    into other applications, APIs, or services.
    """
    
    def __init__(self, config: Optional[MCPConfig] = None):
        """
        Initialize the MCP Library
        
        Args:
            config: MCPConfig instance with settings
        """
        self.config = config or MCPConfig()
        self.client = None
        self.llm = None
        self._initialized = False
        self._sessions = {}
    
    async def initialize(self) -> MCPResponse:
        """
        Initialize the MCP client and connections
        
        Returns:
            MCPResponse indicating success/failure
        """
        try:
            # Setup LLM
            llm_result = self._setup_llm()
            if not llm_result.success:
                return llm_result
            
            # Create MCP client
            self.client = MCPClient(self.config.config_path)
            
            # Create sessions for all configured servers
            self._sessions = await self.client.create_all_sessions(auto_initialize=True)
            
            self._initialized = True
            
            return MCPResponse(
                success=True,
                content=f"Successfully initialized {len(self._sessions)} MCP servers",
                metadata={
                    "servers": list(self._sessions.keys()),
                    "server_count": len(self._sessions),
                    "llm_provider": self._get_provider_name()
                }
            )
            
        except Exception as e:
            return MCPResponse(
                success=False,
                content="Failed to initialize MCP Library",
                error=str(e)
            )
    
    def _setup_llm(self) -> MCPResponse:
        """Setup LLM based on configuration and available API keys"""
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Determine which provider to use
        if self.config.preferred_provider == "openai" and openai_key:
            provider = "openai"
        elif self.config.preferred_provider == "anthropic" and anthropic_key:
            provider = "anthropic"
        elif openai_key:
            provider = "openai"
        elif anthropic_key:
            provider = "anthropic"
        else:
            return MCPResponse(
                success=False,
                content="No API key found",
                error="Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file"
            )
        
        try:
            if provider == "openai":
                self.llm = ChatOpenAI(
                    model=self.config.openai_model,
                    api_key=openai_key,
                    temperature=self.config.temperature
                )
            else:  # anthropic
                self.llm = ChatAnthropic(
                    model=self.config.anthropic_model,
                    api_key=anthropic_key,
                    temperature=self.config.temperature
                )
            
            return MCPResponse(
                success=True,
                content=f"LLM initialized with {provider}",
                metadata={"provider": provider, "model": self._get_model_name()}
            )
            
        except Exception as e:
            return MCPResponse(
                success=False,
                content=f"Failed to initialize {provider} LLM",
                error=str(e)
            )
    
    def _get_provider_name(self) -> str:
        """Get the current LLM provider name"""
        if isinstance(self.llm, ChatOpenAI):
            return "openai"
        elif isinstance(self.llm, ChatAnthropic):
            return "anthropic"
        return "unknown"
    
    def _get_model_name(self) -> str:
        """Get the current model name"""
        if isinstance(self.llm, ChatOpenAI):
            return self.config.openai_model
        elif isinstance(self.llm, ChatAnthropic):
            return self.config.anthropic_model
        return "unknown"
    
    async def query(self, prompt: str) -> MCPResponse:
        """
        Execute a single query using MCP tools
        
        Args:
            prompt: Natural language prompt/query
            
        Returns:
            MCPResponse with the result
        """
        if not self._initialized:
            init_result = await self.initialize()
            if not init_result.success:
                return init_result
        
        try:
            # Create agent for this query
            agent = MCPAgent(
                client=self.client,
                llm=self.llm,
                verbose=self.config.verbose
            )
            
            # Initialize the agent
            await agent.initialize()
            
            # Process the query using the correct run method
            response = await agent.run(prompt)
            
            return MCPResponse(
                success=True,
                content=str(response),  # Convert response to string
                metadata={
                    "prompt": prompt,
                    "provider": self._get_provider_name(),
                    "model": self._get_model_name()
                }
            )
            
        except Exception as e:
            return MCPResponse(
                success=False,
                content="Failed to process query",
                error=str(e),
                metadata={"prompt": prompt}
            )
    
    async def batch_query(self, prompts: List[str]) -> List[MCPResponse]:
        """
        Execute multiple queries in batch
        
        Args:
            prompts: List of natural language prompts/queries
            
        Returns:
            List of MCPResponse objects
        """
        if not self._initialized:
            init_result = await self.initialize()
            if not init_result.success:
                return [init_result]
        
        results = []
        
        for prompt in prompts:
            if not prompt.strip() or prompt.strip().startswith('#'):
                continue  # Skip empty lines and comments
                
            result = await self.query(prompt.strip())
            results.append(result)
        
        return results
    
    async def get_server_info(self) -> MCPResponse:
        """
        Get information about connected MCP servers
        
        Returns:
            MCPResponse with server information
        """
        if not self._initialized:
            return MCPResponse(
                success=False,
                content="Library not initialized",
                error="Call initialize() first"
            )
        
        try:
            server_info = {
                "servers": list(self._sessions.keys()),
                "server_count": len(self._sessions),
                "llm_provider": self._get_provider_name(),
                "llm_model": self._get_model_name(),
                "config_path": self.config.config_path
            }
            
            return MCPResponse(
                success=True,
                content="Server information retrieved",
                metadata=server_info
            )
            
        except Exception as e:
            return MCPResponse(
                success=False,
                content="Failed to get server info",
                error=str(e)
            )
    
    async def initialize(self) -> MCPResponse:
        """
        Initialize the MCP client and connect to servers
        
        Returns:
            MCPResponse indicating initialization status
        """
        if self._initialized:
            return MCPResponse(
                success=True,
                content="MCP client already initialized"
            )
        
        try:
            # Clean up any orphaned processes from previous runs
            await self._cleanup_orphaned_processes()
            
            # Setup LLM first
            llm_result = self._setup_llm()
            if not llm_result.success:
                return llm_result
            
            # Create MCP client with config path
            config_path = self.config.config_path or "mcp_config.json"
            self.client = MCPClient(config_path)
            
            # Create sessions for all configured servers
            self._sessions = await self.client.create_all_sessions(auto_initialize=True)
            
            self._initialized = True
            
            return MCPResponse(
                success=True,
                content=f"Successfully initialized {len(self._sessions)} MCP servers",
                metadata={
                    "servers": list(self._sessions.keys()),
                    "server_count": len(self._sessions),
                    "llm_provider": self._get_provider_name()
                }
            )
            
        except Exception as e:
            return MCPResponse(
                success=False,
                content="Failed to initialize MCP client",
                error=str(e)
            )
    
    async def _cleanup_orphaned_processes(self):
        """Clean up any orphaned MCP server processes from previous runs"""
        import subprocess
        try:
            # Kill any orphaned MCP server processes
            subprocess.run([
                'pkill', '-f', 
                'mcp-server|notion-mcp|teams-mcp|slack-mcp|supabase.*mcp|browsermcp'
            ], capture_output=True, timeout=2)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass  # Ignore errors in cleanup
    
    async def cleanup(self) -> MCPResponse:
        """
        Clean up resources and close connections properly
        
        Returns:
            MCPResponse indicating cleanup status
        """
        if not self.client:
            return MCPResponse(
                success=True,
                content="No cleanup needed - client not initialized"
            )
        
        try:
            # First, try graceful shutdown with timeout
            import asyncio
            
            # Set a timeout for cleanup to prevent hanging
            try:
                # Attempt graceful cleanup with timeout
                await asyncio.wait_for(
                    self._graceful_cleanup(),
                    timeout=5.0  # 5 second timeout
                )
            except asyncio.TimeoutError:
                # If graceful cleanup times out, force cleanup
                await self._force_cleanup()
            
            self._initialized = False
            self._sessions = {}
            
            return MCPResponse(
                success=True,
                content="Resources cleaned up successfully"
            )
            
        except Exception as e:
            # Last resort: force cleanup
            await self._force_cleanup()
            return MCPResponse(
                success=True,  # Still return success since we cleaned up
                content="Cleanup completed with force termination",
                error=str(e)
            )
    
    async def _graceful_cleanup(self):
        """Attempt graceful cleanup"""
        if not self.config.verbose:
            # Suppress output during cleanup
            import sys
            with open(os.devnull, 'w') as devnull:
                old_stderr = sys.stderr
                old_stdout = sys.stdout
                sys.stderr = devnull
                sys.stdout = devnull
                try:
                    await self.client.close_all_sessions()
                finally:
                    sys.stderr = old_stderr
                    sys.stdout = old_stdout
        else:
            await self.client.close_all_sessions()
    
    async def _force_cleanup(self):
        """Force cleanup by terminating MCP server processes"""
        import subprocess
        import os
        
        try:
            # Kill any remaining MCP server processes
            subprocess.run([
                'pkill', '-f', 
                'mcp-server|notion-mcp|teams-mcp|slack-mcp|supabase.*mcp|browsermcp'
            ], capture_output=True, timeout=2)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass  # Ignore errors in force cleanup


# Convenience functions for simple usage
async def simple_query(prompt: str, config: Optional[MCPConfig] = None) -> MCPResponse:
    """
    Simple one-shot query function
    
    Args:
        prompt: Natural language query
        config: Optional configuration
        
    Returns:
        MCPResponse with result
    """
    library = MCPLibrary(config)
    try:
        result = await library.query(prompt)
        return result
    finally:
        await library.cleanup()


async def simple_batch_query(prompts: List[str], config: Optional[MCPConfig] = None) -> List[MCPResponse]:
    """
    Simple batch query function
    
    Args:
        prompts: List of natural language queries
        config: Optional configuration
        
    Returns:
        List of MCPResponse objects
    """
    library = MCPLibrary(config)
    try:
        results = await library.batch_query(prompts)
        return results
    finally:
        await library.cleanup()
