#!/usr/bin/env python3
"""
MCP Library - Core functionality for MCP-Use integration
Provides a clean API for using MCP tools from other applications
"""

import asyncio
import json
import os
import time
import random
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

# Rate limiting and retry configuration
MAX_RETRIES = 5
BASE_DELAY = 2.0  # Base delay in seconds
MAX_DELAY = 60.0  # Maximum delay cap
JITTER_RANGE = 0.1  # Random jitter range

# Rate limit error codes and messages
RATE_LIMIT_ERRORS = [
    "rate_limit_exceeded",
    "too_many_requests", 
    "quota_exceeded",
    "429",
    "rate limit",
    "throttled"
]

# Tool validation error patterns
TOOL_VALIDATION_ERRORS = [
    "validation error",
    "pydantic",
    "input should be",
    "field required",
    "type=list_type",
    "type=dict_type"
]

def is_rate_limit_error(error_msg: str) -> bool:
    """Check if an error is related to rate limiting"""
    error_lower = str(error_msg).lower()
    return any(rate_error in error_lower for rate_error in RATE_LIMIT_ERRORS)

def is_tool_validation_error(error_msg: str) -> bool:
    """Check if an error is related to tool validation/schema issues"""
    error_lower = str(error_msg).lower()
    return any(validation_error in error_lower for validation_error in TOOL_VALIDATION_ERRORS)

def calculate_backoff_delay(attempt: int, base_delay: float = BASE_DELAY) -> float:
    """Calculate exponential backoff delay with jitter"""
    # Exponential backoff: base_delay * (2 ^ attempt)
    delay = base_delay * (2 ** attempt)
    
    # Add random jitter to prevent thundering herd
    jitter = random.uniform(-JITTER_RANGE, JITTER_RANGE) * delay
    delay += jitter
    
    # Cap the maximum delay
    return min(delay, MAX_DELAY)

def retry_on_rate_limit(max_retries: int = MAX_RETRIES):
    """Decorator for retrying functions on rate limit errors"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    
                    # Check if this is a rate limit error
                    if not is_rate_limit_error(error_msg):
                        # Not a rate limit error, re-raise immediately
                        raise e
                    
                    # If this was the last attempt, re-raise
                    if attempt >= max_retries:
                        raise e
                    
                    # Calculate backoff delay
                    delay = calculate_backoff_delay(attempt)
                    
                    # Log the retry attempt
                    print(f"⚠️  Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay:.1f}s...")
                    
                    # Wait before retrying
                    await asyncio.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_error
        
        return wrapper
    return decorator


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
    
    # Rate limiting configuration
    max_retries: int = 5
    base_delay: float = 2.0  # Base delay in seconds for exponential backoff
    max_delay: float = 60.0  # Maximum delay cap
    batch_delay_increment: float = 2.0  # How much to increase delay after rate limits in batch
    max_batch_delay: float = 10.0  # Maximum delay between batch queries


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
                    temperature=self.config.temperature,
                    max_retries=self.config.max_retries,
                    timeout=120,  # 2 minute timeout
                    request_timeout=120,
                    # Configure retry behavior for rate limits
                    default_headers={
                        "User-Agent": "MCP-CLI/1.0"
                    },
                    # Add model kwargs for better tool calling
                    model_kwargs={
                        "tool_choice": "auto",
                        "parallel_tool_calls": False  # Disable parallel calls to reduce errors
                    }
                )
            else:  # anthropic
                self.llm = ChatAnthropic(
                    model=self.config.anthropic_model,
                    api_key=anthropic_key,
                    temperature=self.config.temperature,
                    max_retries=self.config.max_retries,
                    timeout=120,  # 2 minute timeout
                    # Configure retry behavior for rate limits
                    default_headers={
                        "User-Agent": "MCP-CLI/1.0"
                    }
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
        Execute a single query using MCP tools with rate limit handling
        
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
            # Execute query with retry logic for rate limits
            response = await self._execute_query_with_retry(prompt)
            
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
            error_msg = str(e)
            
            # Check if this is a rate limit error for user feedback
            if is_rate_limit_error(error_msg):
                return MCPResponse(
                    success=False,
                    content="Rate limit exceeded after all retry attempts",
                    error=f"Rate limit error: {error_msg}",
                    metadata={"prompt": prompt, "error_type": "rate_limit"}
                )
            # Check if this is a tool validation error
            elif is_tool_validation_error(error_msg):
                return MCPResponse(
                    success=False,
                    content="Tool validation error - there may be an issue with the MCP tool schema or AI's tool usage",
                    error=f"Validation error: {self._extract_validation_error_details(error_msg)}",
                    metadata={"prompt": prompt, "error_type": "tool_validation"}
                )
            else:
                return MCPResponse(
                    success=False,
                    content="Failed to process query",
                    error=error_msg,
                    metadata={"prompt": prompt}
                )
    
    async def _execute_query_with_retry(self, prompt: str) -> str:
        """
        Execute a single query with automatic retry on rate limits
        
        Args:
            prompt: Natural language prompt/query
            
        Returns:
            Response string from the agent
        """
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Create agent for this query
                agent = MCPAgent(
                    client=self.client,
                    llm=self.llm,
                    verbose=self.config.verbose
                )
                
                # Initialize the agent
                await agent.initialize()
                
                # Process the query using the correct run method with error handling
                response = await self._safe_agent_run(agent, prompt)
                
                return response
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                # Check if this is a rate limit error (including OpenAI specific errors)
                if not (is_rate_limit_error(error_msg) or self._is_openai_rate_limit(e)):
                    # Check if this is a tool validation error
                    if is_tool_validation_error(error_msg):
                        # Tool validation errors should not be retried, but we can provide better feedback
                        raise Exception(f"Tool validation error: {self._extract_validation_error_details(error_msg)}")
                    # Not a rate limit error, re-raise immediately
                    raise e
                
                # If this was the last attempt, re-raise
                if attempt >= self.config.max_retries:
                    raise e
                
                # Calculate backoff delay using config parameters
                delay = self.config.base_delay * (2 ** attempt)
                jitter = random.uniform(-JITTER_RANGE, JITTER_RANGE) * delay
                delay = min(delay + jitter, self.config.max_delay)
                
                # Extract retry-after header if available
                retry_after = self._extract_retry_after(e)
                if retry_after and retry_after > delay:
                    delay = min(retry_after, self.config.max_delay)
                
                # Log the retry attempt
                if self.config.verbose or attempt > 0:  # Always show after first retry
                    print(f"⚠️  Rate limit hit (attempt {attempt + 1}/{self.config.max_retries + 1}). Retrying in {delay:.1f}s...")
                    if self.config.verbose:
                        print(f"    Error: {error_msg[:100]}...")
                
                # Wait before retrying
                await asyncio.sleep(delay)
        
        # This should never be reached, but just in case
        raise last_error
    
    def _is_openai_rate_limit(self, error: Exception) -> bool:
        """Check if error is specifically an OpenAI rate limit error"""
        # Check for OpenAI RateLimitError
        error_type = type(error).__name__
        if error_type == 'RateLimitError':
            return True
        
        # Check for rate limit in error attributes
        if hasattr(error, 'status_code') and error.status_code == 429:
            return True
            
        # Check for rate limit in error response
        if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
            if error.response.status_code == 429:
                return True
        
        return False
    
    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract retry-after header from rate limit error"""
        try:
            # Try to extract from OpenAI error message
            error_msg = str(error)
            if "try again in" in error_msg.lower():
                # Extract time from "Please try again in 902ms"
                import re
                match = re.search(r'try again in (\d+)ms', error_msg)
                if match:
                    return float(match.group(1)) / 1000.0  # Convert ms to seconds
                
                match = re.search(r'try again in (\d+)s', error_msg)
                if match:
                    return float(match.group(1))
            
            # Try to extract from response headers
            if hasattr(error, 'response') and hasattr(error.response, 'headers'):
                retry_after = error.response.headers.get('retry-after')
                if retry_after:
                    return float(retry_after)
        except (ValueError, AttributeError):
            pass
        
        return None
    
    def _extract_validation_error_details(self, error_msg: str) -> str:
        """Extract useful details from validation error messages"""
        try:
            # Extract field name and expected type from Pydantic errors
            if "Input should be" in error_msg:
                lines = error_msg.split('\n')
                for line in lines:
                    if "Input should be" in line:
                        return line.strip()
            
            # Extract field path from validation errors
            if "validation error" in error_msg.lower():
                lines = error_msg.split('\n')
                for line in lines:
                    if any(field in line for field in ['properties.', 'field required', 'type=']):
                        return line.strip()
            
            # Return first meaningful line if no specific pattern found
            lines = [line.strip() for line in error_msg.split('\n') if line.strip()]
            return lines[0] if lines else error_msg
            
        except Exception:
            return error_msg
    
    async def _safe_agent_run(self, agent, prompt: str) -> str:
        """Safely run agent with additional error handling"""
        try:
            response = await agent.run(prompt)
            return response
        except Exception as e:
            # Re-raise with additional context for better error handling
            error_msg = str(e)
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                # This is definitely a rate limit error, preserve it
                raise e
            else:
                # Other error, re-raise as-is
                raise e
    
    async def batch_query(self, prompts: List[str]) -> List[MCPResponse]:
        """
        Execute multiple queries in batch with rate limit handling
        
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
        rate_limit_delay = 0.0  # Progressive delay for batch processing
        
        for i, prompt in enumerate(prompts):
            if not prompt.strip() or prompt.strip().startswith('#'):
                continue  # Skip empty lines and comments
            
            # Add progressive delay to prevent hitting rate limits in batch
            if rate_limit_delay > 0:
                if self.config.verbose:
                    print(f"⏱️  Waiting {rate_limit_delay:.1f}s before next query to prevent rate limits...")
                await asyncio.sleep(rate_limit_delay)
            
            result = await self.query(prompt.strip())
            results.append(result)
            
            # If we hit a rate limit, increase the delay for subsequent queries
            if result.metadata and result.metadata.get('error_type') == 'rate_limit':
                rate_limit_delay = min(rate_limit_delay + self.config.batch_delay_increment, self.config.max_batch_delay)
                if self.config.verbose:
                    print(f"📈 Increased batch delay to {rate_limit_delay:.1f}s due to rate limits")
            elif rate_limit_delay > 0:
                # Gradually reduce delay if no rate limits
                rate_limit_delay = max(rate_limit_delay - 0.5, 0.0)
        
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
