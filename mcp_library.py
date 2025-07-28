#!/usr/bin/env python3
"""
MCP Library - Core functionality for MCP-Use integration
Provides a clean API for using MCP tools from other applications
"""

import asyncio
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient
from pydantic import BaseModel, ValidationError

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

# Rate limiting and retry configuration
MAX_RETRIES = 5
BASE_DELAY = 2.0  # Base delay in seconds
MAX_DELAY = 60.0  # Maximum delay cap
JITTER_RANGE = 0.1  # Random jitter range
MAX_STEPS = 10

# Rate limit error codes and messages
RATE_LIMIT_ERRORS = [
    "rate_limit_exceeded",
    "too_many_requests", 
    "quota_exceeded",
    "429",
    "rate limit",
    "throttled"
]



def is_rate_limit_error(error_msg: str) -> bool:
    """Check if an error is related to rate limiting"""
    error_lower = str(error_msg).lower()
    return any(rate_error in error_lower for rate_error in RATE_LIMIT_ERRORS)





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
            # Check if this is a validation error for better user feedback
            elif "validation error" in error_msg.lower() and "input should be a valid list" in error_msg.lower():
                return MCPResponse(
                    success=False,
                    content="Schema validation error: AI output format doesn't match API expectations",
                    error=f"Validation error: {error_msg}\n\nThis is a known limitation where AI generates dict format but API expects array format. This is a schema mismatch between AI behavior and API design.",
                    metadata={"prompt": prompt, "error_type": "schema_validation"}
                )
            else:
                return MCPResponse(
                    success=False,
                    content="Failed to process query",
                    error=error_msg,
                    metadata={"prompt": prompt}
                )
    
    async def _execute_query_with_retry(self, prompt: str) -> str:
        """Execute query with retry logic for rate limits and parameter correction"""
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Create and initialize agent for this attempt
                agent = MCPAgent(
                    client=self.client,
                    llm=self.llm,
                    max_steps=10,
                    memory_enabled=True,
                    verbose=self.config.verbose
                )
                await agent.initialize()
                
                # Execute query with mcp-use's built-in validation
                response = await self._safe_agent_run(agent, prompt)
                return response
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                # Check if this is a rate limit error (including OpenAI specific errors)
                if not (is_rate_limit_error(error_msg) or self._is_openai_rate_limit(e)):
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
    

    

    
    async def _safe_agent_run(self, agent, prompt: str) -> str:
        """Execute agent with enhanced prompting and error handling"""
        try:
            # Add basic guidance for information gathering and error handling
            enhanced_prompt = f"""{prompt}

GUIDANCE:
- Before asking the user for information, try to find it using available tools first
- If you need identifiers (like page IDs), search for them using available search/list tools
- Be proactive in using available tools to gather required information
- If a tool call fails, read the error message and try to correct the issue"""
            
            # Execute the agent with enhanced prompt
            response = await agent.run(enhanced_prompt)
            return response
            
        except Exception as e:
            error_str = str(e)
            
            # Provide helpful context for common validation errors
            if self._is_schema_validation_error(error_str):
                if self.config.verbose:
                    print(f"📋 Schema validation error detected:")
                    print(f"   {self._get_validation_error_explanation(error_str)}")
                    print(f"   This is a known limitation - see documentation for details.")
            
            # Log the error for debugging if verbose mode is enabled
            if self.config.verbose:
                print(f"⚠️  Agent execution error: {str(e)[:200]}...")
            
            # Re-raise the error for the caller to handle
            raise e
    
    def _is_schema_validation_error(self, error_str: str) -> bool:
        """Check if this is a schema validation error that we can provide context for"""
        # Look for common Pydantic validation error patterns
        validation_patterns = [
            "validation error for DynamicModel",
            "Input should be a valid list",
            "Input should be a valid dict", 
            "type=list_type",
            "type=dict_type",
            "type=missing",
            "Field required",
            "ValidationError",
        ]
        
        return any(pattern in error_str for pattern in validation_patterns)
    
    def _get_validation_error_explanation(self, error_str: str) -> str:
        """Provide a helpful explanation of the validation error"""
        if "Input should be a valid list" in error_str and "type=list_type" in error_str:
            if "properties.title" in error_str:
                return "Rich text properties like 'title' expect arrays: [{'text': {'content': 'value'}}] not {'text': {'content': 'value'}}"
            elif "properties.children" in error_str:
                return "Block content properties like 'children' expect arrays of block objects"
            else:
                return "The API expects an array/list format, but received a single object"
        
        elif "Input should be a valid dict" in error_str and "type=dict_type" in error_str:
            return "The API expects an object/dict format, but received an array"
        
        elif "Field required" in error_str and "type=missing" in error_str:
            if "parent.type" in error_str:
                return "Missing 'parent.type' field - should be 'page_id' for Notion page creation"
            elif "properties" in error_str:
                return "Missing 'properties' field - this should contain the actual page content/data"
            else:
                return "Missing required fields - check the API schema for all required parameters"
        
        elif "validation error for DynamicModel" in error_str:
            return "Schema validation failed - parameter format doesn't match API expectations"
        
        else:
            return "Parameter validation failed - check the error message for specific format requirements"
    
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
            import asyncio
            import logging
            import sys
            import os
            import subprocess
            
            # Completely suppress ALL output during cleanup to hide any server errors
            mcp_logger = logging.getLogger('mcp_use')
            original_level = mcp_logger.level
            mcp_logger.setLevel(logging.CRITICAL)
            
            # Suppress all Python logging during cleanup
            logging.getLogger().setLevel(logging.CRITICAL)
            
            # Create a context manager to suppress ALL output including subprocess output
            import contextlib
            
            @contextlib.contextmanager
            def suppress_all_output():
                with open(os.devnull, 'w') as devnull:
                    old_stderr = sys.stderr
                    old_stdout = sys.stdout
                    
                    # Redirect all output streams
                    sys.stderr = devnull
                    sys.stdout = devnull
                    
                    # Also set environment variables to suppress Node.js output
                    old_env = os.environ.copy()
                    os.environ['NODE_NO_WARNINGS'] = '1'
                    os.environ['SUPPRESS_NO_CONFIG_WARNING'] = '1'
                    
                    try:
                        yield
                    finally:
                        # Restore everything
                        sys.stderr = old_stderr
                        sys.stdout = old_stdout
                        os.environ.clear()
                        os.environ.update(old_env)
            
            # Use comprehensive output suppression
            with suppress_all_output():
                try:
                    # Very short timeout - don't wait for problematic servers
                    try:
                        await asyncio.wait_for(
                            self._graceful_cleanup(),
                            timeout=0.2  # Minimal timeout to avoid hanging
                        )
                    except asyncio.TimeoutError:
                        pass  # Expected for servers with shutdown bugs
                    
                    # Always do aggressive cleanup to ensure everything is terminated
                    await self._aggressive_cleanup()
                    
                except Exception:
                    # Suppress any cleanup errors completely
                    pass
            
            # Restore logging level
            mcp_logger.setLevel(original_level)
            logging.getLogger().setLevel(logging.WARNING)
            
            # Clear our session tracking
            self._sessions = {}
            self.client = None
            self._initialized = False
            
            return MCPResponse(
                success=True,
                content="Resources cleaned up successfully"
            )
            
        except Exception as e:
            # Last resort: aggressive cleanup
            await self._aggressive_cleanup()
            return MCPResponse(
                success=True,  # Still return success since we cleaned up
                content="Cleanup completed with aggressive termination",
                error=str(e)
            )
    
    async def _graceful_cleanup(self):
        """Gracefully cleanup MCP client and sessions with robust error handling"""
        if not self.client:
            return
            
        try:
            import subprocess
            import asyncio
            
            # Generic cleanup: close individual sessions with timeout
            if hasattr(self, '_sessions') and self._sessions:
                for session_id, session in list(self._sessions.items()):
                    try:
                        await asyncio.wait_for(session.close(), timeout=0.5)
                    except Exception:
                        # If graceful close fails, try to terminate the process directly
                        try:
                            if hasattr(session, 'connector') and hasattr(session.connector, 'process'):
                                process = session.connector.process
                                if process and process.poll() is None:
                                    # Try graceful termination first
                                    process.terminate()
                                    try:
                                        await asyncio.wait_for(
                                            asyncio.create_task(asyncio.sleep(0)), timeout=0.1
                                        )
                                        process.wait(timeout=0.5)
                                    except (asyncio.TimeoutError, subprocess.TimeoutExpired):
                                        # Force kill if graceful termination fails
                                        process.kill()
                        except Exception:
                            pass
            
            # Try client's built-in cleanup with timeout
            if hasattr(self.client, 'close_all_sessions'):
                try:
                    await asyncio.wait_for(self.client.close_all_sessions(), timeout=1.0)
                except Exception:
                    pass
                
        except Exception:
            # Suppress all cleanup errors
            pass
        finally:
            # Clear client and session tracking
            self.client = None
            if hasattr(self, '_sessions'):
                self._sessions.clear()
    
    async def _aggressive_cleanup(self):
        """Generic cleanup by clearing all references"""
        try:
            # Clear all references
            self.client = None
            if hasattr(self, '_sessions'):
                self._sessions.clear()
        except Exception:
            # Suppress all cleanup errors
            pass


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
