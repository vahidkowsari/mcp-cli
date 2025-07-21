"""
AI Client module for handling OpenAI and Claude API interactions
"""

import asyncio
import logging
import time
import random
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)

# ===== CONFIGURATION CONSTANTS =====
# 
# 🛠️  EASY CONFIGURATION GUIDE:
# 
# To adjust retry behavior:
# - DEFAULT_MAX_RETRIES: How many times to retry failed requests (default: 5)
# - DEFAULT_BASE_DELAY: Initial delay between retries in seconds (default: 3.0)
# - DEFAULT_MAX_DELAY: Maximum delay cap in seconds (default: 60.0)
# 
# To adjust rate limiting:
# - For OpenAI: Modify OPENAI_REQUESTS_PER_MINUTE and OPENAI_TOKENS_PER_MINUTE
# - For Claude: Modify CLAUDE_REQUESTS_PER_MINUTE and CLAUDE_TOKENS_PER_MINUTE
# 
# For 529 errors: Use CLAUDE_CONSERVATIVE_* settings below
# For faster processing: Use CLAUDE_AGGRESSIVE_* settings below
# For normal usage: Use CLAUDE_REQUESTS_PER_MINUTE and CLAUDE_TOKENS_PER_MINUTE
# 

# Retry Configuration
DEFAULT_MAX_RETRIES = 5        # Number of retry attempts
DEFAULT_BASE_DELAY = 3.0       # Base delay in seconds (exponential backoff)
DEFAULT_MAX_DELAY = 60.0       # Maximum delay cap in seconds
DEFAULT_JITTER_RANGE = 0.1     # Jitter range (±10% of delay)

# Rate Limiting Configuration
# OpenAI Rate Limits (adjust based on your tier: Free=3/min, Plus=50/min, Pro=5000/min)
OPENAI_REQUESTS_PER_MINUTE = 30
OPENAI_TOKENS_PER_MINUTE = 80000

# Claude Rate Limits (based on official Anthropic limits)
CLAUDE_REQUESTS_PER_MINUTE = 45    # Official limit: 50/min (using 90% for safety)
CLAUDE_TOKENS_PER_MINUTE = 45000   # Official limit: 40K-50K/min (using conservative estimate)

# Alternative Claude Settings for Different Scenarios:
# 🐌 Conservative (for heavy 529 error periods):
CLAUDE_CONSERVATIVE_REQUESTS = 15
CLAUDE_CONSERVATIVE_TOKENS = 20000

# 🚀 Aggressive (maximum throughput, risk more 529s):
CLAUDE_AGGRESSIVE_REQUESTS = 50    # Use full official limit
CLAUDE_AGGRESSIVE_TOKENS = 50000   # Use full official limit

# Retryable HTTP status codes
RETRYABLE_STATUS_CODES = [429, 502, 503, 504, 529]


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, requests_per_minute: int = 50, tokens_per_minute: int = 100000):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        
        # Request rate limiting
        self.request_tokens = requests_per_minute
        self.request_last_refill = time.time()
        
        # Token rate limiting  
        self.token_tokens = tokens_per_minute
        self.token_last_refill = time.time()
        
        self._lock = asyncio.Lock()
    
    async def wait_for_capacity(self, estimated_tokens: int = 1000):
        """Wait until we have capacity for a request with estimated tokens."""
        async with self._lock:
            now = time.time()
            
            # Refill request tokens (1 per minute / requests_per_minute)
            time_passed = now - self.request_last_refill
            self.request_tokens = min(
                self.requests_per_minute,
                self.request_tokens + (time_passed * self.requests_per_minute / 60.0)
            )
            self.request_last_refill = now
            
            # Refill token bucket
            self.token_tokens = min(
                self.tokens_per_minute,
                self.token_tokens + (time_passed * self.tokens_per_minute / 60.0)
            )
            self.token_last_refill = now
            
            # Check if we need to wait for request capacity
            if self.request_tokens < 1:
                wait_time = (1 - self.request_tokens) * 60.0 / self.requests_per_minute
                logger.info(f"Rate limit: waiting {wait_time:.1f}s for request capacity")
                await asyncio.sleep(wait_time)
                self.request_tokens = 1
            
            # Check if we need to wait for token capacity
            if self.token_tokens < estimated_tokens:
                wait_time = (estimated_tokens - self.token_tokens) * 60.0 / self.tokens_per_minute
                logger.info(f"Rate limit: waiting {wait_time:.1f}s for token capacity ({estimated_tokens} tokens)")
                await asyncio.sleep(wait_time)
                self.token_tokens = estimated_tokens
            
            # Consume tokens
            self.request_tokens -= 1
            self.token_tokens -= estimated_tokens


def retry_on_api_error(max_retries: int = DEFAULT_MAX_RETRIES, base_delay: float = DEFAULT_BASE_DELAY, max_delay: float = DEFAULT_MAX_DELAY):
    """Decorator to retry API calls on retryable errors with exponential backoff."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if error is retryable by status code
                    error_str = str(e)
                    is_retryable = any(str(code) in error_str for code in RETRYABLE_STATUS_CODES)
                    
                    # Also check for common retryable error keywords
                    error_str_lower = error_str.lower()
                    is_retryable = is_retryable or any(keyword in error_str_lower for keyword in [
                        "rate_limit", "overloaded", "timeout", "connection", "network"
                    ])
                    
                    if not is_retryable or attempt == max_retries:
                        # Not retryable or max retries reached
                        raise e
                    
                    # Calculate delay with exponential backoff, jitter, and max cap
                    exponential_delay = base_delay * (2 ** attempt)
                    jitter = random.uniform(-DEFAULT_JITTER_RANGE, DEFAULT_JITTER_RANGE) * exponential_delay
                    delay = min(exponential_delay + jitter, max_delay)
                    
                    logger.warning(f"API error (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        return wrapper
    return decorator


class BaseAIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    async def chat_completion(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generate chat completion with optional tool calls"""
        pass


class OpenAIProvider(BaseAIProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", 
                 requests_per_minute: int = OPENAI_REQUESTS_PER_MINUTE,
                 tokens_per_minute: int = OPENAI_TOKENS_PER_MINUTE):
        self.api_key = api_key
        self.model = model
        self.client = None
        # Rate limits for OpenAI (configurable based on your tier)
        self.rate_limiter = RateLimiter(requests_per_minute=requests_per_minute, tokens_per_minute=tokens_per_minute)
        self._initialize_client()

    def _initialize_client(self):
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")

    
    @retry_on_api_error()  # Uses DEFAULT_MAX_RETRIES and DEFAULT_BASE_DELAY
    async def chat_completion(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generate chat completion using OpenAI API with retry logic"""
        # Estimate tokens (rough approximation: 4 chars = 1 token)
        estimated_tokens = sum(len(str(msg.get('content', ''))) for msg in messages) // 4 + 1000
        await self.rate_limiter.wait_for_capacity(estimated_tokens)
        
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = await self.client.chat.completions.create(**kwargs)
            
            return {
                "content": response.choices[0].message.content,
                "tool_calls": getattr(response.choices[0].message, 'tool_calls', None),
                "usage": response.usage.model_dump() if response.usage else None
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class ClaudeProvider(BaseAIProvider):
    """Claude (Anthropic) API provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022",
                 requests_per_minute: int = CLAUDE_REQUESTS_PER_MINUTE,
                 tokens_per_minute: int = CLAUDE_TOKENS_PER_MINUTE):
        self.api_key = api_key
        self.model = model
        self.client = None
        # Rate limits for Claude (configurable, conservative during high load)
        self.rate_limiter = RateLimiter(requests_per_minute=requests_per_minute, tokens_per_minute=tokens_per_minute)
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
    
    @retry_on_api_error()  # Uses DEFAULT_MAX_RETRIES and DEFAULT_BASE_DELAY
    async def chat_completion(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generate chat completion using Claude API with retry logic"""
        # Estimate tokens (rough approximation: 4 chars = 1 token)
        estimated_tokens = sum(len(str(msg.get('content', ''))) for msg in messages) // 4 + 1000
        await self.rate_limiter.wait_for_capacity(estimated_tokens)
        
        try:
            # Convert messages format for Claude
            system_message = None
            claude_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    # Handle both string content and structured content for Claude
                    if isinstance(msg["content"], str):
                        claude_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    else:
                        # Already in Claude format (tool results, etc.)
                        claude_messages.append(msg)
            
            kwargs = {
                "model": self.model,
                "messages": claude_messages,
                "max_tokens": 2000,
                "temperature": 0.7
            }
            
            if system_message:
                kwargs["system"] = system_message
            
            if tools:
                kwargs["tools"] = tools
            
            response = await self.client.messages.create(**kwargs)
            
            # Extract tool calls from Claude response
            tool_calls = []
            content = ""
            
            for content_block in response.content:
                if content_block.type == "text":
                    content += content_block.text
                elif content_block.type == "tool_use":
                    tool_calls.append({
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input
                    })
            
            return {
                "content": content,
                "tool_calls": tool_calls if tool_calls else None,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                } if response.usage else None
            }
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise


# ===== HELPER FUNCTIONS =====

def get_claude_limits(scenario: str = "normal") -> tuple[int, int]:
    """Get Claude rate limits for different scenarios.
    
    Args:
        scenario: 'conservative', 'normal', or 'aggressive'
        
    Returns:
        tuple: (requests_per_minute, tokens_per_minute)
    """
    scenarios = {
        "conservative": (CLAUDE_CONSERVATIVE_REQUESTS, CLAUDE_CONSERVATIVE_TOKENS),
        "normal": (CLAUDE_REQUESTS_PER_MINUTE, CLAUDE_TOKENS_PER_MINUTE),
        "aggressive": (CLAUDE_AGGRESSIVE_REQUESTS, CLAUDE_AGGRESSIVE_TOKENS)
    }
    return scenarios.get(scenario, scenarios["normal"])


def create_claude_provider(api_key: str, model: str = "claude-3-5-sonnet-20241022", scenario: str = "normal") -> 'ClaudeProvider':
    """Create Claude provider with scenario-based rate limits.
    
    Args:
        api_key: Anthropic API key
        model: Claude model to use
        scenario: 'conservative' (for 529 errors), 'normal', or 'aggressive' (max speed)
        
    Returns:
        ClaudeProvider instance with appropriate rate limits
    """
    requests_per_min, tokens_per_min = get_claude_limits(scenario)
    return ClaudeProvider(api_key, model, requests_per_min, tokens_per_min)


class AIClient:
    """Main AI client that manages different providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = self._create_provider()
        self.conversation_history: List[Dict[str, str]] = []
    
    def _create_provider(self) -> BaseAIProvider:
        """Create appropriate AI provider based on config"""
        provider_name = self.config.get("provider", "openai").lower()
        
        if provider_name == "openai":
            api_key = os.getenv(self.config.get("api_key_env", "OPENAI_API_KEY"))
            if not api_key:
                raise ValueError(f"API key not found in environment variable {self.config.get('api_key_env', 'OPENAI_API_KEY')}")
            return OpenAIProvider(api_key, self.config.get("model", "gpt-4"))
        elif provider_name == "claude":
            api_key = os.getenv(self.config.get("api_key_env", "ANTHROPIC_API_KEY"))
            if not api_key:
                raise ValueError(f"API key not found in environment variable {self.config.get('api_key_env', 'ANTHROPIC_API_KEY')}")
            return ClaudeProvider(api_key, self.config.get("model", "claude-3-5-sonnet-20241022"))
        else:
            raise ValueError(f"Unsupported AI provider: {provider_name}")
    
    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
    
    async def chat(self, user_message: str, available_tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Send chat message and get response"""
        # Add user message to history (only if not empty)
        if user_message.strip():
            self.add_message("user", user_message)
        
        # Prepare messages with system prompt
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant with access to various tools through MCP (Model Context Protocol). Use the available tools when appropriate to help the user."
            }
        ] + self.conversation_history
        
        # Get response from AI provider
        response = await self.provider.chat_completion(messages, available_tools)
        
        # Only add assistant response to history if no tool calls (tool calls are handled separately)
        if response["content"] and not response.get("tool_calls"):
            self.add_message("assistant", response["content"])
        
        return response
