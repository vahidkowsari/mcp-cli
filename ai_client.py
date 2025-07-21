"""
AI Client using LiteLLM + Tenacity
Handles multiple AI providers with robust retry and rate limiting
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Required libraries
import litellm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)

# Configure LiteLLM
litellm.set_verbose = False  # Reduce noise in logs
litellm.drop_params = True   # Handle provider differences automatically

# ===== CONFIGURATION =====
# 
# 🚀 LiteLLM + Tenacity handles rate limiting and retries automatically!
# 
# Retry Configuration (handled by Tenacity)
MAX_RETRIES = 5              # Maximum retry attempts
INITIAL_WAIT = 1             # Initial wait time in seconds
MAX_WAIT = 60                # Maximum wait time in seconds
JITTER = 5                   # Jitter range for randomization

# Model Configuration
DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_OPENAI_MODEL = "gpt-4"
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.7

# Exception types to retry (LiteLLM handles these automatically)
RETRYABLE_EXCEPTIONS = (
    Exception,  # LiteLLM wraps all provider-specific exceptions
)


def sanitize_messages(messages: List[Dict]) -> List[Dict]:
    """Sanitize messages to ensure compatibility with LiteLLM"""
    sanitized = []
    
    for message in messages:
        # Handle Claude-specific message formats
        if isinstance(message.get("content"), list):
            # Convert Claude content blocks to simple text
            text_content = ""
            for block in message["content"]:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_content += block.get("text", "")
                    elif block.get("type") == "tool_result":
                        # Convert tool results to simple text format
                        result_content = block.get("content", "")
                        text_content += f"Tool result: {result_content}"
                else:
                    # Handle string content blocks
                    text_content += str(block)
            
            sanitized.append({
                "role": message["role"],
                "content": text_content.strip() or "[No content]"
            })
        else:
            # Already in simple format
            sanitized.append(message)
    
    return sanitized


# Retry decorator using Tenacity
@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential_jitter(initial=INITIAL_WAIT, max=MAX_WAIT, jitter=JITTER),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
async def robust_ai_call(messages: List[Dict], tools: Optional[List[Dict]] = None, model: str = DEFAULT_CLAUDE_MODEL) -> Dict[str, Any]:
    """AI call with automatic retry and rate limiting via LiteLLM"""
    try:
        # Prepare the call parameters
        call_params = {
            "model": model,
            "messages": messages,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        
        # Only add tools if they exist and are not empty
        if tools and len(tools) > 0:
            call_params["tools"] = tools
        
        # LiteLLM handles all the complexity for us!
        response = await litellm.acompletion(**call_params)
        
        # Convert LiteLLM response to our expected format
        tool_calls = getattr(response.choices[0].message, 'tool_calls', None)
        converted_tool_calls = None
        
        if tool_calls:
            # Convert OpenAI-style tool calls to our expected format
            converted_tool_calls = []
            for tool_call in tool_calls:
                if hasattr(tool_call, 'function'):
                    # OpenAI format - convert to our format
                    converted_tool_calls.append({
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
                    })
                else:
                    # Already in our expected format
                    converted_tool_calls.append(tool_call)
        
        return {
            "content": response.choices[0].message.content or "",
            "tool_calls": converted_tool_calls,
            "usage": {
                "input_tokens": getattr(response.usage, 'prompt_tokens', 0),
                "output_tokens": getattr(response.usage, 'completion_tokens', 0)
            } if response.usage else None
        }
        
    except Exception as e:
        logger.error(f"AI API call failed: {e}")
        raise


class AIClient:
    """AI client using LiteLLM + Tenacity"""
    
    def __init__(self, ai_config: Dict[str, Any]):
        # Handle both full config and just ai section
        if "ai" in ai_config:
            # Full config passed
            self.config = ai_config
            ai_settings = ai_config["ai"]
        else:
            # Just ai section passed (from main.py)
            self.config = {"ai": ai_config}
            ai_settings = ai_config
        
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Set up API keys from environment
        load_dotenv()
        
        # Configure LiteLLM with API keys
        provider = ai_settings.get("provider", "claude")
        if provider == "claude":
            api_key = os.getenv(ai_settings.get("api_key_env", "ANTHROPIC_API_KEY"))
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
            self.model = ai_settings.get("model", DEFAULT_CLAUDE_MODEL)
        elif provider == "openai":
            api_key = os.getenv(ai_settings.get("api_key_env", "OPENAI_API_KEY"))
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            self.model = ai_settings.get("model", DEFAULT_OPENAI_MODEL)
        else:
            raise ValueError(f"Unsupported AI provider: {provider}")
        
        logger.info(f"AI client initialized with {provider} ({self.model})")
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    async def chat(self, user_message: str, available_tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Send chat message and get response using robust retry logic"""
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
        
        # Sanitize messages to ensure LiteLLM compatibility
        sanitized_messages = sanitize_messages(messages)
        
        # Use our robust AI call with automatic retries and rate limiting
        response = await robust_ai_call(sanitized_messages, available_tools, self.model)
        
        # Only add assistant response to history if no tool calls (tool calls are handled separately)
        if response["content"] and not response.get("tool_calls"):
            self.add_message("assistant", response["content"])
        
        return response


# Helper functions for easy configuration switching
def get_claude_model(performance_mode: str = "balanced") -> str:
    """Get Claude model based on performance requirements"""
    models = {
        "fast": "claude-3-5-haiku-20241022",      # Fastest, cheapest
        "balanced": "claude-3-5-sonnet-20241022", # Best balance (default)
        "powerful": "claude-3-opus-20240229"      # Most capable
    }
    return models.get(performance_mode, models["balanced"])


def get_openai_model(performance_mode: str = "balanced") -> str:
    """Get OpenAI model based on performance requirements"""
    models = {
        "fast": "gpt-3.5-turbo",     # Fastest, cheapest
        "balanced": "gpt-4",         # Best balance (default)
        "powerful": "gpt-4-turbo"    # Most capable
    }
    return models.get(performance_mode, models["balanced"])


def create_ai_client(config: Dict[str, Any], performance_mode: str = "balanced") -> AIClient:
    """Create AI client with performance-optimized model selection"""
    # Auto-select best model for performance mode
    provider = config.get("ai", {}).get("provider", "claude")
    if provider == "claude":
        config["ai"]["model"] = get_claude_model(performance_mode)
    elif provider == "openai":
        config["ai"]["model"] = get_openai_model(performance_mode)
    
    return AIClient(config)
