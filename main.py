#!/usr/bin/env python3
"""
AI MCP Host - A Python application that hosts AI assistants with MCP integration
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

from ai_client import AIClient
from mcp_manager import MCPManager
from cli_interface import CLIInterface
from chat_service import ChatService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIMCPHost:
    """Main application class that orchestrates AI and MCP interactions"""
    
    def __init__(self, config_path: str = "mcp_config.json", verbose: bool = False):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.mcp_manager = MCPManager(self.config.get("mcpServers", {}))
        
        # Initialize AI client with error handling
        try:
            self.ai_client = AIClient(self.config.get("ai", {}))
        except (ValueError, ImportError) as e:
            logger.warning(f"AI client initialization failed: {e}")
            self.ai_client = None
        
        self.cli = CLIInterface(verbose=verbose)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._create_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        default_config = {
            "ai": {
                "provider": "openai",  # or "claude"
                "model": "gpt-4",
                "api_key_env": "OPENAI_API_KEY"  # Environment variable name
            },
            "mcpServers": {
                "example_mcp": {
                    "command": "python",
                    "args": ["-m", "example_mcp_server"],
                    "env": {},
                    "disabled": true
                }
            }
        }
        
        # Save default config
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Created default config at {self.config_path}")
        return default_config
    
    async def start(self):
        """Start the AI MCP Host application"""
        logger.info("Starting AI MCP Host...")
        
        try:
            # Initialize MCP connections
            await self.mcp_manager.initialize()
            
            # Start CLI interface
            await self.cli.start(self.ai_client, self.mcp_manager)
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Error during startup: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        await self.mcp_manager.cleanup()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AI MCP Host - Interactive assistant with MCP tool integration"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (show tool arguments and full results)"
    )
    parser.add_argument(
        "-c", "--config",
        default="mcp_config.json",
        help="Path to MCP configuration file (default: mcp_config.json)"
    )
    
    args = parser.parse_args()
    
    app = AIMCPHost(config_path=args.config, verbose=args.verbose)
    await app.start()


if __name__ == "__main__":
    asyncio.run(main())
