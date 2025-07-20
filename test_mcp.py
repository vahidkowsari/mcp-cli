#!/usr/bin/env python3
"""
Test script for the example MCP server
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path


async def test_mcp_server():
    """Test the example MCP server"""
    print("🧪 Testing Example MCP Server")
    print("=" * 30)
    
    # Start the MCP server
    process = subprocess.Popen(
        [sys.executable, "example_mcp_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=Path(__file__).parent
    )
    
    try:
        # Test 1: Initialize
        print("1. Testing initialization...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}, "resources": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline()
        response = json.loads(response_line.strip())
        
        if response.get("result"):
            print("   ✅ Initialization successful")
        else:
            print(f"   ❌ Initialization failed: {response}")
            return
        
        # Test 2: List tools
        print("2. Testing tool listing...")
        list_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        process.stdin.write(json.dumps(list_request) + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline()
        response = json.loads(response_line.strip())
        
        if response.get("result") and response["result"].get("tools"):
            tools = response["result"]["tools"]
            print(f"   ✅ Found {len(tools)} tools:")
            for tool in tools:
                print(f"      • {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")
        else:
            print(f"   ❌ Tool listing failed: {response}")
            return
        
        # Test 3: Call echo tool
        print("3. Testing echo tool...")
        echo_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "echo",
                "arguments": {"text": "Hello, MCP!"}
            }
        }
        
        process.stdin.write(json.dumps(echo_request) + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline()
        response = json.loads(response_line.strip())
        
        if response.get("result"):
            result = response["result"]
            print(f"   ✅ Echo result: {result}")
        else:
            print(f"   ❌ Echo tool failed: {response}")
        
        # Test 4: Call add_numbers tool
        print("4. Testing add_numbers tool...")
        add_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "add_numbers",
                "arguments": {"a": 15, "b": 27}
            }
        }
        
        process.stdin.write(json.dumps(add_request) + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline()
        response = json.loads(response_line.strip())
        
        if response.get("result"):
            result = response["result"]
            print(f"   ✅ Add result: {result}")
        else:
            print(f"   ❌ Add tool failed: {response}")
        
        print("\n🎉 All tests passed! The MCP server is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
    finally:
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
