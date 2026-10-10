"""
Tests for MCP server tools registration.
"""

import pytest
from intelflow.mcp_server import mcp

@pytest.mark.asyncio
async def test_mcp_tools_registration():
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "ask_npu" in tool_names
    assert "summarize_code_npu" in tool_names
    assert "explain_symbol_npu" in tool_names
    assert "embed_npu" in tool_names
    assert "npu_status" in tool_names
