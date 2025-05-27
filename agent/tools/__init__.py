"""
ShieldAgent Tools Package

This package contains all the MCP tools used by ShieldAgent for verification operations.
These tools include:
- Content moderation
- Formal verification
- ASPM integration
- Memory management
- Trajectory analysis
"""

# Import FastMCP
from fastmcp import FastMCP

# Import the modules themselves
import shield.tools.content_moderation as content_moderation_mcp_module
import shield.tools.verification as verification_mcp_module
import shield.tools.aspm_tool as aspm_mcp_module
import shield.tools.memory_tool as memory_mcp_module
import shield.tools.certification as certification_mcp_module

# Also retain the FastMCP instances for backwards compatibility
from shield.tools.content_moderation import content_moderation_mcp
from shield.tools.verification import verification_mcp
from shield.tools.aspm_tool import aspm_mcp
from shield.tools.memory_tool import memory_mcp
from shield.tools.certification import certification_mcp

# Expose the MCP server instances
__all__ = [
    'content_moderation_mcp',
    'verification_mcp',
    'aspm_mcp',
    'memory_mcp',
    'certification_mcp'
] 