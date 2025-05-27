#!/usr/bin/env python3
"""
Memory MCP Tool for ShieldAgent.
Provides short-term and long-term memory capabilities.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
import sqlite3
from datetime import datetime
import hashlib

from fastmcp import FastMCP

# Create FastMCP instance
memory_mcp = FastMCP("Memory Tool")



# Set up the SQLite database for long-term memory storage
def _setup_database():
    """Set up the SQLite database for long-term memory storage."""
    conn = sqlite3.connect(memory_db_path)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS workflows (
        id TEXT PRIMARY KEY,
        action_name TEXT,
        workflow TEXT,
        created_at TEXT,
        last_used TEXT,
        use_count INTEGER DEFAULT 1
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        type TEXT,
        content TEXT,
        metadata TEXT,
        created_at TEXT,
        last_accessed TEXT,
        access_count INTEGER DEFAULT 1
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print(f"[MemoryTool] Database initialized at {memory_db_path}")


def set_db_path(path: str) -> Dict[str, Any]:
    """Set the database path and reinitialize if needed"""
    global memory_db_path
    
    if not path:
        return {"status": "error", "message": "No path provided"}
    
    memory_db_path = path
    _setup_database()  # Reinitialize database with new path
    
    return {
        "status": "success",
        "message": f"Database path set to {path} and reinitialized"
    }

# Helper function to clean expired items
def _clean_expired_short_term():
    """Remove expired items from short-term memory."""
    current_time = time.time()
    expired_keys = [k for k, v in short_term_memory.items() if v["expiry"] < current_time]
    
    for key in expired_keys:
        del short_term_memory[key]

@memory_mcp.tool()
def store_short_term(key: str, value: Any, ttl: int = 3600) -> Dict[str, Any]:
    """Store data in short-term memory with a TTL"""
    global short_term_memory
    
    if not key:
        return {"status": "error", "message": "No key provided"}
    
    expiry = time.time() + ttl
    short_term_memory[key] = {
        "value": value,
        "expiry": expiry
    }
    
    return {
        "status": "success",
        "message": f"Stored data with key '{key}' in short-term memory",
        "expiry": datetime.fromtimestamp(expiry).isoformat()
    }

@memory_mcp.tool()
def get_short_term(key: str) -> Dict[str, Any]:
    """Retrieve data from short-term memory"""
    global short_term_memory
    
    if not key:
        return {"status": "error", "message": "No key provided"}
    
    # Clean expired items
    _clean_expired_short_term()
    
    if key not in short_term_memory:
        return {"status": "error", "message": f"Key '{key}' not found in short-term memory"}
    
    memory_item = short_term_memory[key]
    
    return {
        "status": "success",
        "data": memory_item["value"],
        "expiry": datetime.fromtimestamp(memory_item["expiry"]).isoformat()
    }

@memory_mcp.tool()
def clear_short_term(key: str = "") -> Dict[str, Any]:
    """Clear items from short-term memory"""
    global short_term_memory
    
    if key:
        if key in short_term_memory:
            del short_term_memory[key]
            return {"status": "success", "message": f"Cleared key '{key}' from short-term memory"}
        else:
            return {"status": "error", "message": f"Key '{key}' not found in short-term memory"}
    else:
        # Clear all short-term memory
        short_term_memory.clear()
        return {"status": "success", "message": "Cleared all short-term memory"}

@memory_mcp.tool()
def store_workflow(action_name: str, workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Store a successful verification workflow"""
    if not action_name:
        return {"status": "error", "message": "No action_name provided"}
    
    if not workflow:
        return {"status": "error", "message": "No workflow data provided"}
    
    try:
        conn = sqlite3.connect(memory_db_path)
        cursor = conn.cursor()
        
        # Create a hash ID based on action name and workflow content
        id_string = f"{action_name}:{json.dumps(workflow, sort_keys=True)}"
        workflow_id = hashlib.md5(id_string.encode()).hexdigest()
        
        now = datetime.now().isoformat()
        
        # Check if this workflow already exists
        cursor.execute("SELECT id, use_count FROM workflows WHERE id = ?", (workflow_id,))
        existing = cursor.fetchone()
        
        if existing:
            # Update the existing workflow
            cursor.execute(
                "UPDATE workflows SET last_used = ?, use_count = use_count + 1 WHERE id = ?",
                (now, workflow_id)
            )
            use_count = existing[1] + 1
        else:
            # Insert new workflow
            cursor.execute(
                "INSERT INTO workflows (id, action_name, workflow, created_at, last_used, use_count) VALUES (?, ?, ?, ?, ?, ?)",
                (workflow_id, action_name, json.dumps(workflow), now, now, 1)
            )
            use_count = 1
        
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "message": f"Stored workflow for action '{action_name}'",
            "workflow_id": workflow_id,
            "use_count": use_count
        }
    
    except Exception as e:
        return {"status": "error", "message": f"Error storing workflow: {str(e)}"}

@memory_mcp.tool()
def get_workflow(action_name: str) -> Dict[str, Any]:
    """Retrieve workflows for a specific action"""
    if not action_name:
        return {"status": "error", "message": "No action_name provided"}
    
    try:
        conn = sqlite3.connect(memory_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get workflows for this action, sorted by use count
        cursor.execute(
            "SELECT * FROM workflows WHERE action_name = ? ORDER BY use_count DESC LIMIT 5",
            (action_name,)
        )
        
        rows = cursor.fetchall()
        workflows = []
        
        for row in rows:
            workflow_data = dict(row)
            workflow_data["workflow"] = json.loads(workflow_data["workflow"])
            workflows.append(workflow_data)
        
        conn.close()
        
        if workflows:
            return {
                "status": "success",
                "action_name": action_name,
                "workflows": workflows
            }
        else:
            return {
                "status": "success",
                "action_name": action_name,
                "workflows": [],
                "message": f"No workflows found for action '{action_name}'"
            }
            
    except Exception as e:
        return {"status": "error", "message": f"Error retrieving workflows: {str(e)}"}

@memory_mcp.tool()
def store_memory(memory_type: str, content: Any, metadata: Dict[str, Any] = {}) -> Dict[str, Any]:
    """Store a memory item in long-term memory"""
    if not memory_type:
        return {"status": "error", "message": "No memory_type provided"}
    
    if not content:
        return {"status": "error", "message": "No content provided"}
    
    try:
        conn = sqlite3.connect(memory_db_path)
        cursor = conn.cursor()
        
        # Create a hash ID based on type and content
        content_str = json.dumps(content, sort_keys=True) if isinstance(content, dict) else str(content)
        id_string = f"{memory_type}:{content_str}"
        memory_id = hashlib.md5(id_string.encode()).hexdigest()
        
        now = datetime.now().isoformat()
        
        # Check if this memory already exists
        cursor.execute("SELECT id, access_count FROM memories WHERE id = ?", (memory_id,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing memory
            cursor.execute(
                "UPDATE memories SET last_accessed = ?, access_count = access_count + 1, metadata = ? WHERE id = ?",
                (now, json.dumps(metadata), memory_id)
            )
            access_count = existing[1] + 1
        else:
            # Insert new memory
            cursor.execute(
                "INSERT INTO memories (id, type, content, metadata, created_at, last_accessed, access_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (memory_id, memory_type, json.dumps(content), json.dumps(metadata), now, now, 1)
            )
            access_count = 1
        
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "message": f"Stored memory of type '{memory_type}'",
            "memory_id": memory_id,
            "access_count": access_count
        }
    
    except Exception as e:
        return {"status": "error", "message": f"Error storing memory: {str(e)}"}

@memory_mcp.tool()
def search_memories(query: str = "", memory_type: str = "", limit: int = 5) -> Dict[str, Any]:
    """Search long-term memory for relevant items"""
    try:
        conn = sqlite3.connect(memory_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        sql_params = []
        sql_query = "SELECT * FROM memories WHERE 1=1"
        
        # Add type filter if provided
        if memory_type:
            sql_query += " AND type = ?"
            sql_params.append(memory_type)
        
        # Add content search if query provided
        if query:
            sql_query += " AND (content LIKE ? OR metadata LIKE ?)"
            sql_params.extend([f"%{query}%", f"%{query}%"])
        
        # Add sorting and limit
        sql_query += " ORDER BY access_count DESC LIMIT ?"
        sql_params.append(limit)
        
        cursor.execute(sql_query, sql_params)
        
        rows = cursor.fetchall()
        memories = []
        
        for row in rows:
            memory_data = dict(row)
            memory_data["content"] = json.loads(memory_data["content"])
            memory_data["metadata"] = json.loads(memory_data["metadata"])
            memories.append(memory_data)
        
        conn.close()
        
        return {
            "status": "success",
            "query": query,
            "type": memory_type,
            "memories": memories,
            "count": len(memories)
        }
            
    except Exception as e:
        return {"status": "error", "message": f"Error searching memories: {str(e)}"}

@memory_mcp.tool()
def store_interaction_context(trace_id: str, context_data: Dict[str, Any], ttl: int = 3600) -> Dict[str, Any]:
    """
    Store interaction context data in short-term memory using a consistent key structure.
    
    This tool provides a single entry point for storing context data related to an agent interaction,
    such as conversation history, thought processes, and other contextual information.
    
    Args:
        trace_id: The unique identifier for the current verification trace
        context_data: Dictionary containing context elements like 'history', 'thought_text', etc.
        ttl: Time-to-live for the stored data in seconds (default: 1 hour)
        
    Returns:
        Dictionary with status and details about the stored context
    """
    global short_term_memory
    
    if not trace_id:
        return {"status": "error", "message": "No trace_id provided"}
    
    if not context_data:
        return {"status": "error", "message": "No context data provided"}
    
    # Track what was successfully stored
    stored_items = []
    errors = []
    
    # Store each context item with a consistent key pattern
    for context_key, context_value in context_data.items():
        if context_value is not None:
            key = f"{context_key}:{trace_id}"
            expiry = time.time() + ttl
            
            try:
                short_term_memory[key] = {
                    "value": context_value,
                    "expiry": expiry
                }
                stored_items.append(context_key)
            except Exception as e:
                errors.append(f"Failed to store {context_key}: {str(e)}")
    
    # Return results
    if errors:
        return {
            "status": "partial",
            "message": f"Stored {len(stored_items)} context items with some errors",
            "stored": stored_items,
            "errors": errors,
            "expiry": datetime.fromtimestamp(time.time() + ttl).isoformat()
        }
    
    return {
        "status": "success",
        "message": f"Successfully stored all context data for trace {trace_id}",
        "stored": stored_items,
        "expiry": datetime.fromtimestamp(time.time() + ttl).isoformat()
    }

if __name__ == "__main__":
    # Initialize global state
    short_term_memory = {}
    # Initialize database and load data before starting MCP server
    try:
        default_memory_path = os.environ.get("MEMORY_PATH")
        if default_memory_path:
            set_db_path(os.path.join(os.getcwd(), "memory_data.db"))
            print("[MemoryTool] Initialized with default memory data")
    except Exception as e:
        print(f"[MemoryTool] Warning: Could not load default memory data: {str(e)}")

    memory_mcp.run()