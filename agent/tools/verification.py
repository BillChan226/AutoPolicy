#!/usr/bin/env python3
"""
Verification MCP Tool for ShieldAgent.
Provides formal verification capabilities for rule checking.
"""

import os
import json
from typing import Dict, Any, List, Optional, Set, Union
from shield.utils import *
from fastmcp import FastMCP

# Create FastMCP instance
verification_mcp = FastMCP("Verification Tool")

# Add global state to store predicate map
predicate_map = {}

@verification_mcp.tool()
def verify_rule(rule_id: str, rule_expression: str, predicates: Dict[str, bool]) -> Dict[str, Any]:
    """Verify a single rule using predicate values"""
    if not rule_id or not rule_expression:
        return {"status": "error", "message": "Missing rule_id or rule_expression"}
    
    try:
        # Create a safe verification environment
        verification_scope = predicates.copy()
        
        # Add some basic logical operations
        verification_scope["and"] = lambda a, b: a and b
        verification_scope["or"] = lambda a, b: a or b
        verification_scope["not"] = lambda a: not a
        verification_scope["implies"] = lambda a, b: (not a) or b
        
        # Evaluate the rule expression
        result = eval(rule_expression, {"__builtins__": {}}, verification_scope)
        
        return {
            "status": "success",
            "rule_id": rule_id,
            "verified": bool(result),
            "predicates_used": list(predicates.keys())
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Verification error: {str(e)}",
            "rule_id": rule_id
        }


@verification_mcp.tool()
async def create_predicate_map(grounded_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Initialize predicate map from grounded rules"""
    global predicate_map
    
    try:
        predicate_map.clear()
        for rule in grounded_rules:
            for predicate in rule["grounded_predicates"]:
                name, description, pred_type = predicate
                if name not in predicate_map:
                    predicate_map[name] = {
                        "value": None,  # None indicates unassigned
                        "description": description,
                        "type": pred_type,
                        "source_rules": [rule["source_rule_id"]],   
                        "grounded_rule_ids": [rule["grounded_rule_id"]]
                    }
                else:
                    # Add additional source rule if not already present
                    if rule["source_rule_id"] not in predicate_map[name]["source_rules"]:
                        predicate_map[name]["source_rules"].append(rule["source_rule_id"])
                    # Add additional grounded rule if not already present
                    if rule["grounded_rule_id"] not in predicate_map[name]["grounded_rule_ids"]:
                        predicate_map[name]["grounded_rule_ids"].append(rule["grounded_rule_id"])
        
        return {
            "status": "success",
            "predicate_map": predicate_map
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error creating predicate map: {str(e)}"
        }

@verification_mcp.tool()
async def get_unverified_predicates() -> str:
    """Get all predicates that haven't been assigned values yet, formatted as text."""
    try:
        unverified_text = "Unverified Predicates:\n"
        count = 0
        for name, info in predicate_map.items():
            if info["value"] is None:
                count += 1
                unverified_text += f"\n{count}. Predicate Name: {name}\n"
                unverified_text += f"   Description: {info['description']}\n"
                unverified_text += f"   Type: {info['type']}\n"
        
        if count == 0:
            unverified_text += "  (None)\n"
            
        return unverified_text
    except Exception as e:
        return f"Error getting unverified predicates: {str(e)}"

@verification_mcp.tool()
async def get_predicate_info(predicate_names: List[str]) -> str:
    """Get details (value, description, type) for a list of predicate names, formatted as text."""
    try:
        output_text = "Predicate Information:\n"
        found_count = 0
        errors = []
        
        for name in predicate_names:
            if name in predicate_map:
                found_count += 1
                info = predicate_map[name]
                value_str = str(info['value']) if info['value'] is not None else "Unassigned"
                output_text += f"\nPredicate: {name}\n"
                output_text += f"  Value: {value_str}\n"
                output_text += f"  Description: {info['description']}\n"
                output_text += f"  Type: {info['type']}\n"
            else:
                errors.append(f"Predicate '{name}' not found in map")
        
        if found_count == 0 and not errors:
             output_text += "  (No information found for requested predicates)\n"
             
        if errors:
            output_text += "\nErrors:\n"
            for error in errors:
                output_text += f"  - {error}\n"
                
        return output_text.strip() # Remove trailing newline if any
        
    except Exception as e:
        return f"Error getting predicate info: {str(e)}"

@verification_mcp.tool()
async def update_predicate_values(assignments: Union[Dict[str, bool], Dict[str, List[Dict[str, Any]]]]) -> str:
    """
    Update values for predicates in the map and return a summary string.
    Accepts assignments as either:
    1. A direct dictionary: {"predicate_name": value, ...}
    2. A nested structure: {"values": [{"predicate": name, "value": val}, ...]}
    """
    try:
        updated_names = []
        errors = []
        
        # Normalize assignments to Dict[str, bool] format
        processed_assignments = {}
        if isinstance(assignments, dict) and "values" in assignments and isinstance(assignments["values"], list):
            # Handle {"values": [{"predicate": name, "value": val}, ...]} format
            for item in assignments["values"]:
                if isinstance(item, dict) and "predicate" in item and "value" in item:
                    processed_assignments[item["predicate"]] = bool(item["value"])
                else:
                    errors.append(f"Invalid item format in 'values' list: {item}")
        elif isinstance(assignments, dict):
            # Assume direct {"predicate_name": value, ...} format
             processed_assignments = {k: bool(v) for k, v in assignments.items()}
        else:
             return "Error: Invalid format for assignments. Expected a dictionary."

        # Process the normalized assignments
        for name, value in processed_assignments.items():
            if name in predicate_map:
                predicate_map[name]["value"] = value # Value is already bool
                updated_names.append(f"{name}={value}")
            else:
                errors.append(f"Predicate '{name}' not found in map")
        
        # Build summary string
        summary = "Predicate Update Summary:\n"
        if updated_names:
            summary += f"- Successfully updated: {', '.join(updated_names)}\n"
        else:
            # Check if there were items to process but none were found/valid
            if processed_assignments or (isinstance(assignments, dict) and "values" in assignments):
                 summary += "- No predicates were updated successfully.\n"
            else: # Original input might have been empty or invalid structure handled above
                 summary += "- No valid assignments provided to update.\n"
            
        if errors:
            summary += "- Errors encountered:\n"
            for error in errors:
                summary += f"  - {error}\n"
                
        return summary
        
    except Exception as e:
        return f"Error updating predicate values: {str(e)}"

@verification_mcp.tool()
async def get_predicate_map() -> Dict[str, Any]:
    """Get the current state of the predicate map"""
    return {
        "status": "success",
        "predicate_map": predicate_map
    }

@verification_mcp.tool()
async def reset_state() -> Dict[str, Any]:
    """Reset all variables (NOT AVAILABLE, ADMIN ONLY)."""
    global predicate_map
    predicate_map = {}
    return "All verification state reset to initial state."



if __name__ == "__main__":
    verification_mcp.run()
