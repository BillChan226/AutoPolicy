#!/usr/bin/env python3
"""
Certification MCP Tool for ShieldAgent.
Handles the final rule verification based on assigned predicate values.
"""

import json
import os
from typing import Dict, Any, List, Optional
from fastmcp import FastMCP
from openai import OpenAI
from anthropic import Anthropic
from shield.utils import *

# Create FastMCP instance
certification_mcp = FastMCP("Certification Tool")

# Global state for verification
current_rules: List[Dict[str, Any]] = []
current_predicates: Dict[str, Any] = {}
rule_map: Dict[str, Dict[str, Any]] = {} # Helper map for quick rule lookup


def logic_to_python(expr: str) -> str:
    expr = expr.replace("AND", " and ")
    expr = expr.replace("OR", " or ")
    expr = expr.replace("NOT", " not ")
    
    # Handle IMPLIES using correct transformation
    # Replace A IMPLIES B with "(not A) or B"
    def replace_implies(match):
        left = match.group(1).strip()
        right = match.group(2).strip()
        return f"(not ({left})) or ({right})"

    expr = re.sub(r"(.+?)\s+IMPLIES\s+(.+)", replace_implies, expr)
    
    # Optional: handle modal ops like ALWAYS, EVENTUALLY if needed
    expr = expr.replace("ALWAYS", "")
    expr = expr.replace("EVENTUALLY", "")
    
    return expr

@certification_mcp.tool()
async def initialize_verification_state(
    grounded_rules: List[Dict[str, Any]],
    predicate_map_state: Dict[str, Any]
) -> str:
    """
    Initializes the state for rule verification with grounded rules and assigned predicates.

    Args:
        grounded_rules: The list of grounded rules from the formulation step.
        predicate_map_state: The final state of the predicate map with assigned values.

    Returns:
        A summary string indicating success or failure.
    """
    global current_rules, current_predicates, rule_map
    try:
        current_rules = grounded_rules
        current_predicates = predicate_map_state
        # rule_map = {rule['source_rule_id']: rule for rule in grounded_rules} # Build lookup map
        rule_map = {rule['grounded_rule_id']: rule for rule in grounded_rules} # Build lookup map
        for rule in rule_map:
            rule_map[rule]['verified'] = False
        # with open("/scratch/czr/shieldagent/.cache/log.txt", "a") as f:
        #     f.write(f"\n\nCurrent rules: {grounded_rules}\n")
        # Quick sanity check: ensure all necessary predicates have values
        all_rule_predicates = set()
        for rule in grounded_rules:
            for predicates in rule.get("grounded_predicates", []):
                for pred_info in predicates:
                    all_rule_predicates.add(pred_info[0])

        missing_preds = []
        for pred_name in all_rule_predicates:
            if pred_name not in current_predicates or current_predicates[pred_name]['value'] is None:
                    missing_preds.append(pred_name)

        if missing_preds:
            return f"Error: Initialization failed. The following predicates required by rules are missing or unassigned: {', '.join(missing_preds)}"

        rule_count = len(current_rules)
        predicate_count = len(current_predicates)
        return f"Success: Verification state initialized with {rule_count} rules and {predicate_count} assigned predicates. Ready for rule verification."

    except Exception as e:
        current_rules = []
        current_predicates = {}
        rule_map = {}
        return f"Error initializing verification state: {str(e)}"

@certification_mcp.tool()
async def verify_fol_rules(rule_ids: List[int]) -> str:
    """
    Verifies a list of First-Order Logic (FOL) rules based on their logic and assigned predicate values.

    Args:
        rule_ids: A list of rule ids (int) to verify.
    """
    try:
        rule_ids = [int(rule_id) for rule_id in rule_ids]
    except Exception as e:
        return f"Error: Error converting rule IDs to integers: {str(e)}"
    
    if not current_rules or not current_predicates:
        return "Error: Verification state not initialized."

    # Filter the rules based on the provided rule_ids
    filtered_rules = [rule for rule in current_rules if rule['grounded_rule_id'] in rule_ids]
    if not filtered_rules:
        return "Error: No valid rules found for the provided rule IDs."

    results = {}
    for rule in filtered_rules:
        rule_id = rule['grounded_rule_id']
        rule_logic = rule.get("rule_logic")
        grounded_predicates_info = rule.get("grounded_predicates", [])
        predicate_names_in_rule = {pred[0] for pred in grounded_predicates_info}

        if not rule_logic:
            results[rule_id] = {"verified": False, "error": "Rule logic is missing."}
            continue

        # Prepare the scope for evaluation
        eval_scope = {}
        missing_eval_preds = []
        for pred_name in predicate_names_in_rule:
            if pred_name in current_predicates and current_predicates[pred_name]['value'] is not None:
                eval_scope[pred_name] = bool(current_predicates[pred_name]['value'])
            else:
                missing_eval_preds.append(pred_name)

        if missing_eval_preds:
            results[rule_id] = {"verified": False, "error": f"Cannot evaluate rule, missing values for predicates: {', '.join(missing_eval_preds)}"}
            continue

        # Add logical operators to scope safely
        eval_scope["AND"] = lambda a, b: bool(a) and bool(b)
        eval_scope["OR"] = lambda a, b: bool(a) or bool(b)
        eval_scope["NOT"] = lambda a: not bool(a)
        eval_scope["IMPLIES"] = lambda a, b: (not bool(a)) or bool(b)
        eval_scope["ALWAYS"] = lambda a: bool(a)
        eval_scope["EVENTUALLY"] = lambda a: bool(a)

        # Convert logic expression for safe evaluation
        try:
            py_logic = logic_to_python(rule_logic)
        except Exception as e:
            results[rule_id] = {"verified": False, "error": f"Error converting logic expression: {str(e)}"}
            continue

        try:
            # Use a restricted eval environment
            rule_result = eval(py_logic, {"__builtins__": {"True": True, "False": False}}, eval_scope)
            results[rule_id] = {"verified": bool(rule_result), "error": None}
            if rule_id in rule_map:
                rule_map[rule_id]['verified'] = results[rule_id]['verified']
        except Exception as e:
            results[rule_id] = {"verified": False, "error": f"Evaluation error: {str(e)}"}

    # Format the verification results in structured text format
    verification_summary = "\n".join(
        f"Rule {rule_id}: {'Violated' if not result['verified'] else 'Not Violated'}"
        for rule_id, result in results.items()
    )

    return verification_summary

@certification_mcp.tool()
async def static_certification(
    rule_violation_results: Dict[str, bool]
) -> Dict[str, Any]:
    """
    Determines the overall certification outcome based on rule violation results.

    Args:
        rule_violation_results: A dictionary of rule IDs and their violation status {rule_id: bool, ...} True means violated, False means not violated.

    """
    allowable_violations = 0
    violations = 0
    for rule_id, result in rule_violation_results.items():
        if result:
            violations += 1

    allowed = violations <= allowable_violations
    recommendation = "ALLOW" if allowed else "DENY"
    explanation = (
        f"Action is {'allowed' if allowed else 'denied'} with {violations} violations. "
        f"Allowable violations: {allowable_violations}."
    )

    return {
        "allowed": allowed,
        "recommendation": recommendation,
        "violations": violations,
        "allowable_violations": allowable_violations,
        "explanation": explanation
    }

@certification_mcp.tool()
async def get_unverified_rules() -> str:
    """Returns all the rules that have not been verified yet."""
    unverified_rules = [
        {"rule_id": rule_id, "rule_description": rule['rule_description']}
        for rule_id, rule in rule_map.items()
        if not rule['verified']
    ]
    if not unverified_rules:
        return "All rules are verified."
    else:
        # unverified_rules_str = "\n".join(
        #     f"Rule id: {rule['rule_id']}, Description: {rule['rule_description']}\n" for rule in unverified_rules
        # )
        # return unverified_rules_str
        return unverified_rules

@certification_mcp.tool()
async def verify_llm_rules(rule_ids: List[str]) -> str:
    """
    Uses an LLM to verify specified rules based on predicate assignments and rule descriptions.

    Args:
        rule_ids: A list of rule_ids to verify [int].
    """
    try:
        rule_ids = [int(rule_id) for rule_id in rule_ids]
    except Exception as e:
        return f"Error: Error converting rule IDs to integers: {str(e)}"
    
    if not current_rules or not current_predicates:
        return "Error: Verification state not initialized."

    # Filter the rules based on the provided rule_ids
    filtered_rules = [rule for rule in current_rules if rule['grounded_rule_id'] in rule_ids]
    if not filtered_rules:
        return "Error: No valid rules found for the provided rule IDs."

    # Prepare the prompt
    predicate_assignments = "\n".join(
        f"{name}: {info['value']}" for name, info in current_predicates.items()
    )
    rule_descriptions = "\n".join(
        f"Rule {rule['grounded_rule_id']}: {rule['rule_description']}" for rule in filtered_rules
    )

    prompt = f"""
You are an AI model tasked with verifying rules based on the following predicate assignments and rule descriptions.

Predicate Assignments:
{predicate_assignments}

Rule Descriptions:
{rule_descriptions}

Please output a JSON dictionary indicating which rules are violated. The format should be:
```json
{{
    "violations": [
        {{"rule_id": "rule_id_1", "violation": true/false}},
        ...
    ]
}}
```
"""

    system_prompt = """
    You are an expert in logic and reasoning. You will be given a list of rules and a set of predicate assignments. You will need to determine which rules are violated based on the predicate assignments.
    """

    try:
        # Call the chat_text function
        response = chat_text(
            prompt=prompt, 
            system=system_prompt, 
            model="claude-3-7-sonnet-20250219",
            client=anthropic_client,
            max_tokens=15000, 
            temperature=0.0,
        )

        # Extract the JSON part from the result
        rule_verification_results = extract_json(response)

        # Update the verified status based on the actual verification results
        for result in rule_verification_results.get("violations", []):
            rule_id = int(result["rule_id"])
            if rule_id in rule_map:
                rule_map[rule_id]['verified'] = True 

        # Format the verification results in structured text format
        verification_summary = "\n".join(
            f"Rule {result['rule_id']}: {'Violated' if result['violation'] else 'Not Violated'}"
            for result in rule_verification_results.get("violations", [])
        )

        return verification_summary

    except Exception as e:
        return {"status": "error", "message": f"Error during LLM rule verification: {str(e)}"}

@certification_mcp.tool()
async def reset_state() -> Dict[str, Any]:
    """Reset all variables (NOT AVAILABLE, ADMIN ONLY)."""
    global current_rules, current_predicates, rule_map
    current_rules = []
    current_predicates = {}
    rule_map = {}
    return "All verification state reset to initial state."



if __name__ == "__main__":
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    certification_mcp.run()
