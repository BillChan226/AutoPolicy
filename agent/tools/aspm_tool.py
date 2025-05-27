#!/usr/bin/env python3
"""
ASPM (Action-based Probabilistic Safety Model) Tool for ShieldAgent.
Provides access to ASPM data and rule circuit retrieval.
"""

import os
import json
from typing import Dict, Any, List, Optional, Set
from fastmcp import FastMCP
from dataclasses import dataclass
from anthropic import Anthropic
from openai import OpenAI
import re
from shield.utils import *

# Create FastMCP instance
aspm_mcp = FastMCP("ASPM Tool")

@dataclass
class ActionNode:
    """Represents a node in the action tree"""
    name: str
    parent_path: List[str]  # List of parent node names from root to current
    layer: int
    l2_children: List['ActionNode']
    l3_children: List['ActionNode']

class ActionTree:
    """Represents the complete action tree with easy path lookup"""
    def __init__(self):
        self.nodes: Dict[str, ActionNode] = {}  # name -> node mapping
        self.root_nodes: List[ActionNode] = []  # L1 actions
    
    @classmethod
    def from_dict(cls, tree_dict: Dict[str, Any]) -> 'ActionTree':
        """Create ActionTree from dictionary representation"""
        tree = cls()
        
        def build_node(node_data: Dict[str, Any], parent_path: List[str], layer: int) -> ActionNode:
            name = node_data.get("name", "")
            current_path = parent_path + [name]
            
            # Create node
            node = ActionNode(
                name=name,
                parent_path=parent_path,
                layer=layer,
                l2_children=[],
                l3_children=[]
            )
            
            # Add L2 children
            for child in node_data.get("L2_children", []):
                if isinstance(child, dict):
                    child_node = build_node(child, current_path, layer + 1)
                    node.l2_children.append(child_node)
                else:
                    # Handle string children
                    child_node = ActionNode(
                        name=child,
                        parent_path=current_path,
                        layer=layer + 1,
                        l2_children=[],
                        l3_children=[]
                    )
                    node.l2_children.append(child_node)
                tree.nodes[child_node.name] = child_node
            
            # Add L3 children
            for child in node_data.get("L3_children", []):
                if isinstance(child, dict):
                    child_node = build_node(child, current_path, layer + 1)
                    node.l3_children.append(child_node)
                else:
                    child_node = ActionNode(
                        name=child,
                        parent_path=current_path,
                        layer=layer + 1,
                        l2_children=[],
                        l3_children=[]
                    )
                    node.l3_children.append(child_node)
                tree.nodes[child_node.name] = child_node
            
            return node
        
        # Build tree from L1 actions
        for l1_action in tree_dict.get("L1_actions", []):
            root_node = build_node(l1_action, [], 1)
            tree.root_nodes.append(root_node)
            tree.nodes[root_node.name] = root_node
        
        return tree
    
    def get_node(self, name: str) -> Optional[ActionNode]:
        """Get node by name"""
        return self.nodes.get(name)
    
    def get_parent_path(self, name: str) -> Optional[List[str]]:
        """Get parent path for a node"""
        node = self.get_node(name)
        return node.parent_path if node else None
    
    def get_upstream_actions(self, name: str) -> Dict[str, List[str]]:
        """Get upstream actions by layer"""
        node = self.get_node(name)
        if not node:
            return {}
        
        result = {}
        for i, parent in enumerate(node.parent_path):
            layer = f"L{i+1}"
            result[layer] = [parent]
        
        return result
    
    def get_all_predicates(self) -> Set[str]:
        """Get all predicate names in the tree"""
        return set(self.nodes.keys())

# Global state to store ASPM data
aspm_tree = None
aspm_actions = {}
aspm_rules = {}
aspm_condition_clusters = {}

def load_aspm_json(path: str) -> Dict[str, Any]:
    """Load ASPM data from a JSON file into memory"""
    global aspm_tree, aspm_actions, aspm_rules, aspm_condition_clusters
    
    if not path:
        raise ValueError("No path provided")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        # Load JSON data
        with open(path, 'r', encoding='utf-8') as f:
            aspm_data = json.load(f)
        
        # Create action tree
        aspm_tree = ActionTree.from_dict(aspm_data.get("action_tree", {}))
        
        # Store other data
        aspm_actions = aspm_data.get("actions", {})
        aspm_rules = {rule["rule_id"]: rule for rule in aspm_data.get("rule_list", [])}
        aspm_condition_clusters = aspm_data.get("condition_clusters", {})
        
        print(f"[ASPMTool] Loaded ASPM from {path} with {len(aspm_actions)} actions and {len(aspm_rules)} rules")
        print(f"[ASPMTool] Action tree contains {len(aspm_tree.nodes)} nodes")
        return {
            "status": "success",
            "action_count": len(aspm_actions),
            "rule_count": len(aspm_rules)
        }
    except Exception as e:
        raise Exception(f"Error loading ASPM data: {str(e)}")


def get_rules_info(rule_ids: List[str]) -> Dict[str, Any]:
    """Get complete information for specified rules"""
    if not rule_ids:
        return {"status": "error", "message": "No rule IDs provided"}

    rules_info = {}
    for rule_id in rule_ids:
        rule_data = aspm_rules.get(rule_id)
        if rule_data:
            rules_info[rule_id] = {
                "predicates": rule_data.get("predicates", []),
                "logic": rule_data.get("logic", ""),
                "description": rule_data.get("description", ""),
                "rule_type": rule_data.get("rule_type", "")
            }
    
    return rules_info
        

@aspm_mcp.tool()
async def get_action_circuit(action_names: List[str]) -> Dict[str, Any]:
    """Get the relevant rule circuits for a list of actions"""
    if not action_names:
        return {"status": "error", "message": "No action_names provided"}
    
    try:    
        action_circuits = {}
        for action_name in action_names:
            if not aspm_actions.get(action_name):
                return {"status": "error", "message": f"Action '{action_name}' not found in ASPM"}
            action_circuits[action_name] = {}
            # Get all related rules
            rule_ids = set(aspm_actions[action_name].get("soft_rule_ids", []))
            rules = get_rules_info(list(rule_ids))
            action_circuits[action_name]["rule_ids"] = list(rule_ids)
            action_circuits[action_name]["rules"] = rules
        
        return {
            "status": "success",
            "action_circuits": action_circuits
        }
    except Exception as e:
        return {"status": "error", "message": f"Error retrieving action circuit: {str(e)}"}


@aspm_mcp.tool()
async def action_grounding(agent_thought: str, agent_action: str) -> Dict[str, Any]:
    """Extract and ground all relevant action predicates from current action context"""
    if not agent_action:
        return {"status": "error", "message": "No action context provided"}
        
    try:
        if not aspm_tree:
            return {"status": "error", "message": "Action tree not loaded. Please load ASPM data first."}
        
        # Get all predicates from tree
        all_predicates = aspm_tree.get_all_predicates()
        predicates_list = "\n".join(sorted(all_predicates))
        
        system_prompt = """You are a helpful assistant to extract relevant actions from a LLM agent's output and match it with the available action predicates provided. You should identify the most critical actions that require safety verification (maximum 3 actions)."""
        
        extraction_prompt = f"""Given the following context of a LLM agent's action being invoked at the current time step, please match its most relevant action predicate(s) from the list of available action predicates.

Current action context:
{agent_thought}

Current action:
{agent_action}

Available Action Predicates:
{predicates_list}

Output only the matched action predicate(s) separated by semicolons (;) with no additional text or explanations. Example output format: "action1;action2" or just "action1" for a single action."""
        
        # save the prompt to a file
        # with open("/scratch/czr/shieldagent/.cache/log.txt", "w") as f:
        #     f.write(extraction_prompt)
        for _ in range(3):
            try:
                extraction_result = chat_text(prompt=extraction_prompt, 
                                              system=system_prompt,
                                              model="claude-3-7-sonnet-20250219",
                                              client=anthropic_client,
                                              max_tokens=256, 
                                              temperature=0.0)
                # with open("/scratch/czr/shieldagent/.cache/log.txt", "a") as f:
                #     f.write(f"\n\nGrounding result: {extraction_result}\n")
                matched_predicates = [p.strip() for p in extraction_result.split(';') if p.strip()]

                # Verify all predicates exist
                for predicate in matched_predicates:
                    if predicate not in all_predicates:
                        matched_predicates = []
                        raise Exception(f"Predicate {predicate} not found in ASPM")
                break

            except Exception as e:
                continue

        if not matched_predicates:
            return {"status": "error", "message": "No valid action predicates found"}

        
        # For each matched predicate, get upstream predicates using the tree
        result_predicates = {}
        for predicate in matched_predicates:
            node = aspm_tree.get_node(predicate)
            result_predicates[predicate] = {
                "name": predicate,
                "upstream_predicates": aspm_tree.get_upstream_actions(predicate),
                "matched_locations": [{
                    "path": node.parent_path + [predicate],
                    "layer": node.layer
                }]
            }
        
        return {
            "status": "success",
            "matched_predicates": result_predicates
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error in action grounding: {str(e)}"}



def validate_rule_consistency(rule):
    """
    Validate that a grounded rule's predicates and logic are consistent.
    Returns (is_valid, error_message)
    """
    # Get all predicate names from the grounded predicates
    predicate_names = set(pred[0] for pred in rule["grounded_predicates"])
    
    # Extract all predicates used in the logic
    # First remove common logical operators
    logic = rule["rule_logic"]
    for op in ["IMPLIES", "AND", "OR", "NOT", "ALWAYS", "EVENTUALLY", "UNTIL", "(", ")"]:
        logic = logic.replace(op, " ")
    
    # Get all words in logic as potential predicates
    logic_predicates = set(word.strip() for word in logic.split())
    
    # Check if all predicates in logic are in predicate list
    undefined_predicates = logic_predicates - predicate_names
    if undefined_predicates:
        return False, f"Logic contains undefined predicates: {undefined_predicates}"
    
    # Check if all predicates in predicate list are used in logic
    unused_predicates = predicate_names - logic_predicates
    if unused_predicates:
        return False, f"Predicates not used in logic: {unused_predicates}"
    
    return True, "Valid"



@aspm_mcp.tool()
async def formulate_solution_space(
    agent_thought: str,
    agent_action: str,
    action_predicates: List[str],
    action_circuits: Dict[str, Any],
    reasoning: bool = False
) -> Dict[str, Any]:
    """
    Formulate the solution space by:
    1. Filtering relevant rules based on context
    2. Grounding predicates to concrete context
    """

    # First restructure the rules to remove duplicates
    unique_rules = {}
    for action_name, circuit in action_circuits.items():
        for rule_id, rule_info in circuit.get("rules", {}).items():
            if rule_id not in unique_rules:
                unique_rules[rule_id] = {
                    "rule_id": rule_id,
                    "predicates": rule_info["predicates"],
                    "logic": rule_info["logic"],
                    "description": rule_info["description"],
                    "rule_type": rule_info["rule_type"]
                }

    # Sort by rule ID
    unique_rules = dict(sorted(unique_rules.items(), key=lambda x: int(x[0])))
    
    # Format rules for prompt
    rules_text = "Available Rules to Ground:\n"
    for rule_id, rule in unique_rules.items():
        rules_text += f"\nRule id: {rule_id}\n"
        rules_text += f"Rule type: {rule['rule_type']} rule\n"
        rules_text += f"Description: {rule['description']}\n"
        rules_text += "Predicates:\n"
        for pred in rule["predicates"]:
            predicate_type = pred[3]
            if predicate_type == "action":
                rules_text += f"- {pred[0]}: {pred[1]} (Type: action predicate)\n"
            elif predicate_type == "condition":
                rules_text += f"- {pred[0]}: {pred[1]} (Type: state predicate)\n"
        rules_text += f"Logic: {rule['logic']}\n"
        rules_text += "-" * 50 + "\n"

    system_prompt_reasoning = """You are a helpful assistant to ground abstract safety rules which constrain agents into more concrete, verifiable rules with grounded predicates based on the current agent's interaction context.
Specifically, there are two types of rules: (1) action rule: which sets specific constraints on the agent's invoked action, and (2) physical rule: which describes the internal constraints of the agent's system states or the environment variables.
Given the agent's current interaction context with the environment and all relevant rules, your task is to:
1. Filter out irrelevant rules whose predicates are completely not relevant to the current context (you should only discard rules when they are completely off-topic);
2. Ground remaining rules by making their predicates concrete and specific to the agent's interaction context;
"""
    
    formulation_prompt_reasoning = f"""
Current Context:
Agent Thought: {agent_thought}
Agent Action: {agent_action}
Matched Action Predicates: {action_predicates}

{rules_text}

For each relevant rule:
1. Determine if the rule's predicates can be grounded in the current context, you should keep the rule even if it is just slightly relevant to the current context; Only discard rules when the rules are completely irrelevant to the current context;
2. If yes, create concrete versions of the predicates that are specific and verifiable, make sure both your "state predicate" clearly describes the target system or environment state, and "action predicate" clearly describes the target invoked action;
3. It is possible that one source rule can be grounded into multiple concrete rules (for example, submit_data can be grounded to post_data_1 and post_data_2), and you should list all the possible grounded concrete rules;
4. Update the rule description and logic to match the concrete context;
5. Maintain the original rule logic structure but with grounded predicates;
6. Make sure your updated logic only contains the grounded predicates, and each of your grounded predicates must also appear in your updated logic.

Please make sure to carefully ground each rule and do not miss any possibly relevant rules. 
Provide a brief reasoning process first to carefully decide whether to discard or ground a rule (and how many different concrete rules should be grounded into), and then output the grounded rules in this JSON format:
```json
{{
"grounded_rules": [
    {{
        "source_rule_id": "original_rule_id",
        "grounded_predicates": [
            ["predicate_name", "concrete_description", "predicate_type"]
        ],
        "rule_logic": "updated_logic_expression",
        "rule_description": "updated_rule_description"
    }}
]
}}
```
"""
    
    system_prompt_non_reasoning = """You are a helpful assistant to ground abstract safety rules which constrain agents into more concrete, verifiable rules with grounded predicates based on the current agent's interaction context.
Specifically, there are two types of rules: (1) action rule: which sets specific constraints on the agent's invoked action, and (2) physical rule: which describes the internal constraints of the agent's system states or the environment variables.
Given the agent's current interaction context with the environment and all relevant rules, your task is to:
1. Filter out irrelevant rules whose predicates are completely not relevant to the current context (you should only discard rules when they are completely off-topic);
2. Ground remaining rules by making their predicates concrete and specific to the agent's interaction context;
Your output should contain ONLY the grounded rules in this JSON format and no other text or explanations.
"""
    
    formulation_prompt_non_reasoning = f"""
Current Context:
Agent Thought: {agent_thought}
Agent Action: {agent_action}
Matched Action Predicates: {action_predicates}

{rules_text}

For each relevant rule:
1. Determine if the rule's predicates can be grounded in the current context, you should keep the rule even if it is just slightly relevant to the current context; Only discard rules when the rules are completely irrelevant to the current context;
2. If yes, create concrete versions of the predicates that are specific and verifiable, make sure both your "state predicate" clearly describes the target system or environment state, and "action predicate" clearly describes the target invoked action;
3. It is possible that one source rule can be grounded into multiple concrete rules (for example, submit_data can be grounded to post_data_1 and post_data_2), and you should list all the possible grounded concrete rules;
4. Update the rule description and logic to match the concrete context;
5. Maintain the original rule logic structure but with grounded predicates;
6. Make sure your updated logic only contains the grounded predicates, and each of your grounded predicates must also appear in your updated logic.

Please make sure to carefully ground each rule and do not miss any possibly relevant rules. 
Please only output the grounded rules in this JSON format and no other text or explanations:
```json
{{
"grounded_rules": [
    {{
        "source_rule_id": "original_rule_id",
        "grounded_predicates": [
            ["predicate_name", "concrete_description", "predicate_type"]
        ],
        "rule_logic": "updated_logic_expression",
        "rule_description": "updated_rule_description"
    }}
    ...
]
}}
```
"""
    
    system_prompt = system_prompt_reasoning if reasoning else system_prompt_non_reasoning
    formulation_prompt = formulation_prompt_reasoning if reasoning else formulation_prompt_non_reasoning
    
    for _ in range(3):
        try:
        # if True:
            # Get grounding result from LLM
            grounding_result = chat_text(
                prompt=formulation_prompt,
                system=system_prompt,
                model="claude-3-7-sonnet-20250219",
                client=anthropic_client,
                max_tokens=15000,
                temperature=0.2
            )

            # with open("/scratch/czr/shieldagent/.cache/log.txt", "w") as f:
            #     f.write(f"\n\nGrounding result: {grounding_result}\n")

            grounded_rules = extract_json(grounding_result)["grounded_rules"]

            # # Validate each rule
            invalid_rules = []
            for i, rule in enumerate(grounded_rules):
                is_valid, error_msg = validate_rule_consistency(rule)
                if not is_valid:
                    invalid_rules.append({
                        "rule_id": rule["source_rule_id"],
                        "error": error_msg
                    })
            
            if invalid_rules:
                raise Exception(f"Inconsistency found in grounded rules: {invalid_rules}")

            # label new rule_id for each grounded rule
            for i, rule in enumerate(grounded_rules):
                rule["grounded_rule_id"] = i

            
            return {
                "status": "success",
                "solution_space": grounded_rules
            }
        
        except Exception as e:
            # Continue to next retry attempt
            print(f"[ASPMTool] Error in formulate_solution_space (attempt {_+1}/3): {str(e)}")
            continue

    return {
        "status": "error",
        "message": "Failed to ground rules after multiple attempts"
    }

@aspm_mcp.tool()
async def reset_state() -> Dict[str, Any]:
    """Reset all variables (NOT AVAILABLE, ADMIN ONLY)."""
    global aspm_tree, aspm_actions, aspm_rules, aspm_condition_clusters
    aspm_tree = None
    aspm_actions = {}
    aspm_rules = {}
    aspm_condition_clusters = {}
    return "All ASPM state reset to initial state."


if __name__ == "__main__":
    anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Initialize with default ASPM data if environment variable is set
    try:
        default_aspm_path = os.environ.get("ASPM_PATH")
        if default_aspm_path:
            load_aspm_json(default_aspm_path)
            print("[ASPMTool] Initialized with default ASPM data")
    except Exception as e:
        print(f"[ASPMTool] Warning: Could not load default ASPM data: {str(e)}")
    
    aspm_mcp.run()