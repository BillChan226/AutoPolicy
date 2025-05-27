#!/usr/bin/env python3
"""
ShieldAgent - A guardrail framework for AI safety verification.

ShieldAgent implements a verification workflow for action rule circuits, integrating
specialized shielding operations and a tool library. It employs a hybrid memory module
for efficiency that caches short-term interaction history and stores long-term
successful shielding workflows.
"""

import os, sys
import asyncio
import json
import time
import subprocess
import sqlite3
from typing import List, Dict, Any, Optional, Set, Tuple
import hashlib
from datetime import datetime
sys.path.append("./")
from agents import Agent, Runner, gen_trace_id, trace, ModelSettings
from agents.mcp import MCPServerStdio
from dotenv import load_dotenv
from agents import set_default_openai_key
import base64
from shield.utils import *

# Load environment variables
load_dotenv()
set_default_openai_key(os.environ.get("OPENAI_API_KEY", ""))

class ShieldAgent:
    """
    ShieldAgent - An AI safety verification framework.
    
    ShieldAgent integrates specialized shielding operations and a tool library
    to verify AI actions against rule circuits to ensure safety.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        db_path: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize the ShieldAgent.
        
        Args:
            model: The LLM model to use for the agent
            db_path: Path to the memory database
            debug: Whether to print debug information
        """
        self.model = model
        self.debug = debug
        self.debug_log = []  # Store debug logs
        
        # Set database paths
        memory_db_path = db_path or os.path.join(os.getcwd(), "shield_memory.db")
        
        # Configure memory database
        self._initialize_memory_db(memory_db_path)
        
        # Initialize MCP servers with proper transport
        self._initialize_mcp_servers()
        
        # Initialize Shielding Planning Agent and Safety Certification Agent
        self.planning_agent = self._create_planning_agent()
        self.safety_certification_agent = self._create_safety_certification_agent()
        
        
        # Track current verification state
        self.current_trace_id = None
        self.current_action = None
        self.current_circuit = None
        self.verification_results = {}
        self.predicate_values = {}
        
        print(f"[ShieldAgent] Initialized with model: {model}")
    
    def _initialize_memory_db(self, db_path: str):
        """Initialize the memory database directly"""
        conn = sqlite3.connect(db_path)
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
        
        print(f"[ShieldAgent] Memory database initialized at {db_path}")
        
        # Set global state in memory tool
        self._run_python_command(f"""
import os, sys
sys.path.append(os.getcwd())
from shield.tools.memory_tool import db_path
db_path = "{db_path}"
""")
    
    def _run_python_command(self, code: str):
        """Run a Python command in a subprocess to update global variables in modules"""
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout:
                print(f"[ShieldAgent] Python command output: {result.stdout.strip()}")
            if result.stderr:
                print(f"[ShieldAgent] Warning during Python command: {result.stderr.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"[ShieldAgent] Error running Python command: {e.stderr if hasattr(e, 'stderr') else str(e)}")
            if hasattr(e, 'stdout') and e.stdout:
                print(f"[ShieldAgent] Command output: {e.stdout}")
    
    def _initialize_mcp_servers(self):
        """Initialize MCP servers for all tools using proper transport mechanisms"""
        
        # For ASPM tool, set environment variable for initialization
        if hasattr(self, 'aspm_path') and self.aspm_path:
            os.environ["ASPM_PATH"] = self.aspm_path
        
        self.mcp_servers = {}
        # Create MCP servers for each tool module
        self.mcp_servers["aspm"] = MCPServerStdio(
            params={
                "command": "python",
                "args": ["-m", "shield.tools.aspm_tool"],
                "env": os.environ.copy()  # Pass environment variables
            },
            cache_tools_list=True
        )
        
        self.mcp_servers["content_moderation"] = MCPServerStdio(
            params={
                "command": "python",
                "args": ["-m", "shield.tools.content_moderation"],
                "env": os.environ.copy()  # Pass environment variables
            }, 
            cache_tools_list=True
        )
        
        self.mcp_servers["verification"] = MCPServerStdio(
            params={
                "command": "python",
                "args": ["-m", "shield.tools.verification"],
                "env": os.environ.copy()  # Pass environment variables
            },
            cache_tools_list=True
        )
        
        self.mcp_servers["memory"] = MCPServerStdio(
            params={
                "command": "python",
                "args": ["-m", "shield.tools.memory_tool"],
                "env": os.environ.copy()  # Pass environment variables
            },
            cache_tools_list=True
        )
        
        self.mcp_servers["certification"] = MCPServerStdio(
            params={
                "command": "python",
                "args": ["-m", "shield.tools.certification"],
                "env": os.environ.copy()  # Pass environment variables
            },
            cache_tools_list=True
        )
        
        # For convenience, add direct access to the servers as tool objects
        self.memory_tool = self.mcp_servers["memory"]
        self.verification_tool = self.mcp_servers["verification"]
        self.aspm_tool = self.mcp_servers["aspm"]
        self.content_moderation_tool = self.mcp_servers["content_moderation"]
        self.certification_tool = self.mcp_servers["certification"]
    
    def _create_planning_agent(self) -> Agent:
        """Create a specialized agent for planning verification workflows."""
        instructions = """# Planning Agent for ShieldAgent - Predicate Verification

You are the Planning Agent responsible for systematically verifying predicates in safety rules and assigning their truth values. Your goal is to assign truth values to ALL predicates through appropriate verification methods until no unverified predicates remain.

## Verification Tools

1. Direct Context Inference - Predicate that can be obviously and directly determined from the agent's thought process and action
   - Example: If predicate is "submit_comment" and agent's action is clearly submitting a comment

2. Tool-Based Verification - Predicate that are less obvious and need to be verified using tools
   - Content Moderation: For checking content safety, toxicity, appropriateness
   - Search: For retrieving relevant information from history
   - Binary Check: For making clear yes/no decisions
   - Detect: For specific pattern or content detection

## Available Predicate Tracking Tools

- `get_unverified_predicates`: Get all predicates that have not been verified yet from the predicate map including their descriptions and predicate types
- `update_predicate_values`: Update the truth values of the predicates that have been verified into the predicate map
- `get_predicate_info`: Get the value, description, and predicate type for a list of predicates (you don't need to use this tool if you just called get_unverified_predicates)

## Your Workflow

- You should loop through the following steps until all predicates are verified:
    1. Use get_unverified_predicates to get all predicates needing verification
    2. Analyze the most appropriate verification method for each predicate
    3. You should first assign the predicates whose truth values can be confidently obtained via direct context inference
    4. Then, you should use the appropriate verification tools to verify the remaining predicates
    5. You can use the update_predicate_values tool to push the truth values of the predicates that you have verified into the predicate map after each iteration
    6. You should check if all predicates have successfully been assigned truth values. If not, you should go back to step 1 and verify the remaining predicates
    7. If all predicates have successfully been assigned truth values, you should terminate the workflow

Remember:
- You should only use direct context inference to verify predicates that are very obvious and can be determined from the agent's thought process and action
- You may need to first try several specialized tools to assign a predicate, and if none of the tools work, you can try more generic tool such as binary check or search.
- You should first get all unverified predicates at the beginning of each iteration, and at the end of each iteration, you should update the predicate map with the verified predicates during this turn
- You should make sure all the predicates are assigned even if you are unsure about the truth value, you should assign the predicate with the best guess of the truth value

Verification Guidelines for Different Predicate:
- Default-based predicates: predicates such as "agreed_privacy_policy" and "agreed_terms_of_service" can always be assumed to be true if the user has not explicitly denied them
- Fact-based predicates: predicates that are based on factual information, such as "data_accurate" or "consistent_with_source", should always be verified explicitly using either factual check or binary query tools
- Instruction-related predicates: predicates such as "exact_user_requested" should always align with the initial task intent and not be disturbed by other agent context
"""
        
        return Agent(
            name="Shielding Planning Agent",
            instructions=instructions,
            model=self.model,
            # Ensure all necessary tools are available
            mcp_servers=[
                self.verification_tool,      # Provides predicate map tools
                self.content_moderation_tool # Provides moderation & factual check tools
                # Add other tools like memory/search if needed for specific predicates
            ]
        )
    
    def _create_safety_certification_agent(self) -> Agent:
        """Create a specialized agent for rule verification and safety certification."""
        instructions = """# Safety Certification Agent for ShieldAgent

You are a helpful assistant that verifies all grounded rules based on the assigned predicate truth values and determines the overall safety outcome.

## Available Tools
- `get_unverified_rules()`: Returns all the rules that have not been verified yet.
- `verify_llm_rules(List[rule_id_int])`: Takes a list of rule IDs and verify each rule using an LLM as a judge.
- `verify_fol_rules(List[rule_id_int])`: Takes a list of rule IDs and verify each rule using static first-order logic.
- `static_certification(Dict[rule_id_int: violation_bool])`: Takes a dictionary of rule IDs and their violation assignments and returns a dictionary with the overall certification outcome.

## Your Workflow
1. **Get Rules:** Call `get_unverified_rules` to get all the rules to verify (you may repeat this tool until all rules are verified).
2. **Verify All Rules:** You should first try to verify all the rules using first-order logic as it is much faster. If that fails for some rules, you can then verify the remaining rules using verify_llm_rules.
3. **Safety Certification:** Run safety certification algorithm after all rules are verified to obtain the final guardrail decision.
4. **Guardrail Decision Report:** Produce a final JSON output summarizing the guardrail decision and no other text:
```json
{
    "action_allowed": true/false,
    "violated_rules": [
    {
        "rule_id": "...",
        "explanation": "explanation of why the rule was violated"
        "feedback": "remediation feedback to fix the rule violation"
    }
    // ... list all rules ...
    ],
    "explanation": "A concise report of the guardrail decision and how to remediate the violations"
}
```
Remember:
- You should call safety certification tool after all rules are verified, where you must provide an input of a dict of rule_id: bool where True means violated, False means not violated.
- After getting the certification result, you should terminate the workflow and report the final guardrail decision in the JSON format above.
- Try to send as many rules as possible to the verifier tools as it is much faster for them to run in batch. 
- Both 'violated' and 'not violated' are valid rule stats, and you do not need to call `verify_llm_rules` if you have already have their `verify_fol_rules` results.
"""

        return Agent(
            name="Safety Certification Agent",
            instructions=instructions,
            model=self.model,
            mcp_servers=[self.certification_tool] # This agent uses the certification tools
        )
    
    async def start_mcp_servers(self):
        """Start all MCP servers."""
        # Connect to each MCP server - now they have proper connect methods
        for name, server in self.mcp_servers.items():
            try:
                await server.connect()
                print(f"[ShieldAgent] Connected to {name} MCP server")
            except Exception as e:
                print(f"[ShieldAgent] Failed to connect to {name} MCP server: {str(e)}")
        
        print("[ShieldAgent] All MCP servers ready to use")
    
    async def stop_mcp_servers(self):
        """Stop all MCP servers."""
        # Close connections to each MCP server
        for name, server in self.mcp_servers.items():
            try:
                # Simply ensure the server is released, ignoring errors
                print(f"[ShieldAgent] Releasing {name} MCP server")
                # Don't actually call any methods that might throw, just let Python's GC handle it
            except Exception as e:
                print(f"[ShieldAgent] Warning: {name} MCP server cleanup error: {str(e)}")
        
        # Sleep briefly to allow asyncio to clean up resources
        await asyncio.sleep(0.5)
        print("[ShieldAgent] All MCP servers released")
    
    async def verify_action(self, action_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Verify an action against safety rules."""
        self.current_trace_id = gen_trace_id()
        self.current_action = action_text
        context = context or {}
        
        # Reset debug state for this verification session
        self.debug_log = []
        
        self.log(f"Starting verification for action: {action_text[:100]}...")
        
        # Store all relevant context data in the short-term memory module
        try:
            # Extract only the context items we want to store
            context_to_store = {
                "history": context.get("history", []),
                "thought_text": context.get("thought_text", ""),
                "action_text": action_text,
                "task_intent": context.get("task_intent", ""),
                "context": context  # Store the full context as well
            }
            
            # result = await self.memory_tool.call_tool(
            #     "store_interaction_context",
            #     {
            #         "trace_id": self.current_trace_id,
            #         "context_data": context_to_store,
            #         "ttl": 3600  # 1 hour TTL
            #     }
            # )
            
        except Exception as e:
            print(f"[ShieldAgent] Error storing context: {str(e)}")
        
        # Build the verification query using the action text and context
        task_query = self._build_task_query(context)
        
        try:
            if True:
                # 1. Action Grounding
                self.log("Step 1: Action Grounding")
                action_grounding_result = await self.aspm_tool.call_tool(
                    "action_grounding",
                    {"agent_thought": context["thought_text"], "agent_action": context["action_text"]}
                )
                print(f"[ShieldAgent] Action grounding result: {action_grounding_result}")  
                action_grounding_result = json.loads(action_grounding_result.content[0].text)["matched_predicates"]
                grounded_action_predicates = set()
                for predicate, predicate_info in action_grounding_result.items():
                    grounded_action_predicates.add(predicate)
                    upstream_predicates = predicate_info.get("upstream_predicates", {})
                    for upstream_predicate, upstream_predicate_list in upstream_predicates.items():
                        grounded_action_predicates.update(upstream_predicate_list)
                grounded_action_predicates = list(grounded_action_predicates)
                print(f"[ShieldAgent] Grounded action predicates: {grounded_action_predicates}")

                # 2. Rule Retrieval
                self.log("Step 2: Rule Retrieval")
                action_circuits_result = await self.aspm_tool.call_tool(
                    "get_action_circuit", {"action_names": grounded_action_predicates}
                )
                # TODO: Add error handling
                action_circuits = json.loads(action_circuits_result.content[0].text)["action_circuits"]
                self.log(f"Retrieved action circuits: {len(action_circuits)} actions")

                # 3. Solution Space Formulation
                self.log("Step 3: Solution Space Formulation")

                solution_space_result = await self.aspm_tool.call_tool(
                    "formulate_solution_space",
                    {
                    "agent_thought": context["thought_text"], 
                    "agent_action": context["action_text"], 
                    "action_predicates": grounded_action_predicates, 
                    "action_circuits": action_circuits, 
                    "reasoning": False
                    }
                )
                self.log(f"[ShieldAgent] Solution space result: {solution_space_result}")
                grounded_rules = json.loads(solution_space_result.content[0].text)["solution_space"]
        
                self.log(f"Formulated {len(grounded_rules)} grounded rules.")

                # 4. Predicate Map Initialization (in Verification Tool)
                self.log("Step 4: Initialize Predicate Map")
                init_map_result = await self.verification_tool.call_tool(
                    "create_predicate_map", {"grounded_rules": grounded_rules}
                )
                self.log(f"Predicate map init result: {init_map_result}", "HIDE")

                # 5. Predicate Value Assignment (Planning Agent)
                self.log("Step 5: Run Planning Agent for Predicate Assignment")
                planning_result = await Runner.run(
                    self.planning_agent,
                    task_query,
                    max_turns=30
                )
                self.log(f"Planning agent final output: {planning_result.final_output}", "REPORT")
                # Planning agent should loop until all predicates are assigned or it stops

                # --- Sanity Check ---
                self.log("Performing sanity check on predicate assignments...")
                final_map_result = await self.verification_tool.call_tool("get_predicate_map", {})
                final_predicate_map = json.loads(final_map_result.content[0].text)["predicate_map"]
                # print(f"[ShieldAgent] Final predicate map: {final_predicate_map}")
                all_rule_predicates = set()
                for rule in grounded_rules:
                    for pred_info in rule.get("grounded_predicates", []):
                        all_rule_predicates.add(pred_info[0])

                missing_preds = []
                for pred_name in all_rule_predicates:
                    if pred_name not in final_predicate_map or final_predicate_map[pred_name]['value'] is None:
                        missing_preds.append(pred_name)

                if missing_preds:
                    warning_msg = f"Warning: Sanity check failed! The following predicates needed for rule verification were not assigned: {', '.join(missing_preds)}"
                    self.log(warning_msg, "WARN")
                    raise Exception(warning_msg)

            #     with open("/scratch/czr/shieldagent/.cache/grounded_rules.json", "w") as f:
            #         json.dump(grounded_rules, f, indent=4)
            #     with open("/scratch/czr/shieldagent/.cache/final_predicate_map.json", "w") as f:
            #         json.dump(final_predicate_map, f, indent=4)
            #     input("Press Enter to continue...")
            
            # with open("/scratch/czr/shieldagent/.cache/grounded_rules.json", "r") as f:
            #     grounded_rules = json.load(f)
            # with open("/scratch/czr/shieldagent/.cache/final_predicate_map.json", "r") as f:
            #     final_predicate_map = json.load(f)

            # 6. Initialize Certification State
            self.log("Step 6: Initialize Certification State")
            init_cert_result = await self.certification_tool.call_tool(
                "initialize_verification_state",
                {"grounded_rules": grounded_rules, "predicate_map_state": final_predicate_map}
            )
            self.log(f"Certification init result: {init_cert_result}")
            if "Error:" in init_cert_result: # Simple check for error string
                 raise Exception(f"Certification initialization failed: {init_cert_result}")

            # 7. Rule Verification & Safety Certification
            self.log("Step 7: Run Safety Certification Agent")
            certification_result = await Runner.run(
                self.safety_certification_agent,
                """Please verify all rules and determine the final guardrail decision.
Remember:
- You should call safety certification tool after all rules are verified, where you must provide an input of a dict of rule_id: bool where True means violated, False means not violated.
- After getting the certification result, you should terminate the workflow and report the final guardrail decision in the JSON format above.
""",
                max_turns=20
            )
            self.log(f"Certification agent final output: {certification_result.final_output}", "REPORT")

            # 8. Process Final Result
            self.log("Step 8: Processing Final Verification Result")
            final_output_data = {}
            # Attempt to parse JSON from the final output
            try:
                 # Use a robust JSON extraction method if needed
                 json_start = certification_result.final_output.find('{')
                 json_end = certification_result.final_output.rfind('}')
                 if json_start != -1 and json_end != -1:
                      final_output_data = json.loads(certification_result.final_output[json_start:json_end+1])
                 else:
                      raise ValueError("No JSON object found in final output")
            except Exception as parse_error:
                 self.log(f"Failed to parse final JSON result: {parse_error}", "ERROR")
                 # Fallback if parsing fails
                 final_output_data = {
                     "allowed": False,
                     "recommendation": "DENY",
                     "explanation": f"Error processing certification output. Raw output: {certification_result.final_output}",
                     "rules_verified": [],
                     "predicate_values": final_predicate_map # Include the map we have
                 }

            # Reset all tool states
            # await self.aspm_tool.call_tool("reset_state", {})
            await self.verification_tool.call_tool("reset_state", {})
            await self.certification_tool.call_tool("reset_state", {})
            self.log("All tool states reset to initial state.")


            return final_output_data

        except Exception as e:
            # ... [existing exception handling - seems okay] ...
            error_msg = f"Error during verification: {str(e)}"
            self.log(error_msg, "ERROR")
            import traceback
            trace_str = traceback.format_exc()
            self.log(f"Traceback: {trace_str}", "ERROR")
            
            return {
                "allowed": False,
                "action_name": "error",
                "explanation": error_msg,
                "trace_id": self.current_trace_id,
                "debug_info": {
                    "debug_log": self.debug_log,
                    "traceback": trace_str
                } if self.debug else {}
            }
    
    
    def _build_task_query(self, context: Dict[str, Any]) -> str:
        """Build the Verification Query for the coordinator agent in OpenAI conversation format."""
        task_intent = context.get("task_intent", "")
        history = context.get("history", [])
        action_text = context.get("action_text", "")
        thought_text = context.get("thought_text", "")
        
        # Build a single message in OpenAI conversation format
        messages = [{
            "role": "user",
            "content": []
        }]
        
        # Add task intent
        messages[0]["content"].append({
            "type": "input_text",
            "text": f"Task Intent: {task_intent}\n"
        })
        
        # Add conversation history
        if history:
            history_text = "\n\n## INTERACTION HISTORY:\n"

            messages[0]["content"].append({
                "type": "input_text",
                "text": history_text
            })
            
            for i, entry in enumerate(history):
                role = entry.get("role", "")
                if role == "agent":
                    history_text = f"### AGENT ACTION:\n"
                    history_text += f"Time step: {i+1}\n"
                    if entry.get('thought'):
                        history_text += f"Agent thought:\n{entry.get('thought', '')}\n\n"
                    history_text += f"Agent action:\n{entry.get('action', '')}\n\n"

                    messages[0]["content"].append({
                        "type": "input_text",
                        "text": history_text
                    })
                    
                elif role == "environment":
                    history_text = f"### ENVIRONMENT OBSERVATION:\n"
                    history_text += f"Time step: {i+1}\n"
                    history_text += f"Environment message:\n{entry.get('obs_text', '')}\n\n"

                    messages[0]["content"].append({
                        "type": "input_text",
                        "text": history_text
                    })
                    
                    # Handle images by encoding them as base64
                    if entry.get('obs_visual'):
                        for visual_element in entry.get('obs_visual', []):
                            # try:
                            # Use the separate function to encode the image
                            if visual_element["type"] == "image":
                                assert visual_element["path"].endswith(".jpg") or visual_element["path"].endswith(".jpeg") or visual_element["path"].endswith(".png"), f"Image file must be a jpg, jpeg, or png file. Current file is {visual_element['path']}"
                                # base64_image = self.encode_image(visual_element["path"])
                                base64_image, mime_type  = encode_image(visual_element["path"])
                            else:
                                assert False, f"Unsupported visual element type: {visual_element['type']}. Current only supports image."
                            
                            element_function = visual_element["element_function"]
                            # Add image to content
                            messages[0]["content"].extend([
                                {
                                    "type": "input_text",
                                    "text": f"Visual observation:\n{element_function}\n"
                                },
                                {
                                    "type": "input_image",
                                    "image_url": f"data:{mime_type};base64,{base64_image}"
                                }
                            ])
            
        # Add current action to verify
        target_action_text = f"\n\n## CURRENT ACTION TO VERIFY:\n"
        target_action_text += f"Time step: {len(history)+1}\n"
        if thought_text:
            target_action_text += f"Agent thought:\n{thought_text}\n\n"
        target_action_text += f"Agent action:\n{action_text}\n\n"
        
        # Add additional context
        # additional_context = {k: v for k, v in context.items() if k not in ["task_intent", "history", "action_text", "thought_text"]}
        # if additional_context:
        #     verification_text += "## ADDITIONAL CONTEXT:\n"
        #     verification_text += json.dumps(additional_context, indent=2)
        
        messages[0]["content"].append({
            "type": "input_text",
            "text": target_action_text
        })

        messages[0]["content"].append({
            "type": "input_text",
            "text": "Please start your verification workflow and follow the instructions to assign truth values to predicates step-by-step."
        })
        
        return messages
    
    async def _process_verification_result(self, agent_output: str) -> Dict[str, Any]:
        """Process the agent's verification output and extract structured results."""
        result = {
            "allowed": False,
            "action_name": "unknown",
            "action_predicates": [],
            "predicate_values": {},
            "verification_steps": [],
            "rules_verified": [],
            "safety_score": 0.0,
            "recommendation": "DENY",
            "explanation": "Could not parse verification result",
            "trace_id": self.current_trace_id
        }
        
        # Try to extract JSON from the agent's response
        try:
            # Look for JSON blocks in the output
            json_blocks = []
            in_json_block = False
            current_block = []
            
            for line in agent_output.split('\n'):
                if line.strip() == '```json' or line.strip() == '```':
                    if in_json_block:
                        json_blocks.append('\n'.join(current_block))
                        current_block = []
                    in_json_block = not in_json_block
                elif in_json_block:
                    current_block.append(line)
            
            # Also check for the last block if we're still in it
            if in_json_block and current_block:
                json_blocks.append('\n'.join(current_block))
            
            # If no JSON blocks found, try parsing the whole output
            if not json_blocks:
                parsed_result = json.loads(agent_output)
                result.update(parsed_result)
            else:
                # Try each JSON block until we find a valid one
                for block in json_blocks:
                    try:
                        parsed_result = json.loads(block)
                        if isinstance(parsed_result, dict) and "action_name" in parsed_result:
                            result.update(parsed_result)
                            break
                    except:
                        continue
            
            # Ensure boolean type for allowed
            result["allowed"] = bool(result.get("allowed", False))
            
            # Store predicate values for future use
            self.predicate_values = result.get("predicate_values", {})
            
        except Exception as e:
            if self.debug:
                print(f"[ShieldAgent] Error parsing verification result: {str(e)}")
                print(f"[ShieldAgent] Agent output: {agent_output}")
            
            result["explanation"] = f"Error parsing verification result: {str(e)}"
        
        return result

    def log(self, message: str, level: str = "INFO"):
        """Log a debug message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        if self.debug and level != "HIDE":
            # use green color for debug logs
            if level == "WARN":
                print(f"\033[93m{log_entry}\033[0m")
            elif level == "ERROR":
                print(f"\033[91m{log_entry}\033[0m")
            elif level == "REPORT":
                print(f"\033[92m{log_entry}\033[0m")
            else:
                print(f"\033[92m{log_entry}\033[0m")
        
        self.debug_log.append(log_entry)
        
        # Optionally write to a file for persistent logs
        if self.debug and hasattr(self, 'log_file'):
            with open(self.log_file, 'a') as f:
                f.write(log_entry + "\n")



def extract_tool_result(result):
    """Extract values from either a dictionary or a CallToolResult object."""
    if hasattr(result, 'status') and callable(getattr(result, 'get', None)):
        # It's likely a CallToolResult that also has dict-like methods
        return result
    elif hasattr(result, 'status'):
        # It's a CallToolResult object without dict methods
        return {k: getattr(result, k) for k in dir(result) if not k.startswith('_') and not callable(getattr(result, k))}
    else:
        # Assume it's a dictionary
        return result