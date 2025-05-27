#!/usr/bin/env python3
"""
Script to run ShieldAgent verification on an agent trajectory.
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List

sys.path.append("./")
from agent import ShieldAgent
from debug_utils import export_debug_info


async def verify_trajectory(trajectory_dir: str, output_dir: Optional[str] = None, debug: bool = False) -> None:
    """Verify an agent trajectory against safety policies."""
    print(f"Starting verification for trajectory: {trajectory_dir}")
    
    if not os.path.exists(trajectory_dir):
        print(f"Error: Trajectory directory not found at {trajectory_dir}")
        return
    
    trajectory_file = os.path.join(trajectory_dir, "agent_traj.json")

    if not os.path.exists(trajectory_file):
        print(f"Error: Trajectory file not found at {trajectory_file}")
        return
    

    if output_dir is None:
        output_dir = os.path.dirname(trajectory_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the trajectory file directly
    try:
        with open(trajectory_file, 'r', encoding='utf-8') as f:
            trajectory_data = json.load(f)
        
        # Extract relevant information from the trajectory file
        task_intent = trajectory_data.get('task_intent', 'No task intent provided')
        policies = trajectory_data.get('policies', [])
        conversation = trajectory_data.get('conversation', [])
        
        # Extract agent steps from conversation (filter by role="agent")
        agent_steps = [step for step in conversation if step.get('role') == 'agent']

        print(f"Successfully loaded trajectory with {len(agent_steps)} agent steps")
        print(f"Task Intent: {task_intent}")
        print(f"Loaded {len(policies)} safety policies")
        
        # Create ShieldAgent with debug flag
        shield_agent = ShieldAgent(
            model="gpt-4o-2024-05-13",
            # model="gpt-4o-mini",
            # model="claude",
            debug=debug
        )
        
        if debug:
            # Set up debug log file
            shield_agent.log_file = os.path.join(output_dir, "shield_debug.log")
            print(f"Debug logs will be written to {shield_agent.log_file}")
        
        # Start MCP servers for the verification process
        await shield_agent.start_mcp_servers()
        
        try:
            # Process each agent step
            verification_results = []
            
            for step_idx, agent_step in enumerate(agent_steps):
                if step_idx < len(agent_steps) - 1:
                    continue
                # if step_idx >0:
                #     break
                print(f"\n=== Processing Agent Step {step_idx+1}/{len(agent_steps)} ===")
                
                # Extract action and thought texts from the agent step
                action_text = None
                thought_text = None
                
                for message in agent_step.get("messages", []):
                    if "action" in message:
                        action_text = message["action"]
                    if "thought" in message:
                        thought_text = message["thought"]
                
                if not action_text:
                    print(f"Warning: No action found in agent step {step_idx+1}")
                    continue
                
                print(f"Analyzing action: {action_text[:100]}...")
                
                # Build up the conversation history up to this point
                history = build_conversation_history(conversation, step_idx, trajectory_dir)
                print(f"History: {history}")

                # Create verification context with all necessary information
                current_context = {
                    "task_intent": task_intent,
                    "step_index": step_idx,
                    "action_text": action_text,
                    "thought_text": thought_text,
                    "history": history,
                }
                
                # Simply call verify_action on the ShieldAgent
                verification_result = await shield_agent.verify_action(
                    action_text=action_text,
                    context=current_context
                )
                
                # Store the verification result
                step_verification = {
                    "step_idx": step_idx,
                    "action_text": action_text,
                    "thought_text": thought_text,
                    "allowed": verification_result.get("allowed", False),
                    "explanation": verification_result.get("explanation", ""),
                    "verification_details": verification_result
                }
                
                verification_results.append(step_verification)
                
                print(f"Verification complete for step {step_idx+1}")
                print(f"Allowed: {verification_result.get('allowed', False)}")
                print(f"Explanation: {verification_result.get('explanation', '')[:200]}...")
                
                # After getting verification result
                if debug and "debug_info" in verification_result:
                    # Export debug information
                    export_debug_info(verification_result, output_dir)
                    print(f"Debug information exported to {output_dir}")
            
            # Save verification results
            output_file = os.path.join(output_dir, "verification_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "task_intent": task_intent,
                    "total_steps": len(agent_steps),
                    "policies": policies,
                    "verification_results": verification_results
                }, f, indent=2)
            
            print(f"\nVerification complete. Results saved to {output_file}")
            
        finally:
            # Properly clean up MCP servers - improved approach
            try:
                print("Shutting down MCP servers...")
                await shield_agent.stop_mcp_servers()
                # Give a moment for cleanup to avoid race conditions
                await asyncio.sleep(1) 
            except Exception as e:
                print(f"Warning during MCP server shutdown: {str(e)}")
            
    except Exception as e:
        print(f"Error during trajectory verification: {str(e)}")
        import traceback
        traceback.print_exc()


def build_conversation_history(conversation, current_step_idx, trajectory_dir):
    """
    Build a conversation history up to the current step.
    This includes both agent actions and environmental observations.
    """
    history = []
    
    # Count the number of agent steps we've seen
    agent_count = 0
    target_agent_idx = None
    
    # First pass: Find which conversation index corresponds to our target agent step
    for i, step in enumerate(conversation):
        if step.get('role') == 'agent':
            if agent_count == current_step_idx:
                target_agent_idx = i
                break
            agent_count += 1
    
    if target_agent_idx is None:
        return history  # Invalid index
    

    # Second pass: Include all conversation elements before the target agent step
    for i in range(target_agent_idx):
        step = conversation[i]
        role = step.get('role')
        
        if role == 'agent':
            # Extract action and thought
            action = None
            thought = None
            
            assert len(step.get('messages', [])) == 1, "Agent step should have exactly one message"
            message = step.get('messages', [])[0]
            # if 'action' in message:
            action = message['action']
            # if 'thought' in message:
            thought = message['thought']
            
            # if action or thought:
            history.append({
                'role': 'agent',
                'action': action,
                'thought': thought
            })
        
        elif role == 'environment':
            
            assert len(step.get('messages', [])) == 1, "Environment step should have exactly one message"
            message = step.get('messages', [])[0]
            # if 'textual' in message:
            text_observation = message['textual']
            visual_observation = message['visual']

            for visual_element in visual_observation:
                visual_element["path"] = os.path.join(trajectory_dir, visual_element["path"])

            history.append({
                'role': 'environment',
                # 'obs_text': text_observation,
                'obs_visual': visual_observation
            })
    
    return history

async def main():
    """Parse arguments and run verification."""
    parser = argparse.ArgumentParser(description="Run ShieldAgent verification on an agent trajectory")
    
    # Accept either a file or a folder path
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--traj_dir", "-t", help="Path to directory containing agent_traj.json file")    
    # parser.add_argument("--aspm", "-p", required=True, help="Path to ASPM JSON file")
    parser.add_argument("--output", "-o", help="Directory to save verification results (defaults to trajectory file directory)")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    trajectory_dir = args.traj_dir
    # Determine trajectory file path
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = trajectory_dir
    
    try:
        await verify_trajectory(trajectory_dir, output_dir, args.debug)
    except Exception as e:
        print(f"Error in verification process: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure all pending tasks are properly cleaned up
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        # Wait for all tasks to complete with a timeout
        if tasks:
            try:
                await asyncio.wait(tasks, timeout=2)
                print(f"Cleaned up {len(tasks)} remaining tasks")
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"Error during task cleanup: {str(e)}")

if __name__ == "__main__":
    # Use a more controlled event loop shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        # Properly close the event loop
        try:
            # Cancel any remaining tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            # Give tasks a chance to respond to cancellation
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
            # Finally close the loop
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        except Exception as e:
            print(f"Error during event loop shutdown: {str(e)}") 