#!/usr/bin/env python3
"""
Policy Extraction Agent - A systematic document processing system for extracting 
structured policies from documents using a search tree approach.

This system focuses on extracting policies from PDF or HTML documents
with support for deep policy exploration by traversing document sections or following links.
"""

import os
import sys
sys.path.append("./")
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import heapq
from collections import deque

# Import MCP server for tool access
from fastmcp import FastMCP
from agents.mcp import MCPServerStdio
from dotenv import load_dotenv
from shield.utils import *

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_document_tree(document_tree: Dict[str, Any], output_path: str = None):
    """
    Plot the document tree using matplotlib and networkx.
    
    Args:
        document_tree: The document tree structure to visualize
        output_path: Path to save the visualization image
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_agraph import graphviz_layout
        
        # Create a networkx graph from the document tree
        G = nx.DiGraph()
        
        def add_nodes_edges(node, parent=None):
            node_id = f"{node['name']}:{node.get('type', 'unknown')}"
            policy_count = node.get('policies_count', 0)
            label = f"{node['name']}\n({policy_count} policies)"
            G.add_node(node_id, label=label, policies=policy_count)
            
            if parent:
                G.add_edge(parent, node_id)
            
            for child in node.get('children', []):
                add_nodes_edges(child, node_id)
        
        # Add all nodes and edges
        add_nodes_edges(document_tree)
        
        # Create figure
        plt.figure(figsize=(16, 10))
        
        # Create a hierarchical layout
        pos = graphviz_layout(G, prog="dot")
        
        # Get node sizes based on policy count (minimum size for visibility)
        node_sizes = [max(300, G.nodes[n]['policies'] * 100) for n in G.nodes()]
        
        # Draw the graph
        nx.draw(
            G, pos, 
            with_labels=True,
            node_size=node_sizes,
            node_color='skyblue',
            font_size=8,
            arrows=True,
            labels={n: G.nodes[n]['label'] for n in G.nodes()}
        )
        
        # Save the visualization
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            return True
        else:
            plt.show()
            plt.close()
            return True
            
    except ImportError as e:
        logging.warning(f"Could not create visualization: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Error creating visualization: {str(e)}")
        return False


class PriorityQueueItem:
    """Class for items in the priority queue with custom comparison."""
    
    def __init__(self, priority: int, timestamp: float, section: Dict[str, Any]):
        self.priority = priority
        self.timestamp = timestamp  # Used as a tiebreaker for items with the same priority
        self.section = section
        
    def __lt__(self, other):
        # Higher priority numbers come first, then earlier timestamps
        if self.priority == other.priority:
            return self.timestamp < other.timestamp
        return self.priority > other.priority

class PolicyExtractionAgent:
    """
    Systematic extraction agent framework for safety policies from documents.
    Features:
    - Uses a search tree approach to systematically explore document sections
    - Prioritizes sections based on likelihood of containing policies
    - Implements deduplication to avoid processing the same content multiple times
    - Provides detailed logging and status tracking
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        organization: str = "organization",
        user_request: str = "",
        debug: bool = False,
        model: str = "claude-3-7-sonnet-20250219"
    ):
        """
        Initialize the PolicyExtractionAgent.
        
        Args:
            output_dir: Directory for storing output files
            organization: Name of the organization
            user_request: User request for the policy extraction
            debug: Whether to print debug information
            model: The LLM model to use for analysis
        """
        self.debug = debug
        self.debug_log = []
        self.user_request = user_request
        self.organization = organization
        self.model = model
        
        # Set up output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define the central path for storing all extracted policies
        self.policies_path = os.path.join(self.output_dir, f"{organization}_all_extracted_policies.json")
        
        # Initialize MCP servers
        self._initialize_mcp_servers()
        
        # Initialize clients for OpenAI and Anthropic
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        # Initialize exploration queue and tracking sets
        self.exploration_queue = []  # Priority queue
        self.visited_sections = set()  # Track visited sections to avoid duplicates
        self.current_extraction_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"PolicyExtractionAgent initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Central policies file: {self.policies_path}")
    
    def _initialize_mcp_servers(self):
        """Initialize MCP servers for policy extraction tools."""
        # Policy extraction MCP server with environment variables for the central policy path
        env_vars = os.environ.copy()
        env_vars["POLICY_PATH"] = self.policies_path
        
        self.policy_extraction_tool = MCPServerStdio(
            params={
                "command": "python",
                "args": ["-m", "shield.tools.policy_extraction"],
                "env": env_vars
            },
            cache_tools_list=True
        )
    
    async def start_mcp_servers(self):
        """Start all MCP servers."""
        try:
            await self.policy_extraction_tool.connect()
            logger.info(f"Connected to policy extraction MCP server")
        except Exception as e:
            logger.error(f"Failed to connect to policy extraction MCP server: {str(e)}")
    
    async def stop_mcp_servers(self):
        """Stop all MCP servers."""
        try:
            logger.info("Preparing to release MCP servers")
            # Sleep briefly to allow asyncio to clean up resources
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error during MCP server shutdown preparation: {str(e)}")
    
    async def call_tool(self, tool_name: str, **kwargs) -> str:
        """
        Call a tool from the policy extraction MCP server.
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Tool response as a string or dictionary
        """
        try:
            
            # Call the tool using the correct MCP interface
            response = await self.policy_extraction_tool.call_tool(
                tool_name,
                kwargs
            )
            
            # Extract and process the response
            if hasattr(response, 'content') and len(response.content) > 0:
                # For Anthropic-style responses
                result = response.content[0].text

                if isinstance(result, str):
                    try:
                        return json.loads(result)
                    except json.JSONDecodeError:
                        return {"success": False, "error": "Invalid JSON response"}
                else:
                    return result
            else:
                # For direct string/dict responses
                return response
                
        except Exception as e:
            error_msg = f"Error calling tool {tool_name}: {str(e)}"
            self.log(error_msg, "ERROR")
            return {"success": False, "error": error_msg}
    
    def add_sections_to_queue(self, sections: List[Dict[str, Any]]) -> int:
        """
        Add sections to the exploration queue with deduplication.
        
        Args:
            sections: List of section dictionaries with format:
            {
                "section_name": "Name of subsection",
                "path": "Path to PDF file or URL for HTML",
                "type": "pdf" or "html",
                "range": "Page range (e.g., '5-10') - ONLY for PDF type",
                "priority": 1-10 rating of importance (10 being highest),
                "source": "Where this subsection was found"
            }
            
        Returns:
            Number of sections added to the queue
        """
        added_count = 0
        
        for section in sections:
            # Validate required fields
            if not all(k in section for k in ['section_name', 'path', 'type']):
                self.log(f"Skipping section with missing required fields: {section}", "WARN")
                continue
                
            if section['type'] not in ['pdf', 'html']:
                self.log(f"Skipping section with invalid type '{section['type']}': {section['section_name']}", "WARN")
                continue
                
            # Create a unique identifier for this section
            if section['type'] == 'pdf':
                section_id = f"pdf:{section['path']}:{section.get('range', '-1')}"
            else:
                section_id = f"html:{section['path']}:-1"
                
            # Skip if already visited or already in queue
            if section_id in self.visited_sections:
                self.log(f"Skipping duplicate section: {section['section_name']}", "INFO")
                continue
            
            # Mark as queued
            self.visited_sections.add(section_id)

            
            # Set default priority if not provided
            if 'priority' not in section:
                section['priority'] = 5
                
            # Add to priority queue
            heapq.heappush(
                self.exploration_queue, 
                PriorityQueueItem(
                    section['priority'], 
                    datetime.now().timestamp(),
                    section
                )
            )
            added_count += 1
            
        self.log(f"Added {added_count} new sections to exploration queue")
        return added_count
    
    def get_next_section(self) -> Optional[Dict[str, Any]]:
        """
        Get the next section from the exploration queue.
        
        Returns:
            Next section dictionary or None if queue is empty
        """
        if not self.exploration_queue:
            return None
            
        # Get the highest priority item
        item = heapq.heappop(self.exploration_queue)
        return item.section
    
    async def extract_policies(
        self, 
        document_path: str, 
        input_type: str = "pdf",
        deep_policy: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Extract policies from a document using the search tree approach.
        
        Args:
            document_path: Path to PDF file or URL for HTML
            input_type: "pdf" or "html"
            deep_policy: Whether to explore linked pages/sections
            
        Returns:
            List of extracted policies
        """
        # Create extraction directory for this run
        extraction_dir = os.path.join(self.output_dir, f"extraction_{self.current_extraction_id}")
        os.makedirs(extraction_dir, exist_ok=True)
        self.log(f"Created extraction directory: {extraction_dir}")
        
        # Reset exploration state
        self.exploration_queue = []
        self.sections_found = []
        self.visited_sections = set()
        
        # Initialize section map to track all processed sections
        # Format: {section_id: {data: section_data, parent_id: parent_section_id, has_policies: bool}}
        section_map = {}
        
        # Add the initial document as the first section to explore
        initial_section = {
            "section_name": "Initial Document",
            "path": document_path,
            "range": "-1" if input_type == "pdf" else "",
            "type": input_type,
            "priority": 10,  # Highest priority
            "source": "User Input"
        }
        self.sections_found.append(initial_section)
        self.add_sections_to_queue([initial_section])
        
        # Create initial section ID
        # initial_id = f"{input_type}:{document_path}:{'-1' if input_type == 'pdf' else ''}"
        if input_type == "pdf":
            initial_id = f"pdf:{document_path}:-1"
        else:
            initial_id = f"html:{document_path}:-1"
        
        # Add the initial section to the section map with its parent (None for root)
        section_map[initial_id] = {
            "data": {
                "name": initial_section["section_name"],
                "type": initial_section["type"],
                "path": initial_section["path"],
                "range": initial_section.get("range", "-1"),
                "policies_count": 0,
                "children": []
            },
            "parent_id": None,
            "has_policies": False
        }
        
        # Main extraction loop
        sections_processed = 0
        policies_extracted = 0
        
        while True:
            # Get next section to process
            section = self.get_next_section()
            if not section:
                self.log("Exploration queue is empty, extraction complete")
                break
                
            self.log(f"Processing section: {section['section_name']} ({section['type']}:{section['path']})")
            sections_processed += 1
            
            # Get section identifier
            if section['type'] == 'pdf':
                current_section_id = f"pdf:{section['path']}:{section.get('range', '-1')}"
            else:
                current_section_id = f"html:{section['path']}:-1"

            assert current_section_id in section_map, f"Section {current_section_id} not found in section_map"
            
            # Extract text from the section
            try:
                if section['type'] == 'pdf':
                    result = await self.call_tool(
                        "extract_text_from_pdf", 
                        pdf_path=section['path'],
                        output_dir=extraction_dir,
                        page_range=section.get('range', '-1')
                    )
                else:  # HTML
                    result = await self.call_tool(
                        "extract_text_from_html", 
                        url=section['path'],
                        output_dir=extraction_dir
                    )
                
                if not result.get("success", False):
                    error_msg = result.get("error", "Unknown error")
                    self.log(f"Error extracting text from {section['type']} document: {error_msg}", "ERROR")
                    continue
                
                file_path = result.get("file_path")
                if not file_path:
                    self.log(f"No file path returned in extraction result", "ERROR")
                    continue
                    
                # Read the content from the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.log(f"Loaded text from {section['type']} document: {file_path}")
                
                # Use the MCP tool to analyze the document section
                analysis = await self.call_tool(
                    "analyze_document_section",
                    content=content,
                    section_info=section,
                    user_request=self.user_request,
                    sections_found=self.sections_found
                )
                
                self.log(f"Analysis result: {analysis}")
                
                # Extract policies if the section contains them
                if analysis.get("has_policies", False):
                    self.log(f"Section contains policies, extracting...")
                    
                    policy_result = await self.call_tool(
                        "extract_policies_from_file",
                        file_path=file_path,
                        organization=self.organization
                    )
                    
                    # Process the structured dictionary response
                    if policy_result.get("success", False):
                        count = policy_result.get("count", 0)
                        
                        if count > 0:
                            policies_extracted += count
                            
                            # Update the section map with policy count and mark as having policies
                            # if current_section_id in section_map:
                            section_map[current_section_id]["data"]["policies_count"] = count
                            section_map[current_section_id]["has_policies"] = True
                                
                            self.log(f"Extracted {count} policies from section")
                        else:
                            self.log(f"No policies found in this section", "INFO")
                    else:
                        error = policy_result.get("error", "Unknown error")
                        self.log(f"Failed to extract policies: {error}", "WARN")
                
                # Add subsections to the queue for exploration
                if deep_policy and analysis.get("has_subsections", False):
                    subsections = analysis.get("subsections", [])
                    if subsections:
                        for subsection in subsections:
                            # Create subsection ID
                            if subsection['type'] == 'pdf':
                                subsection_id = f"pdf:{subsection['path']}:{subsection.get('range', '-1')}"
                            else:
                                subsection_id = f"html:{subsection['path']}:-1"
                            
                            if subsection_id not in section_map:
                                # Add subsection to section map with reference to parent
                                section_map[subsection_id] = {
                                    "data": {
                                        "name": subsection["section_name"],
                                        "type": subsection["type"],
                                        "path": subsection["path"],
                                        "range": subsection.get("range", "-1") if subsection["type"] == "pdf" else '-1',
                                        "policies_count": 0,
                                        "children": []
                                    },
                                    "parent_id": current_section_id,
                                    "has_policies": False
                                }
                            
                            self.sections_found.append(subsection)
                        # Add subsections to processing queue
                        added = self.add_sections_to_queue(subsections)
                        self.log(f"Added {added} new subsections to exploration queue")
                
            except Exception as e:
                self.log(f"Error processing section: {str(e)}", "ERROR")
                import traceback
                self.log(traceback.format_exc(), "ERROR")
                continue
                
            # Log status after each section
            queue_size = len(self.exploration_queue)
            self.log(f"Status: {sections_processed} sections processed, {policies_extracted} policies extracted, {queue_size} sections remaining in queue")
        
        # Now build the document tree from the section map, including only sections with policies
        document_tree = {
            "name": "Root",
            "type": "root",
            "path": document_path,
            "policies_count": 0,
            "children": []
        }
        
        # If no policies were found at all, add the initial document for visualization
        if policies_extracted == 0 and initial_id in section_map:
            document_tree["children"].append(section_map[initial_id]["data"])
            self.log("No policies found in any section. Adding initial document to tree for visualization.")
        else:
            # First pass: create nodes for sections with policies
            nodes_with_policies = {}
            
            for section_id, section_info in section_map.items():
                if section_info["has_policies"]:
                    # Deep copy the data to avoid modifying the original
                    node_data = {
                        "name": section_info["data"]["name"],
                        "type": section_info["data"]["type"],
                        "path": section_info["data"]["path"],
                        "range": section_info["data"]["range"],
                        "policies_count": section_info["data"]["policies_count"],
                        "children": []
                    }
                    nodes_with_policies[section_id] = node_data
            
            # Second pass: establish parent-child relationships
            for section_id, node_data in nodes_with_policies.items():
                parent_id = section_map[section_id]["parent_id"]
                
                # If parent has policies, add as child to parent
                if parent_id in nodes_with_policies:
                    nodes_with_policies[parent_id]["children"].append(node_data)
                else:
                    # If parent doesn't have policies or is None, add to root
                    document_tree["children"].append(node_data)
        
        # Get all extracted policies from the central JSON file
        try:
            with open(self.policies_path, 'r') as f:
                policies = json.load(f)
                self.log(f"Loaded {len(policies)} policies from central JSON file")
        except (FileNotFoundError, json.JSONDecodeError):
            self.log("No policies found in central JSON file or file not created yet", "WARN")
            policies = []
            
        # Save document tree to file
        tree_file = os.path.join(extraction_dir, f"{self.organization}_document_tree.json")
        with open(tree_file, 'w') as f:
            json.dump(document_tree, f, indent=2)
        
        # Generate document tree visualization using the helper function
        try:
            viz_file = os.path.join(extraction_dir, f"{self.organization}_document_tree.png")
            plot_document_tree(document_tree, output_path=viz_file)
            self.log(f"Saved document tree visualization to {viz_file}")
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}", "ERROR")
        
        # Save extraction report
        report = {
            "document_path": document_path,
            "input_type": input_type,
            "organization": self.organization,
            "extraction_id": self.current_extraction_id,
            "deep_policy": deep_policy,
            "sections_processed": sections_processed,
            "policies_extracted": policies_extracted,
            "visited_sections": list(self.visited_sections),
            "document_tree": document_tree,
            "document_tree_file": tree_file,
            "visualization_file": viz_file if 'viz_file' in locals() else None,
            "timestamp": datetime.now().isoformat()
        }
        
        report_file = os.path.join(extraction_dir, f"{self.organization}_extraction_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
            
        self.log(f"Extraction complete. Processed {sections_processed} sections, extracted {policies_extracted} policies.")
        self.log(f"Document tree saved to {tree_file}")
        return policies
    
    def log(self, message: str, level: str = "INFO"):
        """
        Log a message with timestamp.
        
        Args:
            message: Message to log
            level: Log level (INFO, WARN, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        if self.debug:
            if level == "WARN":
                print(f"\033[93m{log_entry}\033[0m")
            elif level == "ERROR":
                print(f"\033[91m{log_entry}\033[0m")
            else:
                print(f"\033[92m{log_entry}\033[0m")
        
        self.debug_log.append(log_entry)
        
        # Also log using the standard logging module
        if level == "WARN":
            logger.warning(message)
        elif level == "ERROR":
            logger.error(message)
        else:
            logger.info(message)


async def main():
    """Command-line interface for policy extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract policies from policy documents")
    parser.add_argument("--document-path", "-d", required=True, help="Path to PDF file or URL for HTML")
    parser.add_argument("--organization", "-org", required=True, help="Name of the organization")
    parser.add_argument("--input-type", "-t", choices=["pdf", "html"], default="pdf", 
                        help="Type of input document (pdf or html)")
    parser.add_argument("--deep-policy", "-dp", action="store_true", 
                        help="Whether to explore linked pages/sections")
    parser.add_argument("--output-dir", "-o", default="./output/deep_policy", 
                        help="Directory to save output files")
    parser.add_argument("--model", "-m", default="claude-3-7-sonnet-20250219", 
                        help="Model to use (claude-3-7-sonnet-20250219 or gpt-4o)")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug mode")
    parser.add_argument("--user-request", "-u", default="", 
                        help="User request for the policy extraction")
    
    args = parser.parse_args()
    
    # Initialize PolicyExtractionAgent with output directory
    system = PolicyExtractionAgent(
        output_dir=args.output_dir,
        organization=args.organization,
        user_request=args.user_request,
        debug=args.debug,
        model=args.model
    )
    
    try:
        # Start MCP servers
        await system.start_mcp_servers()
        
        # Extract policies
        policies = await system.extract_policies(
            args.document_path,
            args.input_type,
            args.deep_policy
        )
        
        print(f"\nExtracted {len(policies)} policies from {args.document_path}")
        if policies:
            print("\nSample policies:")
            for i, policy in enumerate(policies[:3]):  # Show up to 3 sample policies
                print(f"\nPolicy {i+1}:")
                print(f"Description: {policy.get('policy_description', '')[:100]}...")
                print(f"Scope: {policy.get('scope', 'None')}")
                print(f"Reference: {', '.join(policy.get('reference', []))[:100]}...")
            
            if len(policies) > 3:
                print(f"\n... and {len(policies) - 3} more policies")
            
            print(f"\nResults saved to {system.policies_path}")
    
    except Exception as e:
        print(f"Error in policy extraction process: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Log the final cleanup process
        print("Starting final cleanup process...")
        
        # Shutdown MCP servers
        print("Signaling MCP servers to shutdown...")
        try:
            await system.stop_mcp_servers()
        except Exception as e:
            print(f"Note: Error during MCP server shutdown (this is expected and can be ignored): {str(e)}")


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