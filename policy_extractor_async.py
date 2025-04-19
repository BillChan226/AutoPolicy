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
import time  # Add this import at the top with other imports
from dotenv import load_dotenv

# Import MCP server for tool access
from agents.mcp import MCPServerStdio
from utility.utils import *

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




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
    - Supports parallel processing of multiple sections (async mode)
    """
    
    def __init__(
        self,
        mcp_server: MCPServerStdio,
        output_dir: Optional[str] = None,
        organization: str = "organization",
        organization_description: str = "organization description",
        target_subject: str = "target subject",
        user_request: str = "",
        debug: bool = False,
        model: str = "claude-3-7-sonnet-20250219",
        async_sections: int = 1  # Default to 1 section at a time (no parallelism)
    ):
        """
        Initialize the PolicyExtractionAgent.
        
        Args:
            output_dir: Directory for storing output files
            organization: Name of the organization
            user_request: User request for the policy extraction
            debug: Whether to print debug information
            model: The LLM model to use for analysis
            async_sections: Number of sections to process in parallel (1-3)
        """
        self.debug = debug
        self.debug_log = []
        self.user_request = user_request
        self.organization = organization
        self.organization_description = organization_description
        self.target_subject = target_subject
        self.model = model
        
        # Limit async_sections to a reasonable range (1-3)
        self.async_sections = min(max(1, async_sections), 3)
        
        # Set up output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create extraction directory for this run
        self.current_extraction_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.extraction_dir = os.path.join(self.output_dir, f"extraction_{self.current_extraction_id}")
        os.makedirs(self.extraction_dir, exist_ok=True)
        # Define the policy path for storing all extracted policies
        self.policies_path = os.path.join(self.extraction_dir, f"{organization}_all_extracted_policies.json")
        self.rules_path = os.path.join(self.extraction_dir, f"{organization}_all_extracted_rules.json")
        self.risk_categories_path = os.path.join(self.extraction_dir, f"{organization}_risk_categories.json")

        
        # Initialize MCP servers
        self.policy_extraction_tool = mcp_server
        
        # Initialize clients for OpenAI and Anthropic
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        # Initialize exploration queue and tracking sets
        self.exploration_queue = []  # Priority queue
        self.processing_sections = set()  # Track sections currently being processed
        self.visited_sections = set()  # Track visited sections to avoid duplicates
        
        
        logger.info(f"PolicyExtractionAgent initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Policy database: {self.policies_path}")
        logger.info(f"Async tasks: {self.async_sections}")
    
    def _initialize_mcp_servers(self):
        """Initialize MCP servers for policy extraction tools."""
        # Policy extraction MCP server with environment variables for the policy database path
        env_vars = os.environ.copy()
        env_vars["POLICY_PATH"] = self.policies_path
        
        self.policy_extraction_tool = MCPServerStdio(
            name="Policy Extraction Server",
            params={
                "command": "python",
                "args": ["-m", "utility.policy_server"],
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
        """Stop all MCP servers with proper task cancellation."""
        try:
            logger.info("Shutting down MCP servers")
            
            # Proper cleanup for the policy_extraction_tool
            if hasattr(self.policy_extraction_tool, "disconnect"):
                await self.policy_extraction_tool.disconnect()
            elif hasattr(self.policy_extraction_tool, "close"):
                await self.policy_extraction_tool.close()
            
            # Allow time for subprocess cleanup
            await asyncio.sleep(0.5)
            logger.info("MCP servers shut down")
        except Exception as e:
            logger.error(f"Error during MCP server shutdown: {str(e)}")
    
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
                
            if section['type'] not in ['pdf', 'html', 'txt']:
                self.log(f"Skipping section with invalid type '{section['type']}': {section['section_name']}", "WARN")
                continue
                
            # Create a unique identifier for this section
            if section['type'] == 'pdf':
                section_id = f"pdf:{section['path']}:{section.get('range', '-1')}"
            elif section['type'] == 'txt':
                section_id = f"txt:{section['path']}:-1"
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
    
    def get_next_sections(self, k: int = None) -> List[Dict[str, Any]]:
        """
        Get the next k sections from the exploration queue.
        
        Args:
            k: Number of sections to retrieve (defaults to self.async_sections)
            
        Returns:
            List of section dictionaries (may be fewer than k if queue is almost empty)
        """
        if k is None:
            k = self.async_sections
            
        # Ensure k is within bounds
        k = min(max(1, k), 3)
        
        sections = []
        for _ in range(k):
            if not self.exploration_queue:
                break
                
            # Get the highest priority item
            item = heapq.heappop(self.exploration_queue)
            section = item.section
            
            # Create section identifier for tracking
            if section['type'] == 'pdf':
                section_id = f"pdf:{section['path']}:{section.get('range', '-1')}"
            elif section['type'] == 'txt':
                section_id = f"txt:{section['path']}:-1"
            else:
                section_id = f"html:{section['path']}:-1"
                
            # Add to processing set
            self.processing_sections.add(section_id)
            sections.append(section)
            
        return sections
    
    async def process_section(
        self, 
        section: Dict[str, Any],
        extraction_dir: str,
        section_map: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a single section for policy extraction.
        
        Args:
            section: Section dictionary
            extraction_dir: Directory for this extraction run
            section_map: Map of all sections
            
        Returns:
            Result dictionary with processing results
        """
        start_time = time.time()  # Start timing for this section
        self.log(f"Processing section: {section['section_name']} ({section['type']}:{section['path']})")
        
        # Get section identifier
        if section['type'] == 'pdf':
            current_section_id = f"pdf:{section['path']}:{section.get('range', '-1')}"
        elif section['type'] == 'txt':
            current_section_id = f"txt:{section['path']}:-1"
        else:
            current_section_id = f"html:{section['path']}:-1"
            
        # Process result dictionary to track outcomes
        result = {
            "section_id": current_section_id,
            "section_name": section['section_name'],
            "has_policies": False,
            "policies_count": 0,
            "has_subsections": False,
            "subsections": [],
            "error": None
        }

        try:
            # Extract text from the section
            if section['type'] == 'pdf':
                extract_result = await self.call_tool(
                    "extract_text_from_pdf", 
                    pdf_path=section['path'],
                    output_dir=extraction_dir,
                    page_range=section.get('range', '-1')
                )
            elif section['type'] == 'txt':
                # For txt files, just read the content directly
                try:
                    with open(section['path'], 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create a temporary file in the extraction directory to store the content
                    import os.path
                    file_name = os.path.basename(section['path'])
                    file_path = os.path.join(extraction_dir, file_name)
                    
                    # Write content to the extraction directory (for consistency with other file types)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    extract_result = {
                        "success": True,
                        "file_path": file_path,
                        "content": content
                    }
                except Exception as e:
                    error_msg = f"Error reading txt file: {str(e)}"
                    self.log(error_msg, "ERROR")
                    extract_result = {
                        "success": False,
                        "error": error_msg
                    }
            else:  # HTML
                extract_result = await self.call_tool(
                    "extract_text_from_html", 
                    url=section['path'],
                    output_dir=extraction_dir
                )
            
            if not extract_result.get("success", False):
                error_msg = extract_result.get("error", "Unknown error")
                self.log(f"Error extracting text from {section['type']} document: {error_msg}", "ERROR")
                result["error"] = error_msg
                return result
            
            file_path = extract_result.get("file_path")
            if not file_path:
                self.log(f"No file path returned in extraction result", "ERROR")
                result["error"] = "No file path returned"
                return result
                
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
            
            self.log(f"Section {section['section_name']} includes policies for extraction and {len(analysis.get('subsections', []))} subsections", "INFO")
            self.log(f"Subsections: {analysis.get('subsections', [])}", "RESULT")
            
            # Extract policies if the section contains them
            if analysis.get("has_policies", False):
                result["has_policies"] = True
                self.log(f"Section contains policies, extracting...")
                
                policy_result = await self.call_tool(
                    "extract_policies_from_file",
                    file_path=file_path,
                    organization=self.organization,
                    organization_description=self.organization_description,
                    target_subject=self.target_subject,
                    policy_db_path=self.policies_path
                )
                
                # Process the structured dictionary response
                if policy_result.get("success", False):
                    count = policy_result.get("count", 0)
                    result["policies_count"] = count
                    
                    # Get the policy IDs that were just extracted
                    policy_ids = []
                    if count > 0:
                        # Extract the policy IDs from the result
                        if "policies" in policy_result and isinstance(policy_result["policies"], list):
                            # If the policies are returned directly
                            for i, p in enumerate(policy_result["policies"]):
                                policy_id = p.get("policy_id")
                                policy_ids.append(policy_id)
                        
                        result["policy_ids"] = policy_ids
                        
                        # Update the section map with policy count, IDs, and mark as having policies
                        if current_section_id in section_map:
                            section_map[current_section_id]["data"]["policies_count"] = count
                            section_map[current_section_id]["data"]["policy_ids"] = policy_ids
                            section_map[current_section_id]["has_policies"] = True
                            self.log(f"Extracted {count} policies from section with IDs: {policy_ids}")
                        else:
                            self.log(f"No policies found in this section", "INFO")
                    else:
                        self.log(f"No policies found in this section", "INFO")
                else:
                    error = policy_result.get("error", "Unknown error")
                    self.log(f"Failed to extract policies: {error}", "WARN")
                    result["error"] = error
            
            # Check for subsections
            if analysis.get("has_subsections", False):
                result["has_subsections"] = True
                subsections = analysis.get("subsections", [])
                result["subsections"] = subsections
                self.log(f"Section has {len(subsections)} subsections")
                
            return result
            
        except Exception as e:
            self.log(f"Error processing section: {str(e)}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            result["error"] = str(e)
            return result
        finally:
            # Calculate processing time
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            self.log(f"Section '{section['section_name']}' processed in {processing_time:.2f} seconds")
            
            # Remove from processing set regardless of outcome
            self.processing_sections.remove(current_section_id)
    
    async def extract_policies(
        self, 
        document_path: str, 
        input_type: str = "pdf",
        initial_page_range: str = "1-3",
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
        # Start timing the overall extraction process
        overall_start_time = time.time()

        self.log(f"Created extraction directory: {self.extraction_dir}")
        
        # Reset exploration state
        self.exploration_queue = []
        self.processing_sections = set()
        self.sections_found = []
        self.visited_sections = set()
        self.section_times = []  # Track processing times for all sections
        
        # Initialize section map to track all processed sections
        # Format: {section_id: {data: section_data, parent_id: parent_section_id, has_policies: bool}}
        section_map = {}
        
        # Add the initial document as the first section to explore
        initial_section = {
            "section_name": "Initial Document",
            "path": document_path,
            "range": initial_page_range if input_type == "pdf" else "",
            "type": input_type,
            "priority": 10,  # Highest priority
            "source": "User Input"
        }
        self.sections_found.append(initial_section)
        self.add_sections_to_queue([initial_section])
        
        # Create initial section ID
        if input_type == "pdf":
            initial_id = f"pdf:{document_path}:{initial_page_range}"
        else:
            initial_id = f"html:{document_path}:-1"
        
        # Add the initial section to the section map with its parent (None for root)
        section_map[initial_id] = {
            "data": {
                "name": initial_section["section_name"],
                "type": initial_section["type"],
                "path": initial_section["path"],
                "range": initial_section.get("range", initial_page_range),
                "policies_count": 0,
                "policy_ids": [],  # Track policy IDs extracted from this section
                "children": []
            },
            "parent_id": None,
            "has_policies": False
        }
        
        # Main extraction loop
        sections_processed = 0
        policies_extracted = 0
        
        while self.exploration_queue or self.processing_sections:
            # Skip getting new sections if we're already processing the maximum number
            if len(self.processing_sections) >= self.async_sections:
                # Wait a bit for an ongoing task to complete
                await asyncio.sleep(0.1)
                continue
                
            # Calculate how many new sections we can process
            available_slots = self.async_sections - len(self.processing_sections)
            if available_slots <= 0 or not self.exploration_queue:
                # Wait for ongoing processing to complete if no slots available
                await asyncio.sleep(0.1)
                continue
                
            # Get batch of sections to process
            sections_to_process = self.get_next_sections(k=available_slots)
            if not sections_to_process:
                # If no sections to process but we have ongoing processing, wait
                if self.processing_sections:
                    await asyncio.sleep(0.1)
                    continue
                else:
                    # Nothing to process and nothing in progress - we're done
                    break
            
            # Process sections in parallel
            self.log(f"Processing {len(sections_to_process)} sections in parallel")
            tasks = [
                self.process_section(section, self.extraction_dir, section_map)
                for section in sections_to_process
            ]
            
            # Wait for all processing to complete
            results = await asyncio.gather(*tasks)
            self.log(f"All {len(sections_to_process)} sections processed")
            
            # Process the results sequentially
            for section, result in zip(sections_to_process, results):
                sections_processed += 1
                
                # Track section processing time
                if "processing_time" in result:
                    self.section_times.append({
                        "section_name": section["section_name"],
                        "processing_time": result["processing_time"]
                    })
                
                if result["has_policies"]:
                    policies_extracted += result["policies_count"]
                
                # Add subsections to the queue for exploration, but do this sequentially to avoid race conditions with section IDs
                if deep_policy and result["has_subsections"]:
                    subsections = result["subsections"]
                    if subsections:
                        # Get section identifier for parent relationship
                        if section['type'] == 'pdf':
                            current_section_id = f"pdf:{section['path']}:{section.get('range', '-1')}"
                        else:
                            current_section_id = f"html:{section['path']}:-1"
                            
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
                                        "policy_ids": [],  # Initialize empty policy IDs array
                                        "children": []
                                    },
                                    "parent_id": current_section_id,
                                    "has_policies": False
                                }
                            
                                self.sections_found.append(subsection)
                        # Add subsections to processing queue
                        added = self.add_sections_to_queue(subsections)
                        self.log(f"Added {added} new subsections to exploration queue")
            
            # Log status after processing batch
            queue_size = len(self.exploration_queue)
            in_progress = len(self.processing_sections)
            self.log(f"Status: {sections_processed} sections processed, {policies_extracted} policies extracted, {queue_size} sections in queue, {in_progress} in progress")
        
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
                        "policy_ids": section_info["data"]["policy_ids"],  # Include policy IDs in the tree
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
                    
        
        # Get all extracted policies from the policy DB JSON file
        try:
            with open(self.policies_path, 'r') as f:
                policies = json.load(f)

                self.log(f"Loaded {len(policies)} indexed policies")
        except (FileNotFoundError, json.JSONDecodeError):
            self.log("No policies found in policy database JSON file or file not created yet", "WARN")
            policies = []
            
        # Save document tree to file
        tree_file = os.path.join(self.extraction_dir, f"{self.organization}_document_tree.json")
        with open(tree_file, 'w') as f:
            json.dump(document_tree, f, indent=2)
        

        # Calculate overall extraction time
        overall_processing_time = time.time() - overall_start_time
        
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
            "timestamp": datetime.now().isoformat(),
            "overall_processing_time": overall_processing_time,
            "section_processing_times": self.section_times,
            "average_section_time": sum(item["processing_time"] for item in self.section_times) / len(self.section_times) if self.section_times else 0
        }
        
        report_file = os.path.join(self.extraction_dir, f"{self.organization}_extraction_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
            
        self.log(f"Extraction complete. Processed {sections_processed} sections, extracted {policies_extracted} policies.")
        self.log(f"Total extraction time: {overall_processing_time:.2f} seconds")
        
        # Print time statistics
        if self.section_times:
            avg_time = sum(item["processing_time"] for item in self.section_times) / len(self.section_times)
            max_time = max(self.section_times, key=lambda x: x["processing_time"])
            min_time = min(self.section_times, key=lambda x: x["processing_time"])
            
            self.log(f"Time statistics:")
            self.log(f"  Average section processing time: {avg_time:.2f} seconds")
            self.log(f"  Fastest section: {min_time['section_name']} ({min_time['processing_time']:.2f} seconds)")
            self.log(f"  Slowest section: {max_time['section_name']} ({max_time['processing_time']:.2f} seconds)")
        
        self.log(f"Document tree saved to {tree_file}")
        return policies


    async def extract_rules(self, policy_path: str=None, organization: str = None, 
                            organization_description: str = None, target_subject: str = None) -> List[Dict[str, Any]]:
        """
        Extract concrete rules from extracted policies.
        
        Args:
            policy_path: Path to the JSON file containing extracted policies
            organization: Name of the organization (overrides instance property if provided)
            organization_description: Description of the organization (overrides instance property if provided)
            target_subject: Target subject of the policies (overrides instance property if provided)
            
        Returns:
            List of extracted rules
        """
        self.log(f"Extracting rules from {policy_path}", "INFO")
        
        try:
            # Use provided values or fall back to instance properties
            org = organization or self.organization
            org_desc = organization_description or self.organization_description
            subject = target_subject or self.target_subject
        
            if policy_path is None:
                policy_path = self.policies_path

            rules_output_path = self.rules_path
            
            # Call the extract_rules_from_policies tool
            result = await self.call_tool(
                "extract_rules_from_policies",
                policy_path=policy_path,
                organization=org,
                organization_description=org_desc,
                target_subject=subject,
                output_path=rules_output_path
            )
            
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                self.log(f"Error extracting rules: {error_msg}", "ERROR")
                return []
            
            # Extract rules from the tool response
            rules = result.get("rules", [])
            rules_count = len(rules)
            
            self.log(f"Successfully extracted {rules_count} rules from the policy database file at {policy_path}")
            self.log(f"Rules saved to: {rules_output_path}")
            
            # Now call the refine_and_categorize_rules tool
            self.log(f"Refining and categorizing rules...")
            
            
            categorization_result = await self.call_tool(
                "refine_and_categorize_rules",
                rules_path=rules_output_path,
                organization=org,
                organization_description=org_desc,
                target_subject=subject,
                output_path=self.risk_categories_path
            )
            
            if categorization_result.get("success", False):
                categories_count = categorization_result.get("categories_count", 0)
                refined_rules_count = categorization_result.get("rules_count", 0)
                
                self.log(f"Successfully refined and categorized {rules_count} rules into {categories_count} risk categories with {refined_rules_count} refined rules")
                self.log(f"Categorized rules saved to: {self.risk_categories_path}")
                
            else:
                error_msg = categorization_result.get("error", "Unknown error")
                self.log(f"Error during rule categorization: {error_msg}", "WARN")
                self.log("Continuing with uncategorized rules")
            
            # After getting the rules, create a mapping from policy IDs to rule IDs
            policy_to_rules_map = {}
            
            # First load the policies to get all policy IDs
            try:
                with open(policy_path or self.policies_path, 'r') as f:
                    policies_data = json.load(f)
                    
                if isinstance(policies_data, list):
                    policies = policies_data
                else:
                    # Handle case where policies might be under a 'policies' key
                    policies = policies_data.get("policies", [])
                    
                for policy in policies:
                    if isinstance(policy, dict):
                        # Try multiple possible ID keys
                        policy_id = policy.get("policy_id")
                        policy_to_rules_map[str(policy_id)] = []
            except Exception as e:
                self.log(f"Error loading policies for mapping: {str(e)}", "WARN")
                import traceback
                self.log(traceback.format_exc(), "DEBUG")
                
            # Then load the rules to map back to policies
            try:
                with open(self.risk_categories_path, 'r') as f:
                    risk_categories_data = json.load(f)
                    
                    # Properly handle different possible structures
                    if isinstance(risk_categories_data, dict) and "risk_categories" in risk_categories_data:
                        categories = risk_categories_data["risk_categories"]
                    elif isinstance(risk_categories_data, list):
                        categories = risk_categories_data
                    else:
                        categories = []
                        
                    for risk_category in categories:
                        if "rules" in risk_category and isinstance(risk_category["rules"], list):
                            for rule in risk_category["rules"]:
                                rule_id = rule.get("rule_id")
                                
                                # Handle both source_policy_id (singular) and source_policy_ids (plural)
                                source_policy_ids = rule.get("source_policy_ids", [])
                                
                                # If we have multiple source_policy_ids, add each one to the mapping
                                if rule_id is not None and source_policy_ids:
                                    if isinstance(source_policy_ids, list):
                                        for str_policy_id in source_policy_ids:
                                            if str_policy_id is not None:
                                                str_policy_id = str(str_policy_id)
                                                if str_policy_id in policy_to_rules_map:
                                                    policy_to_rules_map[str_policy_id].append(rule_id)
                                                else:
                                                    policy_to_rules_map[str_policy_id] = [rule_id]
                                    else:
                                        # Handle case where source_policy_ids might be a single value
                                        str_policy_id = str(source_policy_ids)
                                        if str_policy_id in policy_to_rules_map:
                                            policy_to_rules_map[str_policy_id].append(rule_id)
                                        else:
                                            policy_to_rules_map[str_policy_id] = [rule_id]
            except Exception as e:
                self.log(f"Error mapping rules to policies: {str(e)}", "WARN")
                import traceback
                self.log(traceback.format_exc(), "DEBUG")
            
            # Save the policy-to-rules mapping
            mapping_path = os.path.join(self.extraction_dir, f"{self.organization}_policy_rule_mapping.json")
            with open(mapping_path, 'w') as f:
                json.dump(policy_to_rules_map, f, indent=2)
            
            self.log(f"Created policy-to-rules mapping at: {mapping_path}")
            
            # Save an enhanced document tree with the full hierarchy
            try:
                # Load the existing document tree
                tree_file = os.path.join(self.extraction_dir, f"{self.organization}_document_tree.json")
                with open(tree_file, 'r') as f:
                    document_tree = json.load(f)
                    
                # Enhance the tree with rule information
                self._enhance_tree_with_rules(document_tree, policy_to_rules_map)
                
                # Save the enhanced tree
                enhanced_tree_file = os.path.join(self.extraction_dir, f"{self.organization}_document_tree.json")
                with open(enhanced_tree_file, 'w') as f:
                    json.dump(document_tree, f, indent=2)
                    
                self.log(f"Updated document tree with policy-rule linkage: {enhanced_tree_file}")
            except Exception as e:
                self.log(f"Error enhancing document tree: {str(e)}", "WARN")
            
            return rules
            
        except Exception as e:
            self.log(f"Error in extract_rules: {str(e)}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            return []
        

    def log(self, message: str, level: str = "INFO"):
        """
        Log a message with timestamp.
        
        Args:
            message: Message to log
            level: Log level (INFO, WARN, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        # if self.debug:
        if level == "WARN":
            print(f"\033[93m{log_entry}\033[0m")
        elif level == "ERROR": # red
            print(f"\033[91m{log_entry}\033[0m")
        elif level == "INFO": # green
            print(f"\033[92m{log_entry}\033[0m")
        else: 
            print(log_entry)
        
        self.debug_log.append(log_entry)
        
        # # Also log using the standard logging module
        # if level == "WARN":
        #     logger.warning(message)
        # elif level == "ERROR":
        #     logger.error(message)
        # else:
        #     logger.info(message)


    def _enhance_tree_with_rules(self, tree_node, policy_to_rules_map):
        """
        Recursively enhance tree nodes with rule information.
        
        Args:
            tree_node: The current node in the document tree
            policy_to_rules_map: Mapping from policy IDs to rule IDs
        """
        # Skip if tree_node is not a dictionary
        if not isinstance(tree_node, dict):
            return
        
        # Add rule_ids array if this node has policies
        if "policy_ids" in tree_node and tree_node["policy_ids"]:
            # First collect all rule IDs for statistics and backward compatibility
            all_rule_ids = []
            
            # Then create a structured mapping of policies to their rules
            policies_to_rules = {}
            
            for policy_id in tree_node["policy_ids"]:
                # Skip None policy IDs
                if policy_id is None:
                    continue
                
                # Convert to string for mapping lookup
                str_policy_id = str(policy_id)
                if str_policy_id in policy_to_rules_map:
                    # Get the rules for this policy
                    policy_rules = policy_to_rules_map[str_policy_id]
                    if isinstance(policy_rules, list):
                        all_rule_ids.extend(policy_rules)
                        policies_to_rules[str_policy_id] = policy_rules
                    else:
                        # If it's not a list, add the item directly
                        all_rule_ids.append(policy_rules)
                        policies_to_rules[str_policy_id] = [policy_rules]
            
            # Add both the flat list for backwards compatibility and the structured mapping
            tree_node["rule_ids"] = all_rule_ids
            tree_node["rules_count"] = len(all_rule_ids)
            tree_node["policies_to_rules"] = policies_to_rules
        
        # Process children recursively if they exist and are a list
        if "children" in tree_node and isinstance(tree_node["children"], list):
            for child in tree_node["children"]:
                self._enhance_tree_with_rules(child, policy_to_rules_map)


async def main():
    """Command-line interface for policy extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract policies from policy documents")
    parser.add_argument("--document-path", "-d", required=True, help="Path to PDF file or URL for HTML")
    parser.add_argument("--organization", "-org", required=True, help="Name of the organization")
    parser.add_argument("--organization-description", "-org-desc", required=False, default="", help="Description of the organization")
    parser.add_argument("--target-subject", "-ts", required=False, default="User", help="Target subject of the organization")
    parser.add_argument("--input-type", "-t", choices=["pdf", "html", "txt"], default="pdf", 
                        help="Type of input document (pdf, html, or txt)")
    parser.add_argument("--initial-page-range", "-ipr", default="1-5",
                        help="Initial page range to extract from PDF")
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
    parser.add_argument("--async-num", "-a", type=int, default=1,
                        help="Number of policy extraction tasks to run in parallel (1-3)")
    parser.add_argument("--extract-rules", "-er", action="store_true",
                        help="Extract concrete rules from policies after extraction")
    
    args = parser.parse_args()

    # Check if document is a pdf or txt file then set input_type appropriately
    if args.document_path.endswith(".pdf"):
        assert args.input_type == "pdf", "Input type must be pdf if document is a pdf file"
    elif args.document_path.endswith(".txt"):
        assert args.input_type == "txt", "Input type must be txt if document is a txt file"
    elif args.document_path.startswith("http"):
        assert args.input_type == "html", "Input type must be html if document is a URL"
    
    # Initialize PolicyExtractionAgent with output directory
    env_vars = os.environ.copy()

    # ➊ Open and close the MCP server in the SAME coroutine
    async with MCPServerStdio(
        name="Policy Extraction Server",
        params={
            "command": "python",
            "args": ["-m", "utility.policy_server"],
            "env": env_vars,
        },
        cache_tools_list=True,
    ) as mcp_server:

        # ➋ Pass the ready‑to‑use server into your agent
        policy_agent = PolicyExtractionAgent(
            mcp_server=mcp_server,            
            output_dir=args.output_dir,
            organization=args.organization,
            user_request=args.user_request,
            debug=args.debug,
            model=args.model,
            async_sections=args.async_num,
        )

        policies = await policy_agent.extract_policies(
            args.document_path, args.input_type, args.initial_page_range, args.deep_policy
        )


        if args.extract_rules:
            rules = await policy_agent.extract_rules(
                policy_path=policy_agent.policies_path,
                organization=args.organization,
                organization_description=args.organization_description,
                target_subject=args.target_subject
            )


if __name__ == "__main__":

    asyncio.run(main())