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

from shield.policy_construction.utility import *
from dotenv import load_dotenv
from shield.utils import *

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
        output_dir: Optional[str] = None,
        organization: str = "organization",
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
        # Define the central path for storing all extracted policies
        self.policies_path = os.path.join(self.extraction_dir, f"{organization}_all_extracted_policies.json")
        
        
        # Initialize exploration queue and tracking sets
        self.exploration_queue = []  # Priority queue
        self.processing_sections = set()  # Track sections currently being processed
        self.visited_sections = set()  # Track visited sections to avoid duplicates
        
        
        logger.info(f"PolicyExtractionAgent initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Central policies file: {self.policies_path}")
        logger.info(f"Async sections: {self.async_sections}")
    

    
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
                extract_result = extract_text_from_pdf(
                    pdf_path=section['path'],
                    output_dir=extraction_dir,
                    page_range=section.get('range', '-1')
                )
            else:  # HTML
                extract_result = extract_text_from_html(
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
            analysis = analyze_document_section(
                content=content,
                section_info=section,
                user_request=self.user_request,
                sections_found=self.sections_found
            )
            
            self.log(f"Section {section['section_name']} includes policies and {len(analysis.get('subsections', []))} subsections", "INFO")
            self.log(f"Subsections: {analysis.get('subsections', [])}", "RESULT")
            
            # Extract policies if the section contains them
            if analysis.get("has_policies", False):
                result["has_policies"] = True
                self.log(f"Section contains policies, extracting...")
                
                policy_result = extract_policies_from_file(
                    file_path=file_path,
                    organization=self.organization
                )
                
                # Process the structured dictionary response
                if policy_result.get("success", False):
                    count = policy_result.get("count", 0)
                    result["policies_count"] = count
                    
                    if count > 0:
                        # Update the section map with policy count and mark as having policies
                        if current_section_id in section_map:
                            section_map[current_section_id]["data"]["policies_count"] = count
                            section_map[current_section_id]["has_policies"] = True
                            self.log(f"Extracted {count} policies from section")
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
            "range": "-1" if input_type == "pdf" else "",
            "type": input_type,
            "priority": 10,  # Highest priority
            "source": "User Input"
        }
        self.sections_found.append(initial_section)
        self.add_sections_to_queue([initial_section])
        
        # Create initial section ID
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
                
                # Add subsections to the queue for exploration, but do this sequentially
                # to avoid race conditions with section IDs
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
        tree_file = os.path.join(self.extraction_dir, f"{self.organization}_document_tree.json")
        with open(tree_file, 'w') as f:
            json.dump(document_tree, f, indent=2)
        
        # Generate document tree visualization using the helper function
        try:
            viz_file = os.path.join(self.extraction_dir, f"{self.organization}_document_tree.png")
            plot_document_tree(document_tree, output_path=viz_file)
            self.log(f"Saved document tree visualization to {viz_file}")
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}", "ERROR")
        
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
    parser.add_argument("--async-num", "-a", type=int, default=1,
                        help="Number of policy extraction tasks to run in parallel (1-3)")
    
    args = parser.parse_args()
    
    # Initialize PolicyExtractionAgent with output directory
    system = PolicyExtractionAgent(
        output_dir=args.output_dir,
        organization=args.organization,
        user_request=args.user_request,
        debug=args.debug,
        model=args.model,
        async_sections=args.async_num
    )
    
    policies = await system.extract_policies(
        args.document_path,
        args.input_type,
        args.deep_policy
    )

if __name__ == "__main__":
    # Use a more controlled event loop shutdown
    asyncio.run(main())