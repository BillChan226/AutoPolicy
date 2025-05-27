#!/usr/bin/env python3
"""
Policy Extraction MCP Tool for ShieldAgent.
Provides capabilities for extracting text from policy documents (PDF/HTML)
and parsing structured policies from the extracted text.
"""

import os
import json
import pathlib
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
import base64
import queue
import threading
from collections import deque

# For PDF processing
import fitz, pymupdf4llm
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
from datetime import datetime


from openai import OpenAI
from anthropic import Anthropic
from shield.utils import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastMCP instance

# Add a global variable to store the pending document sections
# Using a thread-safe queue to store the sections
document_sections_queue = queue.Queue()
visited_sections = set()  # To track which sections have already been visited

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def extract_text_from_pdf(pdf_path: str, output_dir: str = None, page_range: str = "-1", include_links: bool = True) -> Dict[str, Any]:
    """
    Extract text content and links from a PDF file with optional page range and save to a file.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the extracted text file
        page_range: Page number to extract (e.g. "1-10, 5-20"). Use -1 for all pages,
        include_links: Whether to include link information in the output
        
    Returns:
        Dict: Information about the extraction including the file path where text was saved
    """
    try:
        # Create a descriptive filename based on the PDF and page range
        pdf_name = Path(pdf_path).stem
        if page_range == "-1":
            page_desc = "all_pages"
        elif isinstance(page_range, int):
            page_desc = f"page_{page_range}"
        else:
            page_desc = f"pages_{page_range}"
        
        # Create output directory if it doesn't exist
        if not output_dir:
            output_dir = Path(pdf_path).parent
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{pdf_name}_{page_desc}_{timestamp}.txt")
        
        doc = fitz.open(pdf_path)
        full_text = ""
        
        # Determine which pages to process
        if page_range == -1:
            # Process all pages
            pages_to_process = range(len(doc))
        elif isinstance(page_range, int):
            # Process a single page (convert to 0-based index)
            default_range = 5
            if page_range < 1 or page_range > len(doc):
                return {"error": f"Invalid page number {page_range}. The document has {len(doc)} pages."}
            pages_to_process = range(page_range - 1, page_range + default_range)
        elif isinstance(page_range, str) and "-" in page_range:
            # Process a range of pages
            try:
                start, end = map(int, page_range.split("-"))
                if start < 1 or end > len(doc) or start > end:
                    return {"error": f"Invalid page range {page_range}. The document has {len(doc)} pages."}
                pages_to_process = range(start - 1, end)  # Convert to 0-based index
            except ValueError:
                return {"error": f"Invalid page range format '{page_range}'. Use format like '5-10'."}
        else:
            return {"error": f"Invalid page_range parameter. Use -1 for all pages, a single number, or a range like '5-10'."}
        
        # Get TOC items for contextual information
        toc = doc.get_toc()
        toc_by_page = {}
        
        for item in toc:
            level, title, page_num = item
            if page_num not in toc_by_page:
                toc_by_page[page_num] = []
            toc_by_page[page_num].append({"level": level, "title": title})
        
        all_links = []
        # Process each page
        for page_idx in pages_to_process:
            page = doc[page_idx]
            page_num = page_idx + 1  # 1-based page number for output
            
            # Add page header with TOC information if available
            if page_num in toc_by_page:
                headings = toc_by_page[page_num]
                full_text += f"\n--- Page {page_num} Headings ---\n"
                for heading in headings:
                    full_text += f"{'  ' * (heading['level']-1)}• {heading['title']}\n"
            
            # Extract text from the page
            page_text = page.get_text()
            full_text += f"\n--- Page {page_num} Content ---\n{page_text}\n"
            
            # Extract links from the page and include them with this page
            if include_links:
                links = page.get_links()
                if links:
                    full_text += f"\n--- Page {page_num} Links ---\n"
                    for i, link in enumerate(links):
                        if 'uri' in link:
                            full_text += f"Link {i+1}: Web link: {link['uri']}\n"
                            all_links.append({
                                'url': link['uri'],
                                'text': f"Link {i+1}: Web link: {link['uri']}"
                            })
                        elif 'page' in link:
                            target_page = link['page'] + 1  # Convert 0-based to 1-based
                            full_text += f"Link {i+1}: Internal link to page {target_page}\n"
                            all_links.append({
                                'url': f"page_{target_page}",
                                'text': f"Link {i+1}: Internal link to page {target_page}"
                            })
        
        # Write the extracted text to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
            
        logger.info(f"Saved extracted PDF text to {output_file}")
        
        preview_count = 1000
        # Get a preview of the content (first 200 chars)
        preview = full_text[:preview_count] + "..." if len(full_text) > preview_count else full_text
        
        # Return structured data with file path directly accessible
        result = {
            "success": True,
            "file_path": output_file,
            "source": pdf_path,
            "pages": page_desc,
            "total_pages_processed": len(list(pages_to_process)),
            "preview": preview,
            "message": f"""PDF TEXT EXTRACTION COMPLETE:
- Source: {pdf_path}
- Pages: {page_desc}
- Total pages processed: {len(list(pages_to_process))}
- Full page content saved to: {output_file}
- Page content preview: {preview}

If needed, USE THIS FILE PATH to extract policies: {output_file}"""
        }
        
        return result
    
    except Exception as e:
        error_msg = f"Error extracting text from PDF: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    


def extract_text_from_html(url: str, output_dir: str = None, include_links: bool = True, allow_redirects: bool = True) -> Dict[str, Any]:
    """
    Extract text content from an HTML page and save to a file.
    
    Args:
        url: URL of the HTML page
        output_dir: Directory to save the extracted text file
        include_links: Whether to include links in the output
        allow_redirects: Whether to follow URL redirects
        
    Returns:
        Dict: Information about the extraction including the file path where text was saved
    """
    try:
        # Create a descriptive filename based on the URL
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace(".", "_")
        path = parsed_url.path.replace("/", "_").strip("_")
        if not path:
            path = "home"
        
        # Create output directory if it doesn't exist
        if not output_dir:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{domain}_{path}_{timestamp}.txt")
        
        # Make request with redirect handling
        response = requests.get(url, timeout=15, allow_redirects=allow_redirects)
        
        # Check if we were redirected to a different URL
        final_url = response.url
        was_redirected = (final_url != url)
        
        # If the request failed with a 403 Forbidden and redirects weren't allowed, try again with redirects
        if response.status_code == 403 and not allow_redirects:
            logger.info(f"Got 403 Forbidden. Retrying with redirects enabled.")
            response = requests.get(url, timeout=15, allow_redirects=True)
            final_url = response.url
            was_redirected = (final_url != url)
        
        # Raise an exception if the response status code indicates an error
        response.raise_for_status()
        
        # If we were redirected, update the output filename to reflect the final URL
        if was_redirected:
            logger.info(f"URL was redirected: {url} -> {final_url}")
            parsed_final_url = urlparse(final_url)
            final_domain = parsed_final_url.netloc.replace(".", "_")
            final_path = parsed_final_url.path.replace("/", "_").strip("_")
            if not final_path:
                final_path = "home"
            
            # Update output file path with final URL info
            output_file = os.path.join(output_dir, f"{final_domain}_{final_path}_{timestamp}.txt")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get the page title
        title = soup.title.string if soup.title else "No title"
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Extract headings to understand document structure
        headings = []
        for i in range(1, 7):  # h1 to h6
            for heading in soup.find_all(f'h{i}'):
                headings.append(f"h{i}: {heading.get_text(strip=True)}")
        
        # Get text
        text = soup.get_text(separator='\n')
        
        # Format the output
        output = f"--- {title} ---\n\n"
        output += f"Original URL: {url}\n"
        if was_redirected:
            output += f"Redirected URL: {final_url}\n"
        output += "\n"
        
        if headings:
            output += "--- DOCUMENT STRUCTURE ---\n"
            output += "\n".join(headings)
            output += "\n\n"
        
        output += "--- CONTENT ---\n"
        output += text
        
        all_links = []
        # Extract links for potential deep policy exploration
        if include_links:
            links = []
            for a_tag in soup.find_all('a', href=True):
                link = a_tag['href']
                link_text = a_tag.get_text(strip=True)
                
                # Handle relative URLs - use the final URL as base if redirected
                base_url = urlparse(final_url if was_redirected else url)
                if not link.startswith(('http://', 'https://')):
                    if link.startswith('/'):
                        link = f"{base_url.scheme}://{base_url.netloc}{link}"
                    else:
                        link = urljoin(final_url if was_redirected else url, link)
                
                links.append({
                    'url': link,
                    'text': link_text
                })
                all_links.append({
                    'url': link,
                    'text': link_text
                })
            
            # Add links section to the text
            if links:
                output += "\n\n--- PAGE LINKS ---\n"
                for i, link in enumerate(links):
                    output += f"Link {i+1}: {link['text']} -> {link['url']}\n"
        
        # Write the extracted text to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
            
        logger.info(f"Saved extracted HTML text to {output_file}")
        
        # Get a preview of the content (first 1000 chars)
        preview_count = 1000
        preview = text[:preview_count] + "..." if len(text) > preview_count else text
        
        # Return structured data with file path directly accessible
        result = {
            "success": True,
            "file_path": output_file,
            "source": url,
            "final_url": final_url if was_redirected else url,
            "was_redirected": was_redirected,
            "title": title,
            "headings_count": len(headings),
            "links_count": len(links) if include_links and 'links' in locals() else 0,
            "preview": preview,
            "message": f"""HTML TEXT EXTRACTION COMPLETE:
- Source: {url}
{f'- Redirected to: {final_url}' if was_redirected else ''}
- Title: {title}
- Number of headings: {len(headings)}
- Number of links: {len(links) if include_links and 'links' in locals() else 0}
- Full page content saved to: {output_file}
- Page content preview: {preview}

If needed, USE THIS FILE PATH to extract policies: {output_file}"""
        }
        
        return result
    
    except Exception as e:
        error_msg = f"Error extracting text from HTML: {str(e)}"
        logger.error(error_msg)
        
        # If we got a connection error or HTTP error, try one more time with different redirect settings
        if isinstance(e, (requests.HTTPError, requests.ConnectionError)) and not allow_redirects:
            logger.info(f"Retrying with redirects enabled after error: {str(e)}")
            try:
                # Try again with redirects enabled
                return extract_text_from_html(url, output_dir, include_links, allow_redirects=True)
            except Exception as retry_e:
                # If retry also fails, report both errors
                error_msg = f"Error extracting text from HTML (both with and without redirects): {str(e)} and then {str(retry_e)}"
                logger.error(error_msg)
                
        return {"success": False, "error": error_msg}


def extract_policies_from_file(file_path: str, organization: str) -> Dict[str, Any]:
    """
    Extract structured policies from a text file and save them to a central JSON file.
    
    Args:
        file_path: Path to the file containing text to extract policies from
        organization: Name of the organization
        
    Returns:
        Dict: A dictionary containing extracted policies and metadata
    """

    # Check if file exists
    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"File does not exist: {file_path}",
            "policies": [],
            "count": 0
        }
        
    # Read text from file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    logger.info(f"Read {len(text)} characters from {file_path}")
    
    # Get file info for reference
    file_info = os.path.basename(file_path)
    
    # Get central JSON path from environment variable
    central_json_path = os.environ.get("POLICY_PATH")
    if not central_json_path:
        logger.warning("POLICY_PATH environment variable not set. Policies will only be saved locally.")
    
    system_prompt = """You are a helpful policy extraction model to identify actionable policies from organizational safety
guidelines. Your task is to exhaust all the potential policies from the provided organization handbook
which sets restrictions or guidelines for user or entity behaviors in this organization. You will extract
specific elements from the given guidelines to produce structured and actionable outputs."""

    user_prompt = f"""As a policy extraction model to clean up policies from {organization}, your tasks are:
1. Read and analyze the provided safety policies carefully, section by section.
2. Exhaust all actionable policies that are concrete and explicitly constrain behaviors.
3. For each policy, extract the following four elements:
   1. Definition: Any term definitions, boundaries, or interpretative descriptions for the policy to
      ensure it can be interpreted without any ambiguity. These definitions should be organized in a
      list.
   2. Scope: Conditions under which this policy is enforceable (e.g. time period, user group).
   3. Policy Description: The exact description of the policy detailing the restriction or guideline.
   4. Reference: All the referenced sources in the original policy article from which the policy elements
      were extracted. These sources should be organized piece by piece in a list.

Extraction Guidelines:
• Do not summarize, modify, or simplify any part of the original policy. Copy the exact descriptions.
• Ensure each extracted policy is self-contained and can be fully interpreted by looking at its Definition,
  Scope, and Policy Description.
• If the Definition or Scope is unclear, leave the value as None.
• Avoid grouping multiple policies into one block. Extract policies as individual pieces of statements.

Source document information: {file_info}

Provide the output in the following JSON format:
```json
[
  {{
    "definition": ["Exact term definition or interpretive description."],
    "scope": "Conditions under which the policy is enforceable.",
    "policy_description": "Exact description of the policy.",
    "reference": ["Original source where the elements were extracted."]
  }},
  ...
]
```

Output Requirement:
- Each policy must focus on explicitly restricting or guiding behaviors.
- Ensure policies are actionable and clear.
- Do not combine unrelated statements into one policy block.

Here are the policy texts to analyze:

{text[:100000]}"""  # Limit to 100k chars to avoid token limits

    # Initialize policies
    policies = None
    
    # Try up to 3 times to get a valid JSON response
    for i in range(3):
        try:
            response = chat_text(
                prompt=user_prompt, 
                system=system_prompt, 
                model="claude-3-7-sonnet-20250219",
                client=anthropic_client,
                max_tokens=20000, 
                temperature=0.2,
                )
    
            # Extract the JSON part from the response
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
        
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                # Validate JSON by parsing it
                policies = json.loads(json_str)
            
            break

        except Exception as e:
            logger.error(f"Error extracting policies from file: {str(e)}")
            # logger.error(f"Response: {response}")
            continue

    if policies is None:
        return {
            "success": False,
            "error": f"Failed to extract policies from {file_path}",
            "policies": [],
            "count": 0
        }
        
    # Add source file information to each policy
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for policy in policies:
        policy["source_file"] = file_path
        policy["extraction_time"] = timestamp
    
    # For very large files, handle them in chunks if needed
    if len(text) > 100000:
        logger.info(f"Large file detected ({len(text)} chars). Processing first 100k characters.")
        logger.info(f"Extracted {len(policies)} policies from the first chunk.")
    
    # Create a local file with just the extracted policies from this file
    output_dir = os.path.dirname(file_path)
    local_policies_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_policies.json")
    with open(local_policies_file, 'w', encoding='utf-8') as f:
        json.dump(policies, f, indent=2)
    
    logger.info(f"Saved extracted policies to {local_policies_file}")
    
    # Save to central JSON file if path is provided
    if central_json_path:
        all_policies = []
        
        # Load existing policies if the file exists
        if os.path.exists(central_json_path):
            try:
                with open(central_json_path, 'r', encoding='utf-8') as f:
                    all_policies = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not load existing policies from {central_json_path}. Creating new file.")
        
        # Add new policies
        all_policies.extend(policies)
        
        # Save all policies
        with open(central_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_policies, f, indent=2)
        
        logger.info(f"Added {len(policies)} policies to policy database file at {central_json_path}")
    
    # Extract policy descriptions for the text portion of the return
    policy_descriptions = [policy.get("policy_description", "No description available") for policy in policies]
    descriptions_text = "\n\n".join([f"{i+1}. {desc}" for i, desc in enumerate(policy_descriptions)])
    
    # Return structured data with all important information
    return {
        "success": True,
        "count": len(policies),
        "policies": policies,
        "policy_descriptions": policy_descriptions,
        "local_file": local_policies_file,
        "central_file": central_json_path,
        "text_output": f"""EXTRACTED {len(policies)} POLICIES FROM {file_path}:

{descriptions_text}
"""
    }





def analyze_document_section(content: str, section_info: Dict[str, Any], user_request: str, sections_found: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze a document section to determine if it contains policies and identify subsections.
    
    Args:
        content: The text content of the section
        section_info: Information about the current section
        user_request: User's request for policy extraction
        sections_found: List of all sections already found and added to the queue
            
    Returns:
        Dictionary with analysis results indicating if the section contains policies
        and if it has subsections that should be explored further
    """
    try:
        # Truncate content if too long
        max_content_length = 120000
        if len(content) > max_content_length:
            truncated_content = content[:max_content_length] + "...[TRUNCATED]"
        else:
            truncated_content = content
        
        # Format information about already found sections
        sections_found_text = ""
        if sections_found and len(sections_found) > 0:
            sections_found_text = "SECTIONS ALREADY FOUND (DO NOT SUGGEST THESE AGAIN):\n"
            for i, section in enumerate(sections_found):
                if section['type'] == 'pdf':
                    section_desc = f"PDF: {section['path']} (range: {section.get('range', '-1')})"
                else:
                    section_desc = f"HTML: {section['path']}"
                sections_found_text += f"{i+1}. {section['section_name']} - {section_desc}\n"
            sections_found_text += "\n"
        
        system_prompt = """You are a specialized policy analysis system. Your task is to analyze document sections to:
1. Determine if the current section contains concrete, extractable safety policies
2. Identify subsections or links that should be explored further for additional policies
3. You should take the user's request into account when analyzing the section and identify policy sections that correspond to the user's request.
4. You should set the has_policies to true if any possible policies are found to be relevant to the user's request.
5. IMPORTANT: You should try to suggest diverse subsections that are not likely to be duplicates of the already found sections. For example, if we already found a section with `https://redditinc.com/policies/moderator-code-of-conduct` you may not suggest `https://redditinc.com/policies/moderator-code-of-conduct?lang=en` as a subsection as they are very likely to be the same section.

Be precise and factual in your assessment. Do not exaggerate or make assumptions.
"""

        location_info = ""
        if section_info.get('type') == 'pdf':
            path = section_info.get('path', 'Unknown')
            range_val = section_info.get('range', '-1')
            location_info = f"PDF Path: {path}, Range: {range_val}"
        else:
            location_info = f"URL: {section_info.get('path', 'Unknown')}"

        user_prompt = f"""
Please analyze the following document section:

SECTION NAME: {section_info.get('section_name', 'Unknown Section')}
LOCATION: {section_info.get('type', 'Unknown Type')} - {location_info}
SOURCE: {section_info.get('source', 'Unknown Source')}
USER REQUEST: {user_request}

SECTIONS ALREADY FOUND:
{sections_found_text}

CONTENT:
{truncated_content}

Your analysis should address two key questions:

1. POLICY CONTENT ASSESSMENT: Does this section contain concrete, extractable policies? (YES/NO)
   - If YES, explain why you believe there are policies (e.g., presence of rules, guidelines, prohibitions)
   - If NO, explain why no extractable policies are present

2. SUBSECTION IDENTIFICATION: Are there additional subsections or links that should be explored? (YES/NO)
   - If YES, list each NEW subsection (that is NOT already in the list of already found sections) in the following JSON format:
     [
       {{
         "section_name": "Name of subsection",
         "path": "Path to PDF file or URL for HTML",
         "type": "pdf" or "html",
         "range": "Page range (e.g., '5-10') - ONLY for PDF type",
         "priority": 1-10 rating of importance (10 being highest),
         "source": "Where this subsection was found"
       }},
       ...
     ]
   - IMPORTANT: DO NOT suggest any subsections that are already in the sections found list.
   - If NO, explain why no further exploration is needed

DEDUPLICATION INSTRUCTIONS:
- For PDF subsections: Consider a subsection a duplicate if the same PDF path and a similar/overlapping page range exists in the visited list
- For HTML subsections: Consider a subsection a duplicate if:
  * The exact same URL exists in the visited list
  * The URL differs only in query parameters like language settings (e.g., ?lang=en)
  * The URL points to the same content but with different URL formatting

Format your response exactly as follows (including the assessment labels):

```json
{{
  "has_policies": true|false,
  "policy_assessment": "Your explanation of why this section does or doesn't contain policies",
  "has_subsections": true|false,
  "subsection_assessment": "Your explanation of why there are or aren't further subsections to explore",
  "subsections": [] or list of subsection objects as shown above
}}
```

Be objective and thorough in your analysis. Focus on identifying actual policy content rather than general information.
"""

        # Use the LLM to analyze the document section
        response_text = chat_text(
            prompt=user_prompt, 
            system=system_prompt, 
            model="claude-3-7-sonnet-20250219",
            client=anthropic_client,
            max_tokens=4000, 
            temperature=0.2,
        )
        
        # Extract JSON from the response text
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            analysis = json.loads(json_str)
            logger.info(f"Successfully analyzed section {section_info.get('section_name')}")
            return analysis
        else:
            logger.warning(f"Failed to extract JSON from analysis response")
            return {
                "has_policies": False,
                "policy_assessment": "Error in analysis response format",
                "has_subsections": False,
                "subsection_assessment": "Error in analysis response format",
                "subsections": []
            }
            
    except Exception as e:
        error_msg = f"Error during document section analysis: {str(e)}"
        logger.error(error_msg)
        return {
            "has_policies": False,
            "policy_assessment": f"Error during analysis: {str(e)}",
            "has_subsections": False,
            "subsection_assessment": f"Error during analysis: {str(e)}",
            "subsections": []
        }

if __name__ == "__main__":

    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    policy_extraction_mcp.run()
