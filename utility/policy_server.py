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

from fastmcp import FastMCP
from openai import OpenAI
from anthropic import Anthropic
from utility.utils import *


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastMCP instance
policy_extraction_mcp = FastMCP("Policy Extraction Server")

# Add a global variable to store the pending document sections
# Using a thread-safe queue to store the sections
document_sections_queue = queue.Queue()
visited_sections = set()  # To track which sections have already been visited

# @policy_extraction_mcp.tool()
# def extract_text_from_pdf(pdf_path: str, output_dir: str = None, page_range: str = "-1", include_links: bool = True, toc: bool = False) -> Dict[str, Any]:
#     """
#     Extract text content and links from a PDF file with optional page range and save to a file.
    
#     Args:
#         pdf_path: Path to the PDF file
#         output_dir: Directory to save the extracted text file
#         page_range: Page number to extract (e.g. "1-10, 5-20"). Use -1 for all pages,
#         include_links: Whether to include link information in the output
#         toc: Whether to include an overview of each page (i.e., table of contents) or the entire document in the output
#     Returns:
#         Dict: Information about the extraction including the file path where text was saved
#     """
#     try:
#         # Create a descriptive filename based on the PDF and page range
#         pdf_name = Path(pdf_path).stem
#         if page_range == "-1":
#             page_desc = "all_pages"
#         elif isinstance(page_range, int):
#             page_desc = f"page_{page_range}"
#         else:
#             page_desc = f"pages_{page_range}"
        
#         # Create output directory if it doesn't exist
#         if not output_dir:
#             output_dir = Path(pdf_path).parent
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Create output file path
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_file = os.path.join(output_dir, f"{pdf_name}_{page_desc}_{timestamp}.txt")
        
#         doc = fitz.open(pdf_path)
#         full_text = ""
        
#         # Determine which pages to process
#         if page_range == -1:
#             # Process all pages
#             pages_to_process = range(len(doc))
#         elif isinstance(page_range, int):
#             # Process a single page (convert to 0-based index)
#             default_range = 5
#             if page_range < 1 or page_range > len(doc):
#                 return {"error": f"Invalid page number {page_range}. The document has {len(doc)} pages."}
#             pages_to_process = range(page_range - 1, page_range + default_range)
#         elif isinstance(page_range, str) and "-" in page_range:
#             # Process a range of pages
#             try:
#                 start, end = map(int, page_range.split("-"))
#                 if start < 1 or end > len(doc) or start > end:
#                     # return {"error": f"Invalid page range {page_range}. The document has {len(doc)} pages."}
#                     end = len(doc)
#                     logger.warning(f"Invalid page range {page_range}. The document has {len(doc)} pages. Adjusting end to {end}.")
#                 pages_to_process = range(start - 1, end)  # Convert to 0-based index
#             except ValueError:
#                 return {"error": f"Invalid page range format '{page_range}'. Use format like '5-10'."}
#         else:
#             return {"error": f"Invalid page_range parameter. Use -1 for all pages, a single number, or a range like '5-10'."}
        
#         # Get TOC items for contextual information
#         toc = doc.get_toc()
#         toc_by_page = {}
        
#         for item in toc:
#             level, title, page_num = item
#             if page_num not in toc_by_page:
#                 toc_by_page[page_num] = []
#             toc_by_page[page_num].append({"level": level, "title": title})
        
#         all_links = []
#         # Process each page
#         for page_idx in pages_to_process:
#             page = doc[page_idx]
#             page_num = page_idx + 1  # 1-based page number for output
            
#             # Add page header with TOC information if available
#             if page_num in toc_by_page:
#                 headings = toc_by_page[page_num]
#                 full_text += f"\n--- Page {page_num} Headings ---\n"
#                 for heading in headings:
#                     full_text += f"{'  ' * (heading['level']-1)}• {heading['title']}\n"
            
#             # Extract text from the page
#             page_text = page.get_text()
#             full_text += f"\n--- Page {page_num} Content ---\n{page_text}\n"
            
#             # Extract links from the page and include them with this page
#             if include_links:
#                 links = page.get_links()
#                 if links:
#                     full_text += f"\n--- Page {page_num} Links ---\n"
#                     for i, link in enumerate(links):
#                         if 'uri' in link:
#                             full_text += f"Link {i+1}: Web link: {link['uri']}\n"
#                             all_links.append({
#                                 'url': link['uri'],
#                                 'text': f"Link {i+1}: Web link: {link['uri']}"
#                             })
#                         elif 'page' in link:
#                             target_page = link['page'] + 1  # Convert 0-based to 1-based
#                             full_text += f"Link {i+1}: Internal link to page {target_page}\n"
#                             all_links.append({
#                                 'url': f"page_{target_page}",
#                                 'text': f"Link {i+1}: Internal link to page {target_page}"
#                             })
        
#         # Write the extracted text to the output file
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write(full_text)
            
#         logger.info(f"Saved extracted PDF text to {output_file}")
        
#         preview_count = 1000
#         # Get a preview of the content (first 200 chars)
#         preview = full_text[:preview_count] + "..." if len(full_text) > preview_count else full_text
        
#         # Return structured data with file path directly accessible
#         result = {
#             "success": True,
#             "file_path": output_file,
#             "source": pdf_path,
#             "pages": page_desc,
#             "total_pages_processed": len(list(pages_to_process)),
#             "preview": preview,
#             "message": f"""PDF TEXT EXTRACTION COMPLETE:
# - Source: {pdf_path}
# - Pages: {page_desc}
# - Total pages processed: {len(list(pages_to_process))}
# - Full page content saved to: {output_file}
# - Page content preview: {preview}

# If needed, USE THIS FILE PATH to extract policies: {output_file}"""
#         }
        
#         return result
    
#     except Exception as e:
#         error_msg = f"Error extracting text from PDF: {str(e)}"
#         logger.error(error_msg)
#         return {"success": False, "error": error_msg}
    

@policy_extraction_mcp.tool()
def extract_text_from_pdf(pdf_path: str, output_dir: str = None, page_range: str = "-1", include_links: bool = True, toc: bool = False) -> Dict[str, Any]:
    """
    Extract text content and links from a PDF file with optional page range and save to a file.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the extracted text file
        page_range: Page number to extract (e.g. "1-10, 5-20"). Use -1 for all pages,
        include_links: Whether to include link information in the output
        toc: Whether to include an overview of each page (i.e., table of contents) or the entire document in the output
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
                    # return {"error": f"Invalid page range {page_range}. The document has {len(doc)} pages."}
                    end = len(doc)
                    logger.warning(f"Invalid page range {page_range}. The document has {len(doc)} pages. Adjusting end to {end}.")
                pages_to_process = range(start - 1, end)  # Convert to 0-based index
            except ValueError:
                return {"error": f"Invalid page range format '{page_range}'. Use format like '5-10'."}
        else:
            return {"error": f"Invalid page_range parameter. Use -1 for all pages, a single number, or a range like '5-10'."}
        
        # Get TOC items for contextual information
        toc_items = doc.get_toc()
        toc_by_page = {}
        
        for item in toc_items:
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
            
            # Check if we're doing TOC-only mode (page summary with headers)
            if toc:
                # For TOC mode, only include page headers and top-level headings
                full_text += f"\n=== PAGE {page_num} SUMMARY ===\n"
                
                # Extract text from the page
                page_text = page.get_text()
                
                # Try to identify headers (# and ##) in the text
                lines = page_text.split("\n")
                headers = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Look for potential headings based on formatting
                    # For h1/h2 level headers: ALL CAPS, Title Case, or numerical prefixes like "1.2"
                    if (len(line) < 100 and line.upper() == line and len(line) > 3) or \
                       (len(line) < 80 and line[0].isupper() and ":" in line) or \
                       (re.match(r"^\d+(\.\d+)*\s+[A-Z]", line) and len(line) < 80):
                        # Could be a header - add as a level 1 or 2 heading
                        if len(line) < 50 or line.upper() == line:  # Shorter or ALL CAPS = level 1
                            headers.append(f"# {line}")
                        else:
                            headers.append(f"## {line}")
                
                # Add the headers to the text
                if headers:
                    full_text += "\n".join(headers) + "\n"
                    full_text += f"Page content:\n{page_text}\n"
                else:
                    full_text += "(No clear headers found on this page)\n"
                
                # If there are TOC entries for this page, add them as well
                if page_num in toc_by_page:
                    headings = toc_by_page[page_num]
                    if headings and not headers:  # Only add if we didn't find headers in the text
                        full_text += "\nFrom TOC:\n"
                        for heading in headings:
                            level_marker = "#" * min(heading['level'], 2)  # Limit to ## level
                            full_text += f"{level_marker} {heading['title']}\n"
                
            else:
                # Regular mode - extract full text
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
            
        logger.info(f"Saved extracted PDF text to {output_file} (TOC mode: {toc})")
        
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
            "toc_mode": toc,
            "message": f"""PDF TEXT EXTRACTION COMPLETE:
- Source: {pdf_path}
- Pages: {page_desc}
- Total pages processed: {len(list(pages_to_process))}
- Mode: {'Page summaries (TOC mode)' if toc else 'Full text'}
- Full page content saved to: {output_file}
- Page content preview: {preview}

If needed, USE THIS FILE PATH to extract policies: {output_file}"""
        }
        
        return result
    
    except Exception as e:
        error_msg = f"Error extracting text from PDF: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}



@policy_extraction_mcp.tool()
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

@policy_extraction_mcp.tool()
def extract_policies_from_file(file_path: str, organization: str, organization_description: str, target_subject: str, policy_db_path: str, user_request: str = "") -> Dict[str, Any]:
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
        
    system_prompt = """You are a helpful policy extraction model to identify actionable policies from organizational safety guidelines. Your task is to extract all the meaningful policies from the provided organization handbook which sets restrictions or guidelines for user or entity behaviors in this organization. You will extract specific elements from the given policies and guidelines to produce structured and actionable outputs. Meanwhile, you should follow the user's request and extract policies accordingly."""
    
    user_prompt = f"""As a policy extraction model to extract and clean up useful policies from {organization} ({organization_description}), your tasks are:
1. Read and analyze the provided safety policy document (e.g. likely a PDF handbook or HTML website). Specifically, this document may contain irrelevant information such as structure text, headers, footers, etc. However, you should focus on meaningful policies that constrain the behaviors of the target subject {target_subject}.
2. Exhaust all meaningful policies that are concrete and explicitly constrain the behaviors of {target_subject}. You should carefully analyze what are the target audience or subject for each policy, and avoid extracting policies that are not targeting {target_subject}. For example, if the target subject is "user" or "customer" of the organization, you should avoid extracting policies that target "developer" or "employee" of the organization.
3. Extract each individual policy separately into a policy block, where you should try to use the original text from the document as much as possible. Avoid paraphrasing or generalizing the original text that may change the original meaning.
- For each individual policy, extract the following four elements in a block:
   1) Definitions: Any term definitions, boundaries, or interpretative descriptions for the policy to ensure it can be interpreted without any ambiguity. These definitions should be organized in a list.
   2) Scope: Conditions under which this policy is enforceable (e.g. time period, user group).
   3) Policy Description: The exact description of the policy detailing the restriction or guideline targeting {target_subject}.
   4) Reference: All the referenced sources in the original policy article from which the policy elements were extracted. These sources should be organized piece by piece in a list.
4. If the user has provided an additional request, you should follow the user's request and extract policies accordingly. If not, you should extract all the meaningful policies from the document.

USER REQUEST: {user_request}

Here is the document to extract policies from:

---Start of Document---
{text[:150000]}
---End of Document---

**Output format**:
Provide the output in the following JSON format:
```json
[
  {{
    "definitions": ["A list of term definitions or interpretive descriptions."],
    "scope": "Conditions under which the policy is enforceable.",
    "policy_description": "Exact description of the individual policy targeting {target_subject}.",
    "reference": ["A list of original sources where the policy can be traced back to."]
  }},
  ...
]
```
"""

    # Initialize policies
    policies = None
    
    # Try up to 3 times to get a valid JSON response
    for _ in range(5):
        try:
            # logger.info(f"user_prompt: {user_prompt}")
            response = chat_text(
                prompt=user_prompt, 
                system=system_prompt, 
                model="claude-3-7-sonnet-20250219",
                client=anthropic_client,
                max_tokens=20000, 
                temperature=0.2,
                )

            # with open(f"response_{file_info}.txt", "w", encoding="utf-8") as f:
            #     f.write(response)

            # logger.info(f"user_prompt: {user_prompt}")

            # logger.info(f"Internal response: {response}")
            # input("Press Enter to continue...")
            # Extract the JSON part from the response
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
        
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                # Validate JSON by parsing it
                policies = json.loads(json_str)
                logger.info(f"Extracted {len(policies)} policies from {file_path}")
            
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
    for idx, policy in enumerate(policies):
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
    if policy_db_path:
        all_policies = []
        num_existing_policies = 0
        # Load existing policies if the file exists
        if os.path.exists(policy_db_path):
            try:
                with open(policy_db_path, 'r', encoding='utf-8') as f:
                    all_policies = json.load(f)
                    num_existing_policies = len(all_policies)
            except json.JSONDecodeError:
                logger.warning(f"Could not load existing policies from {policy_db_path}. Creating new file.")
        
        # Add new policies
        all_policies.extend(policies)

        policy_counter = num_existing_policies
        for local_policy, global_policy in zip(policies, all_policies):
            global_policy["policy_id"] = policy_counter
            local_policy["policy_id"] = policy_counter
            policy_counter += 1
        
        # Save all policies
        with open(policy_db_path, 'w', encoding='utf-8') as f:
            json.dump(all_policies, f, indent=2)
        
        logger.info(f"Added {len(policies)} policies to policy database file at {policy_db_path}")
    
    
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
        "policy_db_path": policy_db_path,
        "text_output": f"""EXTRACTED {len(policies)} POLICIES FROM {file_path}:

{descriptions_text}
"""
    }


@policy_extraction_mcp.tool()
def extract_rules_from_policies(policy_path: str, organization: str, organization_description: str = "", target_subject: str = "User", output_path: str = None) -> Dict[str, Any]:
    """
    Extract concrete rules from structured policies and maintain references to source policies.
    
    Args:
        policy_path: Path to the JSON file containing structured policies
        organization: Name of the organization
        organization_description: Description of the organization
        target_subject: Target subject of the policies (e.g., "User")
        output_path: Path to save the extracted rules (default: same directory as policy file with _rules suffix)
        
    Returns:
        Dictionary with extraction results
    """
    try:
        # Load policies from JSON file
        with open(policy_path, 'r', encoding='utf-8') as f:
            policies = json.load(f)
            
        # If policies is not a list, handle that case
        if not isinstance(policies, list):
            return {
                "success": False,
                "error": f"Expected a list of policies, got {type(policies)}"
            }
            
        # Sanity check that policy_idx is present for each policy
        for idx, policy in enumerate(policies):
            if "policy_id" not in policy:
                assert False, "Policy index not found. You must provide a policy database with policy indices."
            
        # Generate the rule extraction prompt
        all_policy_text = ""
        for idx, policy in enumerate(policies):
            policy_id = policy.get("policy_id", idx)
            term_definitions = policy.get("definitions", [])
            policy_description = policy.get("policy_description", "")
            policy_scope = policy.get("scope", "")
            
            all_policy_text += f"Policy #{policy_id}:\n"
            if term_definitions:
                all_policy_text += f"Terminology Definitions: {term_definitions}\n"
            if policy_scope:
                all_policy_text += f"Policy Scope: {policy_scope}\n"
            all_policy_text += f"Policy Description: {policy_description}\n"
            all_policy_text += "\n---\n\n"
        
        system_prompt = """You are an expert rule extraction model to extract actionable rules from individual policy descriptions extracted from a given organization's handbook or policy documents.
        """

        # Create the rule extraction prompt
        extraction_prompt = f"""You are tasked to extract actionable rules from the policies of the organization {organization} ({organization_description}). The rules you extract will be used to evaluate compliance with {organization}'s policies.

The individual policies extracted from the organization handbook or policy documents are presented below, each with a unique identifier (Policy #X):

{all_policy_text}

**Your task**:
- Extract concrete, actionable rules from each policy. You should focus on the rules that constrain the behaviors of the target subject "{target_subject}".
- Each rule should be specific, actionable, and clearly state what the target subject "{target_subject}" must or must not do. Note that you should only include constraints on the behaviors of the target subject "{target_subject}", not other entities. For example, if the target subject is "user" or "customer" of the organization, you should avoid extracting rules that target "developer" or "employee" of the organization.
- Each rule must be concrete and directly indicate a desired behavior or action of the target subject "{target_subject}". Avoid vague or abstract language; be specific and actionable.
- Each rule description should explicitly include the applicable scope and subject of the rule.
- Complex policies may be decomposed into multiple atomic rules - a single policy can be the source for multiple rules.
- Each extracted rule must align with the original policy including its full scope, intent, and descriptions. Do not paraphrase or generalize the original text that may change the original meaning!
- For each rule, provide:
   - rule_description: A clear, concise statement of the rule
   - source_policy_idx: The Policy # reference(s) this rule is derived from

Note that some complex or compound policy blocks can be decomposed into several rules,
and each rule may reference one or more source policies.

Output format:
Return a JSON array of rules, where each rule has the following structure:
```json
[
  {{
    "rule_description": <string>,
    "source_policy_idx": [<integer>, ...]
  }},
  ...
]
```

Your output should be valid JSON only, with no additional text or explanation.
"""

        for _ in range(3):
            try:
                response_text = chat_text(
                    prompt=extraction_prompt, 
                    system=system_prompt, 
                    model="claude-3-7-sonnet-20250219",
                    client=anthropic_client,
                max_tokens=20000, 
                temperature=0.2,
            )
            
                # Extract JSON from the response text
                json_start = response_text.find("[")
                json_end = response_text.rfind("]") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    rules = json.loads(json_str)
                else:
                    raise Exception("No valid JSON found in the response")

                break

            except Exception as e:
                logger.error(f"Error extracting rules from policies: {str(e)}")
                continue

        logger.info(f"Extracted {len(rules)} rules from {len(policies)} policies")
        
        # Add policy information to each rule
        for rule_idx, rule in enumerate(rules):
            rule["rule_id"] = rule_idx + 1
            source_policy_indices = rule.get("source_policy_idx", [])
            if isinstance(source_policy_indices, int):
                source_policy_indices = [source_policy_indices]
                
            rule["source_policies"] = []
            for idx in source_policy_indices:
                if 0 <= idx < len(policies):
                    # policy_info = {
                    #     "policy_idx": idx,
                    #     "definitions": policies[idx].get("definitions", ""),
                    #     "scope": policies[idx].get("scope", ""),
                    #     "description": policies[idx].get("policy_description", ""),
                    # }
                    # rule["source_policies"].append(policy_info)
                    rule["scope"] = policies[idx].get("scope", "")
                    rule["term_definitions"] = policies[idx].get("definitions", "")
                    rule["source_policy_description"] = policies[idx].get("policy_description", "")
                    # rule.pop("source_policy_idx")
        
        # Determine output file path if not provided
        if not output_path:
            output_dir = os.path.dirname(policy_path)
            file_name = os.path.basename(policy_path)
            file_base, file_ext = os.path.splitext(file_name)
            output_path = os.path.join(output_dir, f"{file_base}_rules{file_ext}")
            
        # Save the extracted rules
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=2)
            
        return {
            "success": True,
            "message": f"Successfully extracted {len(rules)} rules from {len(policies)} policies",
            "rules_count": len(rules),
            "policies_count": len(policies),
            "output_path": output_path,
            "rules": rules
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@policy_extraction_mcp.tool()
def get_pdf_toc(pdf_path: str) -> str:
    """
    Extract a structured representation of the PDF document with headings organized by page.
    Focus on higher-level headings (# and ##) for better navigation.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Structured representation of the PDF with headings by page
    """
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Extract the formal TOC
        formal_toc = doc.get_toc()
        
        # Format the formal TOC
        toc_markdown = "# DOCUMENT FORMAL TABLE OF CONTENTS\n\n"
        
        if formal_toc:
            for item in formal_toc:
                level, title, page = item
                indent = "  " * (level - 1)
                toc_markdown += f"{indent}- [{title}] (page {page})\n"
        else:
            toc_markdown += "No formal table of contents found in the document.\n"
        
        # Extract headings page by page
        page_headings = {}
        
        # First, generate markdown for the entire document to work with
        markdown_content = pymupdf4llm.to_markdown(pdf_path)
        
        # Process the document page by page to extract structured content
        for page_idx in range(len(doc)):
            page_num = page_idx + 1  # Convert to 1-based page numbering
            page = doc[page_idx]
            
            # Extract text from the page
            page_text = page.get_text()
            
            # Create a soup from the markdown content to identify headings
            # This is a basic approach - for a more accurate extraction, 
            # we'd need to analyze the PDF structure more deeply
            page_headings[page_num] = []
            
            # Look for potential headings in the text
            lines = page_text.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for potential headings based on formatting
                # This is a heuristic approach - headings are often shorter and may have certain formats
                if len(line) < 100 and line.upper() == line and len(line) > 3:
                    # Likely a top-level heading (ALL CAPS)
                    page_headings[page_num].append({"level": 1, "text": line})
                elif len(line) < 80 and line[0:1].isupper() and ":" in line and len(line) > 10:
                    # Likely a second-level heading (Title: format)
                    page_headings[page_num].append({"level": 2, "text": line})
        
        # Include any formal TOC entries as well
        for item in formal_toc:
            level, title, page = item
            if level <= 2:  # Only consider level 1 and 2 headings from the TOC
                if page not in page_headings:
                    page_headings[page] = []
                
                # Check if this heading is already in our list
                if not any(h["text"] == title for h in page_headings[page]):
                    page_headings[page].append({"level": level, "text": title})
        
        # Format the extracted headings by page
        page_headings_markdown = "# DOCUMENT HEADINGS BY PAGE\n\n"
        
        # For each page, include its headings
        for page_num in sorted(page_headings.keys()):
            if page_headings[page_num]:  # Only include pages that have headings
                page_headings_markdown += f"## PAGE {page_num}\n\n"
                
                for heading in page_headings[page_num]:
                    level = heading["level"]
                    text = heading["text"]
                    
                    if level == 1:
                        page_headings_markdown += f"# {text}\n"
                    elif level == 2:
                        page_headings_markdown += f"## {text}\n"
                
                page_headings_markdown += "\n"  # Add space between pages
        
        # Include basic document information
        doc_info = f"""
# PDF DOCUMENT INFORMATION

- Filename: {os.path.basename(pdf_path)}
- Total Pages: {len(doc)}
- Has Formal TOC: {"Yes" if formal_toc else "No"}
- Pages with Identified Headings: {len([p for p in page_headings if page_headings[p]])}

"""
        
        # Generate a structured outline of the document content
        structured_content = "# DOCUMENT STRUCTURE OVERVIEW\n\n"
        
        policy_keywords = [
            "chapter", "article", "policy", "section", "rule", "guideline",
            "standard", "procedure", "regulation", "part", "title"
        ]
        # Create a hierarchical structure of headings
        all_headings = []
        for page_num, headings in page_headings.items():
            for heading in headings:
                if any(keyword in heading["text"].lower() for keyword in policy_keywords):
                    all_headings.append({
                        "level": heading["level"],
                        "text": heading["text"],
                        "page": page_num
                    })
        
        # Sort headings by page
        all_headings.sort(key=lambda x: x["page"])
        
        # Format the headings hierarchically
        current_level_1 = None
        for idx, heading in enumerate(all_headings):
            if heading["level"] == 1:
                structured_content += f"# {heading['text']} (From page {heading['page']} to page {all_headings[idx+1]['page'] if idx+1 < len(all_headings) else 'end of document'})\n\n"
        
        # Combine everything into a comprehensive document structure
        full_output = doc_info + toc_markdown + "\n\n" + structured_content + "\n\n" + page_headings_markdown
        
        logger.info(f"Successfully extracted structured headings from {pdf_path}")
        return full_output
    
    except Exception as e:
        error_msg = f"Error extracting PDF structure: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"

@policy_extraction_mcp.tool()
def get_html_site_structure(url: str) -> str:
    """
    Extract the site structure from an HTML page.
    
    Args:
        url: URL of the HTML page
        
    Returns:
        str: JSON string of the site structure
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract page title
        title = soup.title.string if soup.title else "No title"
        
        # Extract headings to understand document structure
        headings = []
        for i in range(1, 7):  # h1 to h6
            for heading in soup.find_all(f'h{i}'):
                headings.append({
                    "level": i,
                    "text": heading.get_text(strip=True)
                })
        
        # Extract navigation links
        nav_links = []
        for nav in soup.find_all(['nav', 'div'], class_=re.compile(r'nav|menu|header')):
            for a in nav.find_all('a', href=True):
                link = a['href']
                if not link.startswith(('http://', 'https://')):
                    base_url = urlparse(url)
                    if link.startswith('/'):
                        link = f"{base_url.scheme}://{base_url.netloc}{link}"
                    else:
                        link = urljoin(url, link)
                
                nav_links.append({
                    "text": a.get_text(strip=True),
                    "url": link
                })
        
        # Extract main policy-related links
        policy_links = []
        keywords = ['policy', 'policies', 'terms', 'privacy', 'guidelines', 'rules', 'regulation']
        for a in soup.find_all('a', href=True):
            link_text = a.get_text(strip=True).lower()
            if any(keyword in link_text for keyword in keywords):
                link = a['href']
                if not link.startswith(('http://', 'https://')):
                    base_url = urlparse(url)
                    if link.startswith('/'):
                        link = f"{base_url.scheme}://{base_url.netloc}{link}"
                    else:
                        link = urljoin(url, link)
                
                policy_links.append({
                    "text": a.get_text(strip=True),
                    "url": link
                })
        
        # Build structured output
        structure = {
            "title": title,
            "headings": headings,
            "navigation_links": nav_links,
            "policy_links": policy_links
        }
        
        return json.dumps(structure, indent=2)
    
    except Exception as e:
        error_msg = f"Error extracting HTML structure: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"



@policy_extraction_mcp.tool()
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
        max_content_length = 160000
        if len(content) > max_content_length:
            truncated_content = content[:max_content_length] + "...[TRUNCATED]"
        else:
            truncated_content = content
        # truncated_content = content
        
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
        
        # with open(f"response.txt", "w", encoding="utf-8") as f:
        #     f.write(response_text)

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

@policy_extraction_mcp.tool()
def refine_and_categorize_rules(rules_path: str, organization: str, organization_description: str = "", target_subject: str = "User", output_path: str = None) -> Dict[str, Any]:
    """
    Refine and categorize rules extracted from policies into structured risk categories.
    
    Args:
        rules_path: Path to the JSON file containing extracted rules
        organization: Name of the organization/platform
        organization_description: Brief description of the organization/platform
        target_subject: The target subject of the rules (default: "User")
        output_path: Path to save the categorized rules (if None, will use rules_path with _categorized suffix)
        
    Returns:
        Dictionary with results of the categorization process
    """
    try:
        logger.info(f"Refining and categorizing rules from {rules_path}")
        
        # Load rules from JSON file
        if not os.path.exists(rules_path):
            return {
                "success": False,
                "error": f"Rules file not found: {rules_path}"
            }
            
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
            
        if not rules:
            return {
                "success": False,
                "error": "No rules found in the input file"
            }
            
        # Format rules as a numbered list for the prompt
        all_rules_text = ""
        for idx, rule in enumerate(rules):
            # Handle different possible rule formats
            if isinstance(rule, dict):
                rule_desc = rule.get("rule_description", "")
                rule_id = rule.get("rule_id", idx+1)  # Use index+1 as fallback rule_id
            else:
                # Handle non-dictionary rules
                rule_desc = str(rule)
                rule_id = idx+1
                
            all_rules_text += f"{rule_id}. {rule_desc}\n"
            
        # Create the refine and categorize prompt
        system_prompt = """You are an expert rule categorization system designed to analyze safety rules and organize them into clear, structured risk categories. You excel at identifying patterns, removing redundancy, and creating meaningful taxonomies that capture the full range of safety concerns."""

        categorization_prompt = f"""You are given a numbered list of safety rules extracted from a safety policy document for the platform/organization {organization}.
{organization_description}

The target subject of the rules is "{target_subject}".

Some rules may be overly broad, contain multiple sub-parts, or overlap with others in meaning. Your task is to process these rules to produce a concise, well-organized, and non-redundant set of safety principles grouped by clearly defined safety risk categories.

🔧 **Your Tasks**

1. Decompose Complex Rules
- Identify rules that include multiple safety ideas or conditions.
- Break them into atomic (single-action or single-concern) rules.
- Ensure each rule is specific and cannot be split further without losing meaning.

2. Merge Redundant or Similar Rules
- Identify rules that are semantically similar or convey overlapping concepts.
- Combine them into a single unified rule that preserves all important details.

3. Cluster into Risk Categories
- Organize the refined rules into meaningful safety categories (e.g., Harassment, Hate Speech, Privacy Violations).
- Each category should capture a distinct type of safety concern relevant to {organization} governed by a distinct set of rules.
- Each risk category should be unique and try to overlap with the other categories as little as possible.
- You should not merge two clusters as one risk category if they are different types of safety concerns. You can keep as many risk categories as you want as long as they are distinct and representative of different types of safety concerns.

4. Refine and Standardize Wording
- Use clear, professional language for all rules.
- Ensure each rule is concise, precise, and consistently formatted.
- Avoid vague, overly broad, or compound statements.
- Format rules from the perspective of "{target_subject} must/must not..." where appropriate.

5. Assign Risk Level to Each Risk Category
- Assign a risk level to each risk category based on the severity of the risks it covers. You should consider both the severity of the risks and the potential impact of the risks on {organization} and {target_subject}.
- Use the following scale: [low, medium, high]

🧾 **Input**
A raw, numbered list of safety rules (may include overlapping, vague, or compound rules):

{all_rules_text}

✅ **Expected Output Format**
Your response must be VALID JSON with the following structure:

```json
[
  {{
    "category_name": "Descriptive Name of the Risk Category",
    "category_rationale": "Brief explanation of which risks this category covers for {organization} and {target_subject}",
    "risk_level": "low, medium, high",
    "rules": [
      {{
        "rule_description": "Refined rule text",
        "original_rule_ids": [1, 5, 9], // Array of original rule numbers that contributed to this rule
      }},
      ...more rules in this category...
    ]
  }},
  ...more categories...
]
```

⚠️ **Important Instructions**
- Do not omit any safety concept from the original list.
- Each final rule must be atomic (irreducible further).
- No semantically redundant rules should remain.
- Ensure the categories and rule interpretations are relevant to the behaviors typical on {organization}.
"""

        # Process the rules with the LLM
        for _ in range(3):
            try:
                response_text = chat_text(
                    prompt=categorization_prompt, 
                    system=system_prompt, 
                    model="claude-3-7-sonnet-20250219",
                    client=anthropic_client,
                    max_tokens=20000, 
                    temperature=0.2,
                )
                
                # Extract JSON from the response text
                json_start = response_text.find("[")
                json_end = response_text.rfind("]") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    categorized_rules = json.loads(json_str)
                else:
                    raise Exception("No valid JSON found in the response")
                    
                break
                
            except Exception as e:
                logger.error(f"Error categorizing rules: {str(e)}")
                continue
        
        # Update the rule processing in refine_and_categorize_rules function
        rules_count = 0
        for category in categorized_rules:
            for rule in category["rules"]:
                # Keep the rule_id assignment
                rule["rule_id"] = f"{rules_count+1}"
                
                # Make sure each rule has source_policy_id/source_policy_ids if it came from original rules
                if "original_rule_ids" in rule and rule["original_rule_ids"]:
                    # Get the original rules from which this refined rule was derived
                    original_rule_ids = rule["original_rule_ids"]
                    
                    # Create an array to track all source policy IDs
                    source_policy_ids = set()
                    
                    # If we have mapping of original rules to source policies, use it
                    if isinstance(original_rule_ids, list):
                        for orig_id in original_rule_ids:
                            for i, orig_rule in enumerate(rules):
                                # Make sure orig_rule is a dictionary and has the needed fields
                                if isinstance(orig_rule, dict):
                                    rule_id_str = str(orig_rule.get("rule_id", i+1))
                                    if rule_id_str == str(orig_id):
                                        # Found the original rule, get its policy ID
                                        if "source_policy_idx" in orig_rule:
                                            source_ids = orig_rule.get("source_policy_idx")
                                            if isinstance(source_ids, list):
                                                source_policy_ids.update(source_ids)
                                            else:
                                                source_policy_ids.add(source_ids)
                                        elif "source_policy_id" in orig_rule:
                                            source_id = orig_rule.get("source_policy_id")
                                            if source_id is not None:
                                                source_policy_ids.add(source_id)

                    if source_policy_ids:
                        rule["source_policy_ids"] = list(source_policy_ids)
                
                rules_count += 1

        # Determine output file path if not provided
        if not output_path:
            output_dir = os.path.dirname(rules_path)
            file_name = os.path.basename(rules_path)
            file_base, file_ext = os.path.splitext(file_name)
            output_path = os.path.join(output_dir, f"{file_base}_categorized{file_ext}")
            
        # Save the categorized rules
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(categorized_rules, f, indent=2)
            
        # Count stats for reporting
        total_categories = len(categorized_rules)
        total_refined_rules = sum(len(category["rules"]) for category in categorized_rules)
        
        return {
            "success": True,
            "message": f"Successfully categorized {len(rules)} rules into {total_categories} risk categories with {total_refined_rules} refined rules",
            "categories_count": total_categories,
            "rules_count": total_refined_rules,
            "original_rules_count": len(rules),
            "output_path": output_path,
            "categorized_rules": categorized_rules
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":

    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    policy_extraction_mcp.run()
