import gradio as gr
import os
import json
import asyncio
import sys
import threading
import time
from pathlib import Path
import shutil
from functools import partial

sys.path.append("./")
from policy_extractor_async import PolicyExtractionAgent, MCPServerStdio

# Define demo examples
DEMO_EXAMPLES = [
    ["https://redditinc.com/policies/reddit-rules", "Reddit", "html", "1-1000", "Focus on rules that target users and customers", False, True],
    ["/scratch/czr/GuardBench/ShieldAgent/policy_docs/eu_ai_act_art_5.txt", "EU AI ACT", "txt", "1-1000", "", False, True],
    ["/scratch/czr/GuardBench/ShieldAgent/policy_docs/eu_ai_act_art5.pdf", "EU AI ACT", "pdf", "1-1000", "", False, True]
]

# Temporary directory for uploaded files
TEMP_DIR = Path("./temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

# Configure available models
MODELS = ["claude-3-7-sonnet-20250219", "gpt-4o"]

# State variable to control ongoing extraction
extraction_active = threading.Event()

def run_extraction(openai_api_key, anthropic_api_key, file_upload, url_input, policy_text, organization, input_type, page_range, 
                  organization_description, target_subject, deep_policy, 
                  extract_rules, model, user_request, async_num, debug, exploration_budget,
                  progress=gr.Progress()):
    """Run the policy extraction process"""
    # Reset active state
    extraction_active.set()
    
    # Validate API keys
    if not openai_api_key or not anthropic_api_key:
        return None, None, "<p>Both OpenAI and Anthropic API keys are required.</p>", "Error: Missing API keys"
    
    # Set API keys as environment variables
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    
    progress(0, desc="Initializing...")
    
    # Create output directory
    output_dir = f"./output/gradio_{organization}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine document path and type based on input
    document_path = None
    
    # Special case for PDF demo
    if input_type == "pdf_demo":
        # Find the matching demo example
        demo_file = None
        for example in DEMO_EXAMPLES:
            if example[1] == organization and example[2] == "pdf":
                demo_file = example[0]
                break
        
        if demo_file and os.path.exists(demo_file):
            document_path = demo_file
            input_type = "pdf"  # Reset to normal pdf type
            print(f"Using demo PDF: {document_path}")
        else:
            return None, None, "<p>Demo PDF file not found.</p>", "Error: Demo file not found"
    
    # Handle other input types as before
    elif input_type == "html":
        document_path = url_input
    elif input_type == "text" and policy_text.strip():
        # Save text to a temporary file
        text_file = os.path.join(TEMP_DIR, f"policy_text_{int(time.time())}.txt")
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(policy_text)
        document_path = text_file
        input_type = "txt"  # Set to txt for processing
    else:
        # Handle file upload
        if file_upload is not None:
            if isinstance(file_upload, dict):
                if "path" in file_upload:  # For filepath type
                    file_path = file_upload["path"]
                elif "name" in file_upload:  # For file type
                    file_path = file_upload["name"]
                else:
                    return None, None, "<p>Invalid file format</p>", "Error: Invalid file format"
            else:
                # Handle case where file_upload is a string (this is likely what's happening)
                file_path = file_upload
            
            file_name = os.path.basename(file_path)
            
            # Validate file extension
            if not (file_name.lower().endswith('.pdf') or file_name.lower().endswith('.txt')):
                return None, None, "<p>Invalid file type. Please upload a PDF or TXT file.</p>", "Error: Invalid file type"
            
            # Copy to temp directory with appropriate extension
            dest_path = os.path.join(TEMP_DIR, file_name)
            shutil.copy2(file_path, dest_path)
            document_path = dest_path
    
    if not document_path:
        return None, None, "<p>No input provided. Please upload a file, enter a URL, or paste text.</p>", "Error: No input provided"
    
    # Prepare arguments for command
    cmd_args = [
        "--document-path", document_path,
        "--organization", organization,
        "--input-type", input_type,
        "--initial-page-range", page_range,
        "--output-dir", output_dir,
        "--model", model,
        "--async-num", str(async_num),
        "--user-request", user_request,
        "--exploration-budget", str(exploration_budget)
    ]
    
    if organization_description:
        cmd_args.extend(["--organization-description", organization_description])
    
    if target_subject:
        cmd_args.extend(["--target-subject", target_subject])
        
    if deep_policy:
        cmd_args.append("--deep-policy")
        
    if extract_rules:
        cmd_args.append("--extract-rules")
        
    if debug:
        cmd_args.append("--debug")
    
    # Set up progress monitoring
    progress(10, desc="Starting extraction...")
    
    # We'll use a thread to run the async policy extraction
    result = {"policies": None, "rules": None, "categories": None, "output_dir": output_dir, "logs": []}
    
    async def run_extraction_async():
        progress(15, desc="Initializing extraction agent...")
        
        # Initialize MCP server and policy agent
        env_vars = os.environ.copy()
        
        async with MCPServerStdio(
            name="Policy Extraction Server",
            params={
                "command": "python",
                "args": ["-m", "utility.policy_server"],
                "env": env_vars,
            },
            cache_tools_list=True,
        ) as mcp_server:
            progress(20, desc="Server initialized, starting extraction...")
            
            # Initialize the policy agent
            policy_agent = PolicyExtractionAgent(
                mcp_server=mcp_server,            
                output_dir=output_dir,
                organization=organization,
                organization_description=organization_description,
                target_subject=target_subject or "User",
                user_request=user_request,
                debug=debug,
                model=model,
                async_sections=async_num,
                exploration_budget=exploration_budget
            )
            
            # Add a custom log handler to capture logs
            original_log = policy_agent.log
            def log_capture(message, level="INFO"):
                result["logs"].append(f"[{level}] {message}")
                original_log(message, level)
            policy_agent.log = log_capture
            
            progress(30, desc="Extracting policies...")
            
            # Extract policies
            try:
                policies = await policy_agent.extract_policies(
                    document_path, input_type, page_range, deep_policy, exploration_budget
                )
                result["policies"] = policies
                
                progress(70, desc="Policies extracted")
                
                # Check if we should stop
                if not extraction_active.is_set():
                    result["logs"].append("[INFO] Extraction stopped by user")
                    return
                
                # Extract rules if requested
                if extract_rules:
                    progress(75, desc="Extracting rules...")
                    rules = await policy_agent.extract_rules(
                        policy_path=policy_agent.policies_path,
                        organization=organization,
                        organization_description=organization_description,
                        target_subject=target_subject or "User"
                    )
                    result["rules"] = rules
                    
                    # Load risk categories
                    try:
                        with open(policy_agent.risk_categories_path, 'r') as f:
                            result["categories"] = json.load(f)
                    except Exception as e:
                        result["logs"].append(f"[ERROR] Error loading risk categories: {e}")
                    
                    progress(95, desc="Rules extracted and categorized")
            except asyncio.CancelledError:
                result["logs"].append("[INFO] Extraction task was cancelled")
            except Exception as e:
                result["logs"].append(f"[ERROR] Extraction failed: {str(e)}")
                import traceback
                result["logs"].append(f"[ERROR] {traceback.format_exc()}")
    
    # Run the async function in a thread
    def run_thread():
        asyncio.run(run_extraction_async())
        extraction_active.clear()  # Clear the flag when done
        
    thread = threading.Thread(target=run_thread)
    thread.start()
    thread.join()  # Wait for the thread to complete
    
    progress(100, desc="Extraction complete!")
    
    # Load and return results
    policies_json = None
    rules_json = None
    categories_html = ""
    
    # Format policies
    if result["policies"]:
        try:
            policies_json = json.dumps(result["policies"], indent=2)
        except:
            policies_json = "Error formatting policies"
    else:
        try:
            # Try to load from file
            policies_path = os.path.join(output_dir, f"extraction_{os.path.basename(output_dir).split('_')[-1]}", f"{organization}_all_extracted_policies.json")
            if os.path.exists(policies_path):
                with open(policies_path, 'r') as f:
                    policies_json = json.dumps(json.load(f), indent=2)
                if not policies_json:
                    policies_json = "No policies were extracted"
            else:
                policies_json = "No policies were extracted"
        except:
            policies_json = "No policies were extracted"
    
    # Format rules
    if result["rules"]:
        rules_json = json.dumps(result["rules"], indent=2)
    else:
        try:
            # Try to load from file
            rules_path = os.path.join(output_dir, f"extraction_{os.path.basename(output_dir).split('_')[-1]}", f"{organization}_all_extracted_rules.json")
            if os.path.exists(rules_path):
                with open(rules_path, 'r') as f:
                    rules_json = json.dumps(json.load(f), indent=2)
        except:
            rules_json = "No rules found"
    
    # Format risk categories as HTML
    if result["categories"]:
        categories_html = format_risk_categories_html(result["categories"])
    else:
        try:
            # Try to load from file
            categories_path = os.path.join(output_dir, f"extraction_{os.path.basename(output_dir).split('_')[-1]}", f"{organization}_risk_categories.json")
            if os.path.exists(categories_path):
                with open(categories_path, 'r') as f:
                    result["categories"] = json.load(f)
                    categories_html = format_risk_categories_html(result["categories"])
        except Exception as e:
            categories_html = f"<p>No risk categories found or error loading: {str(e)}</p>"
    
    # Format logs
    logs = "\n".join(result["logs"]) if result["logs"] else "No logs captured"
    
    return policies_json, rules_json, categories_html, logs

def stop_extraction():
    """Stop the extraction process"""
    if extraction_active.is_set():
        extraction_active.clear()
        return "Stopping extraction... This may take a moment to complete."
    else:
        return "No extraction process currently running."

def format_risk_categories_html(categories_data):
    """Format risk categories as HTML for better display"""
    html = "<div class='risk-categories-container'>"
    
    # Handle different possible structures of the categories data
    if isinstance(categories_data, dict) and "risk_categories" in categories_data:
        categories = categories_data["risk_categories"]
    elif isinstance(categories_data, list):
        categories = categories_data
    else:
        return "<p>Invalid risk categories format</p>"
    
    # Add summary section
    html += f"<div class='risk-summary'>"
    html += f"<h2>Risk Analysis Summary</h2>"
    html += f"<p>{len(categories)} risk categories identified</p>"
    html += f"<p>{sum(len(category.get('rules', [])) for category in categories)} total rules extracted</p>"
    html += "</div>"
    
    # Add categories grid
    html += "<div class='risk-categories-grid'>"
    
    for i, category in enumerate(categories):
        # Determine severity and appropriate color
        severity = category.get('risk_level', 'medium').lower()
        if not severity:
            severity = "medium"  # Default if missing
            
        # Get the category name or fallback
        category_name = category.get('name', category.get('category_name', f'Category {i+1}'))
            
        severity_color = "#ff0000" if severity == "high" else "#ff9900" if severity == "medium" else "#00cc00"
        severity_class = f"severity-{severity}"
        
        html += f"<div class='risk-category-card {severity_class}'>"
        html += f"<div class='category-header' style='border-left: 5px solid {severity_color};'>"
        html += f"<h3>{category_name}</h3>"
        html += f"<span class='severity-badge' style='background-color: {severity_color};'>{severity.upper()}</span>"
        html += "</div>"
        
        # Category description
        description = category.get('description', category.get('category_rationale', 'No description available'))
        html += f"<p class='category-description'>{description}</p>"
        
        # Rules section
        if "rules" in category and category["rules"]:
            html += "<div class='rules-container'>"
            html += f"<h4>Rules ({len(category['rules'])})</h4>"
            html += "<ul class='rules-list'>"
            for rule in category["rules"]:
                rule_text = rule.get('rule_text', rule.get('rule_description', 'No description'))
                rule_id = rule.get('rule_id', 'Unknown')
                
                # Source policy information if available
                source_policies = ""
                if 'source_policy_ids' in rule:
                    policy_ids = rule['source_policy_ids']
                    if isinstance(policy_ids, list) and policy_ids:
                        source_policies = f"<span class='source-policies'>Source: Policy {', '.join(str(pid) for pid in policy_ids)}</span>"
                
                html += f"<li class='rule-item'>"
                html += f"<div class='rule-id'>Rule {rule_id}</div>"
                html += f"<div class='rule-text'>{rule_text}{source_policies}</div>"
                html += f"</li>"
            html += "</ul>"
            html += "</div>"
        else:
            html += "<p class='no-rules'>No rules in this category</p>"
        
        html += "</div>"  # Close risk-category-card
    
    html += "</div>"  # Close risk-categories-grid
    html += "</div>"  # Close risk-categories-container
    return html

def handle_input_type_change(input_type):
    """Update UI based on input type selection"""
    if input_type == "html":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    elif input_type == "text":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

def load_demo_example(demo_name):
    """Load a demo example based on name"""
    if not demo_name:
        return None, "", "", "", "", "", "", False, False
    
    # Find the matching example
    example_index = -1
    if demo_name == "Reddit Policies (HTML)":
        example_index = 0
    elif demo_name == "EU AI Act Article 5 (TXT)":
        example_index = 1
    elif demo_name == "EU AI Act Article 5 (PDF)":
        example_index = 2
    
    if example_index < 0:
        return None, "", "", "", "", "", "", False, False
    
    example = DEMO_EXAMPLES[example_index]
    document_path = example[0]
    
    # For URL demo
    if example[2] == "html":
        print(f"Loading HTML example: {document_path}")
        return None, document_path, "", example[1], example[2], example[3], example[4], example[5], example[6]
    elif example[2] == "txt" or example[2] == "pdf":
        # For file demos, check if file exists
        file_path = Path(document_path)
        absolute_path = file_path.absolute()
        print(f"Looking for file: {document_path}")
        print(f"Absolute path: {absolute_path}")
        
        if file_path.exists():
            print(f"File found! Using: {file_path}")
            return str(file_path), "", "", example[1], example[2], example[3], example[4], example[5], example[6]
        else:
            # Check a few possible locations
            alt_paths = [
                Path(f"./GuardBench/ShieldAgent/{document_path}"),
                Path(f"/scratch/czr/GuardBench/ShieldAgent/{document_path}")
            ]
            
            for alt_path in alt_paths:
                print(f"Trying alternative path: {alt_path}")
                if alt_path.exists():
                    print(f"File found at alternative path: {alt_path}")
                    return str(alt_path), "", "", example[1], example[2], example[3], example[4], example[5], example[6]
            
            print(f"Warning: Demo file not found: {document_path}")
            return None, "", "", example[1], example[2], example[3], example[4], example[5], example[6]
    else:
        return None, "", "", example[1], example[2], example[3], example[4], example[5], example[6]

def show_deep_policy_warning(deep_policy):
    """Show warning when deep policy is enabled"""
    if deep_policy:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def create_ui():
    """Create the policy extraction UI"""
    # Define custom CSS with improved styling
    custom_css = """
    .container {
        margin: 0 auto;
        max-width: 1200px;
        padding: 20px;
    }
    .header {
        text-align: center;
        margin-bottom: 30px;
        background: linear-gradient(135deg, #2c7be5, #1a53a1);
        padding: 25px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .section-container {
        margin-bottom: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        overflow: hidden;
        border: 1px solid #eaeaea;
    }
    .section-header {
        background: linear-gradient(135deg, #2c7be5, #1a53a1);
        color: white;
        padding: 15px 20px;
        font-size: 18px;
        font-weight: bold;
        margin: 0;
    }
    .section-content {
        padding: 20px;
    }
    .file-upload-area {
        border: 2px dashed #aaa;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        background-color: #f8f9fa;
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }
    .file-upload-area:hover {
        border-color: #2c7be5;
        background-color: #f0f7ff;
    }
    .control-buttons {
        display: flex;
        gap: 10px;
        margin-top: 15px;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 5px solid #ffc107;
    }
    
    /* Improved risk categories styling */
    .risk-categories-container {
        padding: 10px;
    }
    .risk-summary {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .risk-summary h2 {
        margin-top: 0;
        color: #2c7be5;
    }
    .risk-categories-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    .risk-category-card {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        border: 1px solid #eaeaea;
    }
    .risk-category-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .category-header {
        padding: 15px;
        background-color: #f8f9fa;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #eaeaea;
    }
    .category-header h3 {
        margin: 0;
        font-size: 18px;
        color: #333;
        flex: 1;
    }
    .severity-badge {
        padding: 5px 10px;
        border-radius: 20px;
        color: white;
        font-size: 12px;
        font-weight: bold;
        white-space: nowrap;
    }
    .category-description {
        padding: 15px;
        margin: 0;
        color: #555;
        font-size: 14px;
        border-bottom: 1px solid #eaeaea;
    }
    .rules-container {
        padding: 15px;
        flex: 1;
    }
    .rules-container h4 {
        margin-top: 0;
        color: #333;
        font-size: 16px;
    }
    .rules-list {
        padding-left: 0;
        list-style-type: none;
    }
    .rule-item {
        margin-bottom: 15px;
        padding: 12px;
        background-color: #f8f9fa;
        border-radius: 5px;
        border-left: 3px solid #2c7be5;
    }
    .rule-id {
        font-weight: bold;
        margin-bottom: 5px;
        color: #2c7be5;
    }
    .rule-text {
        color: #333;
        font-size: 14px;
    }
    .source-policies {
        display: block;
        margin-top: 5px;
        font-style: italic;
        color: #6c757d;
        font-size: 12px;
    }
    .no-rules {
        color: #6c757d;
        font-style: italic;
    }
    
    /* Severity-specific styling */
    .severity-high .category-header {
        background-color: #fff5f5;
    }
    .severity-medium .category-header {
        background-color: #fff9f0;
    }
    .severity-low .category-header {
        background-color: #f0fff4;
    }
    
    .results-container {
        display: flex;
        flex-direction: column;
        gap: 20px;
    }
    .result-section {
        margin-bottom: 20px;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        overflow: hidden;
        border: 1px solid #eaeaea;
    }
    .result-section-header {
        background: linear-gradient(135deg, #2c7be5, #1a53a1);
        padding: 15px;
        font-weight: bold;
        border-bottom: 1px solid #eaeaea;
        color: white;
    }
    .result-section-content {
        padding: 15px;
        color: #000000;
    }
    .policies-container {
        margin-top: 20px;
    }
    .extraction-controls {
        display: flex;
        gap: 10px;
    }
    .extraction-controls button {
        flex: 1;
        padding: 12px;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stop-button {
        background-color: #e53935 !important;
        color: white !important;
    }
    .stop-button:hover {
        background-color: #c62828 !important;
    }
    .input-options {
        margin-top: 20px;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .footer {
        margin-top: 30px;
        text-align: center;
        color: #6c757d;
        padding: 20px;
        border-top: 1px solid #eaeaea;
        background-color: #f8f9fa;
        border-radius: 0 0 10px 10px;
    }
    
    /* Additional text color fixes to ensure visibility */
    .risk-categories-section h1, 
    .risk-categories-section h2, 
    .risk-categories-section h3, 
    .risk-categories-section h4, 
    .risk-categories-section p,
    .risk-categories-section div,
    .risk-categories-section span:not(.severity-badge) {
        color: #000000;
    }
    
    /* Make sure gradio JSON and text components use black text */
    .gradio-container [data-testid="json"] pre,
    .gradio-container textarea {
        color: #000000 !important;
    }
    """
    
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=custom_css, title="Automatic Policy Risk Category Extraction") as app:
        gr.Markdown("""
        # 📄 Automatic Policy Risk Category Extraction
        
        > An agentic approach to extract structured policies, rules, and risk categories from documents (PDF, HTML, TXT).
        """, elem_classes="header")
        
        with gr.Row():
            with gr.Column(scale=1):
                # API Key inputs in the left column
                with gr.Group(elem_classes="section-container"):
                    gr.Markdown("### 🔑 API Keys", elem_classes="section-header")
                    
                    with gr.Column(elem_classes="section-content"):
                        openai_api_key = gr.Textbox(
                            label="OpenAI API Key",
                            placeholder="sk-...",
                            type="password",
                            value="",  # Empty by default
                            show_label=True,
                            info="Required for extraction"
                        )
                        
                        anthropic_api_key = gr.Textbox(
                            label="Anthropic API Key",
                            placeholder="sk-ant-...",
                            type="password",
                            value="",  # Empty by default
                            show_label=True,
                            info="Required for extraction"
                        )
                
                # Input Source Section (same as before)
                with gr.Group(elem_classes="section-container"):
                    gr.Markdown("### 📋 Input Source", elem_classes="section-header")
                    
                    with gr.Column(elem_classes="section-content"):
                        demo_dropdown = gr.Dropdown(
                            ["Reddit Policies (HTML)", "EU AI Act Article 5 (TXT)", "EU AI Act Article 5 (PDF)"],
                            label="🔍 Try a Demo Example",
                            value=None
                        )
                        
                        input_type = gr.Radio(
                            ["pdf", "html", "text"],
                            label="📄 Input Type",
                            value="pdf"
                        )
                        
                        with gr.Group(elem_classes="input-options"):
                            # PDF Upload Area
                            with gr.Group(elem_classes="file-upload-area", visible=True) as file_upload_box:
                                file_upload = gr.File(
                                    label="📎 Drag and drop your PDF or TXT file here",
                                    type="filepath",
                                    file_types=[".pdf", ".txt"]
                                )
                            
                            # URL Input Area
                            with gr.Group(visible=False) as url_input_box:
                                url_input = gr.Textbox(
                                    label="🌐 Enter URL to the policy page",
                                    placeholder="https://example.com/policy",
                                    lines=1
                                )
                            
                            # Text Input Area
                            with gr.Group(visible=False) as text_input_box:
                                policy_text = gr.TextArea(
                                    label="📝 Paste your policy text here",
                                    placeholder="Paste the policy text content here...",
                                    lines=10
                                )
                
                # Configuration Section (same as before)
                with gr.Group(elem_classes="section-container"):
                    gr.Markdown("### ⚙️ Configuration", elem_classes="section-header")
                    
                    with gr.Column(elem_classes="section-content"):
                        # Basic Configuration
                        with gr.Group():
                            organization = gr.Textbox(label="🏢 Organization Name", value="Organization")
                            org_description = gr.Textbox(label="📝 Organization Description (optional)")
                            target_subject = gr.Textbox(label="👤 Target Subject", value="User")
                            page_range = gr.Textbox(label="📑 Page Range (PDF only)", value="1-1000")
                            user_request = gr.Textbox(
                                label="❓ User Request (optional)", 
                                placeholder="Focus on specific aspects...",
                                lines=2
                            )
                        
                        # Advanced Configuration
                        with gr.Accordion("Advanced Settings", open=False):
                            with gr.Row():
                                model = gr.Dropdown(MODELS, label="🧠 Model", value=MODELS[0])
                                async_num = gr.Slider(1, 3, step=1, value=1, label="⚡ Parallel Tasks")
                            
                            with gr.Row():
                                deep_policy = gr.Checkbox(label="🔍 Deep Policy Exploration", value=False)
                                extract_rules = gr.Checkbox(label="📋 Extract Rules", value=True)
                                debug = gr.Checkbox(label="🐛 Debug Mode", value=False)
                                
                            with gr.Row():
                                exploration_budget = gr.Slider(
                                    minimum=1, 
                                    maximum=100, 
                                    step=1, 
                                    value=20, 
                                    label="📊 Exploration Budget", 
                                    info="Maximum number of sections to process in deep policy mode"
                                )
                            
                            # Warning box for deep policy
                            with gr.Group(visible=False, elem_classes="warning-box") as deep_policy_warning:
                                gr.Markdown("""
                                ⚠️ **WARNING**: Deep Policy Exploration is an experimental feature and still under testing.
                                
                                This feature may consume significantly more tokens and processing time as it explores linked documents and nested policy pages. Use with caution.
                                
                                Set the Exploration Budget to limit the maximum number of sections processed.
                                """)
                
                # Control buttons (same as before)
                with gr.Row(elem_classes="extraction-controls"):
                    extract_btn = gr.Button("🚀 Extract Policies", variant="primary")
                    stop_btn = gr.Button("🛑 Stop Extraction", elem_classes="stop-button")
                
                # Stop message display
                stop_message = gr.Textbox(label="Status", visible=False)
            
            # Results area - Modified to display sections vertically without header/bar
            with gr.Column(scale=2):
                with gr.Column(elem_classes="results-container"):
                    # Risk Categories Section
                    with gr.Group(elem_classes="result-section"):
                        gr.Markdown("#### 🔍 Risk Categories & Rules", elem_classes="result-section-header")
                        categories_output = gr.HTML(elem_classes="risk-categories-section")
                    
                    # Policies Section
                    with gr.Group(elem_classes="result-section"):
                        gr.Markdown("#### 📋 Extracted Policies", elem_classes="result-section-header")
                        policies_output = gr.JSON(elem_classes="policies-container")
                    
                    # Logs Section
                    with gr.Group(elem_classes="result-section"):
                        gr.Markdown("#### 📝 Execution Logs", elem_classes="result-section-header")
                        logs_output = gr.Textbox(lines=10)
                    
                    # Keep rules output for compatibility with existing code, but hide it in the UI
                    rules_output = gr.JSON(visible=False)
        
        # Footer
        gr.Markdown("""
        <div class="footer">
            <p>GuardBench Policy Extraction Tool | © 2025</p>
            <p>For more information, visit <a href="https://guardbench.org">guardbench.org</a></p>
        </div>
        """)
        
        # Set up event handlers
        # Input type change
        input_type.change(
            handle_input_type_change,
            inputs=[input_type],
            outputs=[file_upload_box, url_input_box, text_input_box]
        )
        
        # Add an extra handler to accommodate the special pdf_demo type
        input_type.change(
            lambda x: "pdf" if x == "pdf_demo" else x,
            inputs=[input_type],
            outputs=[input_type]
        )
        
        # Deep policy warning
        deep_policy.change(
            show_deep_policy_warning,
            inputs=[deep_policy],
            outputs=[deep_policy_warning]
        )
        
        # Demo dropdown
        demo_dropdown.change(
            load_demo_example,
            inputs=[demo_dropdown],
            outputs=[file_upload, url_input, policy_text, organization, input_type, page_range, user_request, deep_policy, extract_rules]
        )
        
        # Extract button
        extract_btn.click(
            run_extraction,
            inputs=[
                openai_api_key,
                anthropic_api_key,
                file_upload,
                url_input,
                policy_text,
                organization,
                input_type,
                page_range,
                org_description,
                target_subject,
                deep_policy,
                extract_rules,
                model,
                user_request,
                async_num,
                debug,
                exploration_budget
            ],
            outputs=[policies_output, rules_output, categories_output, logs_output]
        )
        
        # Stop button
        stop_btn.click(
            stop_extraction,
            outputs=[stop_message]
        )
    
    return app

# Launch the app
if __name__ == "__main__":
    app = create_ui()
    app.launch(share=True)