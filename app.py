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
    ["https://redditinc.com/policies/reddit-rules", "Reddit", "html", "1-5", "Focus on rules that target users and customers", True, True],
    ["./policy_docs/eu_ai_act_art_5.txt", "EU AI ACT", "txt", "1-5", "", True, True],
    ["./policy_docs/eu_ai_act_art5.pdf", "EU AI ACT", "pdf", "2-8", "", False, True]
]

# Temporary directory for uploaded files
TEMP_DIR = Path("./temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

# Configure available models
MODELS = ["claude-3-7-sonnet-20250219", "gpt-4o"]

def run_extraction(document_path, organization, input_type, page_range, 
                  organization_description, target_subject, deep_policy, 
                  extract_rules, model, user_request, async_num, debug,
                  progress=gr.Progress()):
    """Run the policy extraction process"""
    progress(0, desc="Initializing...")
    
    # Create output directory
    output_dir = f"./output/gradio_{organization}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle file upload if needed
    if isinstance(document_path, dict) and "path" in document_path:
        file_path = document_path["path"]
        file_name = os.path.basename(file_path)
        
        # Copy to temp directory with appropriate extension
        dest_path = os.path.join(TEMP_DIR, file_name)
        shutil.copy2(file_path, dest_path)
        document_path = dest_path
    
    # Prepare arguments for command
    cmd_args = [
        "--document-path", document_path,
        "--organization", organization,
        "--input-type", input_type,
        "--initial-page-range", page_range,
        "--output-dir", output_dir,
        "--model", model,
        "--async-num", str(async_num),
        "--user-request", user_request
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
            )
            
            # Add a custom log handler to capture logs
            original_log = policy_agent.log
            def log_capture(message, level="INFO"):
                result["logs"].append(f"[{level}] {message}")
                original_log(message, level)
            policy_agent.log = log_capture
            
            progress(30, desc="Extracting policies...")
            
            # Extract policies
            policies = await policy_agent.extract_policies(
                document_path, input_type, page_range, deep_policy
            )
            result["policies"] = policies
            
            progress(70, desc="Policies extracted")
            
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
                    print(f"Error loading risk categories: {e}")
                
                progress(95, desc="Rules extracted and categorized")
    
    # Run the async function in a thread
    def run_thread():
        asyncio.run(run_extraction_async())
        
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
        except:
            policies_json = "No policies found"
    
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

def format_risk_categories_html(categories_data):
    """Format risk categories as HTML for better display"""
    html = "<div class='risk-categories'>"
    
    # Handle different possible structures of the categories data
    if isinstance(categories_data, dict) and "risk_categories" in categories_data:
        categories = categories_data["risk_categories"]
    elif isinstance(categories_data, list):
        categories = categories_data
    else:
        return "<p>Invalid risk categories format</p>"
    
    for i, category in enumerate(categories):
        html += f"<div class='risk-category'>"
        html += f"<h3>Category {i+1}: {category.get('name', 'Unnamed')}</h3>"
        html += f"<p><strong>Description:</strong> {category.get('description', 'No description')}</p>"
        
        # Add severity level with appropriate color
        severity = category.get('severity', 'Unknown').lower()
        severity_color = "#ff0000" if severity == "high" else "#ff9900" if severity == "medium" else "#00cc00"
        html += f"<p><strong>Severity:</strong> <span style='color: {severity_color};'>{severity.upper()}</span></p>"
        
        # Add rules
        if "rules" in category and category["rules"]:
            html += "<div class='rules'>"
            html += "<h4>Rules:</h4>"
            html += "<ul>"
            for rule in category["rules"]:
                html += f"<li><strong>Rule ID:</strong> {rule.get('rule_id', 'Unknown')}<br>"
                html += f"<strong>Description:</strong> {rule.get('rule_text', 'No description')}</li>"
            html += "</ul>"
            html += "</div>"
        else:
            html += "<p>No rules in this category</p>"
        
        html += "</div>"
    
    html += "</div>"
    return html

def handle_input_type_change(input_type):
    """Update UI based on input type selection"""
    if input_type == "html":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False)

def load_demo_example(demo_name):
    """Load a demo example based on name"""
    if not demo_name:
        return None, "", "", "", "", "", False, False
    
    # Find the matching example
    example_index = -1
    if demo_name == "Reddit Policies (HTML)":
        example_index = 0
    elif demo_name == "EU AI Act Article 5 (TXT)":
        example_index = 1
    elif demo_name == "EU AI Act Article 5 (PDF)":
        example_index = 2
    
    if example_index < 0:
        return None, "", "", "", "", "", False, False
    
    example = DEMO_EXAMPLES[example_index]
    document_path = example[0]
    
    # For URL demo
    if example[2] == "html":
        return None, document_path, example[1], example[2], example[3], example[4], example[5], example[6]
    else:
        # For file demo - we'll need to handle demo file paths specially in the extraction function
        return document_path, "", example[1], example[2], example[3], example[4], example[5], example[6]

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as app:
    gr.Markdown("""
    # 📄 GuardBench Policy Extraction Tool
    
    > Extract structured policies and rules from documents (PDF, HTML, TXT) using a systematic search tree approach.
    
    Upload a document, enter a URL, or try one of our demo examples to extract policies and rules.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Select Input Source")
            
            demo_dropdown = gr.Dropdown(
                ["Reddit Policies (HTML)", "EU AI Act Article 5 (TXT)", "EU AI Act Article 5 (PDF)"],
                label="🔍 Try a Demo Example",
                value=None
            )
            
            input_type = gr.Dropdown(
                ["pdf", "html", "txt"],
                label="📄 Input Type",
                value="pdf"
            )
            
            with gr.Group():
                file_upload = gr.File(label="📎 Upload Document (PDF/TXT)")
                url_input = gr.Textbox(label="🌐 Enter URL", visible=False)
            
            # Configuration options
            organization = gr.Textbox(label="🏢 Organization Name", value="Organization")
            org_description = gr.Textbox(label="📝 Organization Description (optional)")
            target_subject = gr.Textbox(label="👤 Target Subject", value="User")
            page_range = gr.Textbox(label="📑 Page Range (PDF only)", value="1-5")
            user_request = gr.Textbox(label="❓ User Request (optional)", placeholder="Focus on specific aspects...")
            
            with gr.Row():
                model = gr.Dropdown(MODELS, label="🧠 Model", value=MODELS[0])
                async_num = gr.Slider(1, 3, step=1, value=1, label="⚡ Parallel Tasks")
            
            with gr.Row():
                deep_policy = gr.Checkbox(label="🔍 Deep Policy Exploration", value=False)
                extract_rules = gr.Checkbox(label="📋 Extract Rules", value=True)
                debug = gr.Checkbox(label="🐛 Debug Mode", value=False)
            
            extract_btn = gr.Button("🚀 Extract Policies", variant="primary")
        
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📊 Results"):
                    with gr.Accordion("Logs", open=False):
                        logs_output = gr.Textbox(label="Execution Logs", lines=10, max_lines=20)
                    
                    gr.Markdown("### 📋 Extracted Policies")
                    policies_output = gr.JSON(label="Policies")
                    
                    gr.Markdown("### 📏 Extracted Rules")
                    rules_output = gr.JSON(label="Rules")
                    
                    gr.Markdown("### 🔍 Risk Categories")
                    categories_output = gr.HTML()
    
    # Set up event handlers
    input_type.change(
        handle_input_type_change,
        inputs=[input_type],
        outputs=[file_upload, url_input]
    )
    
    demo_dropdown.change(
        load_demo_example,
        inputs=[demo_dropdown],
        outputs=[file_upload, url_input, organization, input_type, page_range, user_request, deep_policy, extract_rules]
    )
    
    # Extract button handler with dynamic inputs based on input type
    def get_document_path(file_upload, url_input, input_type):
        if input_type == "html":
            return url_input
        else:
            return file_upload
    
    extract_btn.click(
        run_extraction,
        inputs=[
            lambda file, url, input_type=input_type: get_document_path(file, url, input_type),
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
            debug
        ],
        outputs=[policies_output, rules_output, categories_output, logs_output]
    )
    
    # Add CSS for better formatting
    gr.Markdown("""
    <style>
    .risk-categories {
        margin-top: 20px;
    }
    .risk-category {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #2c7be5;
    }
    .risk-category h3 {
        margin-top: 0;
        color: #1a1a1a;
    }
    .rules {
        margin-left: 15px;
    }
    .rules ul {
        padding-left: 20px;
    }
    .rules li {
        margin-bottom: 10px;
    }
    /* Additional styling for better UX */
    .gradio-container {
        max-width: 1200px !important;
    }
    .footer {
        margin-top: 20px;
        text-align: center;
        color: #555;
    }
    </style>
    """)
    
    # Add footer
    gr.Markdown("""
    <div class="footer">
        <p>GuardBench Policy Extraction Tool | © 2025</p>
    </div>
    """)

# Launch the app
if __name__ == "__main__":
    app.launch()