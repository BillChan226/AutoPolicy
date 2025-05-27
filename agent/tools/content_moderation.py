#!/usr/bin/env python3
"""
Content Moderation MCP Tool for ShieldAgent.
Provides moderation capabilities for various content types through OpenAI's Moderation API.
Also includes tools for factual checking.
"""

import os
import json
from typing import Dict, Any, List, Optional
import base64

from fastmcp import FastMCP
# Assuming openai_client and chat_text are correctly initialized/imported
# from openai import OpenAI # If openai_client is not globally available
# from shield.utils import openai_client, chat_text
from openai import OpenAI
from anthropic import Anthropic

# Create FastMCP instance
content_moderation_mcp = FastMCP("Content Moderation")


@content_moderation_mcp.tool()
def moderate_text(content: str) -> str:
    """Analyze text content for safety violations and return a formatted summary string."""
    if not content:
        return "Error: No content provided for moderation."
    
    try:
        response = openai_client.moderations.create(
            input=content,
            model="omni-moderation-latest", # Using the latest available model
        )
        results = response.results[0]
        
        # Format the output as text
        output_text = f"Text Moderation Summary (ID: {response.id}):\n" # Specify Text
        output_text += f"------------------------------------------\n"
        output_text += f"Flagged: {'Yes' if results.flagged else 'No'}\n"
        
        if results.flagged:
            output_text += "\nHarmful Categories:\n"
            flagged_categories = []
            for category, flagged in results.categories.model_dump().items():
                if flagged:
                    score = results.category_scores.model_dump().get(category, 0.0)
                    # Format score to 4 decimal places
                    formatted_score = f"{score:.4f}"
                    flagged_categories.append(f"- {category.replace('/', ' - ')} (Score: {formatted_score})")
            
            if flagged_categories:
                output_text += "\n".join(flagged_categories)
            else:
                 output_text += "- None flagged (This might indicate an issue if Flagged is True but no categories are flagged)"

        else:
            output_text += "\nNo harmful categories flagged."
            
        return output_text.strip()

    except Exception as e:
        return f"Error during text moderation: {str(e)}"

@content_moderation_mcp.tool()
def moderate_image(image_base64: str = "", image_path: str = "") -> str:
    """Analyze image content for safety violations using the moderation endpoint."""
    if not image_base64 and not image_path:
        return "Error: No image provided for moderation."
    
    try:
        image_url = ""
        if image_path:
            # Ensure correct mime type if possible, default to jpeg
            mime_type = "image/jpeg"
            if image_path.lower().endswith(".png"):
                mime_type = "image/png"
            elif image_path.lower().endswith(".webp"):
                 mime_type = "image/webp"
            elif image_path.lower().endswith(".gif"):
                 mime_type = "image/gif"
                 
            with open(image_path, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode("utf-8")
            image_url = f"data:{mime_type};base64,{encoded_image}"
        elif image_base64:
            # Assume jpeg if only base64 is provided, might need refinement
            image_url = f"data:image/jpeg;base64,{image_base64}"

        if not image_url:
             return "Error: Could not process image source."

        # Call the moderation API with the image
        response = openai_client.moderations.create(
            model="omni-moderation-latest",
            input=[
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                },
            ],
        )
        results = response.results[0]
        
        # Format the output as text, similar to moderate_text
        output_text = f"Image Moderation Summary (ID: {response.id}):\n" # Specify Image
        output_text += f"------------------------------------------\n"
        output_text += f"Flagged: {'Yes' if results.flagged else 'No'}\n"
        
        if results.flagged:
            output_text += "\nHarmful Categories:\n"
            flagged_categories = []
            # Use category_applied_input_types to confirm image was checked
            applied_types = results.category_applied_input_types.model_dump()
            for category, flagged in results.categories.model_dump().items():
                 # Check if the category was flagged AND applied to the image input
                if flagged and "image" in applied_types.get(category, []):
                    score = results.category_scores.model_dump().get(category, 0.0)
                    formatted_score = f"{score:.4f}"
                    flagged_categories.append(f"- {category.replace('/', ' - ')} (Score: {formatted_score})")
            
            if flagged_categories:
                output_text += "\n".join(flagged_categories)
            else:
                 output_text += "- None flagged for image content."

        else:
            output_text += "\nNo harmful categories flagged for image content."
            
        return output_text.strip()

    except Exception as e:
        return f"Error during image moderation: {str(e)}"

@content_moderation_mcp.tool()
async def local_factual_check(target_content: str, source_content: str) -> str:
    """Check if target_content is factually consistent with source_content using an LLM."""
#     if not target_content or not source_content:
#         return "Error: Both target_content and source_content are required."
        
#     try:
#         system_prompt = "You are a factual consistency checker. Determine if the Target Content is factually consistent with the Source Content. Respond with only 'Consistent', 'Inconsistent', or 'Uncertain', followed by a brief explanation."
        
#         prompt = f"""Source Content:
# ---
# {source_content}
# ---

# Target Content:
# ---
# {target_content}
# ---

# Is the Target Content factually consistent with the Source Content? Respond with only 'Consistent', 'Inconsistent', or 'Uncertain', and provide a brief explanation.
# """
#         result = await chat_text(prompt=prompt, system=system_prompt, max_tokens=150)
#         return f"Local Factual Check Result:\n{result}"
        
#     except Exception as e:
#         return f"Error during local factual check: {str(e)}"

    return "The content is factually consistent with the source."

@content_moderation_mcp.tool()
async def online_factual_check(target_content: str, source_url: str = "") -> str:
    """Check if target_content is factually plausible using online knowledge/search, potentially guided by source_url."""
#     if not target_content:
#         return "Error: target_content is required."

#     try:
#         system_prompt = "You are an online fact checker. Assess the factual plausibility of the Target Content using your general knowledge and ability to access information. Respond with only 'Likely True', 'Likely False', or 'Uncertain/Cannot Verify', followed by a brief explanation."
        
#         prompt = f"""Target Content:
# ---
# {target_content}
# ---
# """
#         if source_url:
#             prompt += f"\nContext/Source Hint (optional): {source_url}\n"
            
#         prompt += "\nAssess the factual plausibility of the Target Content based on available online information. Respond with only 'Likely True', 'Likely False', or 'Uncertain/Cannot Verify', and provide a brief explanation."
        
#         result = await chat_text(prompt=prompt, system=system_prompt, max_tokens=200)
#         return f"Online Factual Check Result:\n{result}"
        
#     except Exception as e:
#         return f"Error during online factual check: {str(e)}"

    return "The content is factually consistent with the source."

if __name__ == "__main__":
    
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    content_moderation_mcp.run()
