import os
import base64
import mimetypes
from anthropic import Anthropic
from openai import OpenAI
import re
import json

# Extract the JSON part from the result
def extract_json(text):

    json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
    for block in json_blocks:
        try:
            return json.loads(block.strip())
        except:
            continue
    
    # If still no valid JSON found, raise an exception
    raise ValueError("No valid JSON found in response")

def encode_image(image_path):
    """
    Encode an image file to base64 and return with correct MIME type.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (base64_encoded_image, mime_type)
    """

    # raise error if file does not exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    # Determine correct MIME type based on file extension
    file_ext = os.path.splitext(image_path)[1].lower()
    mime_type = mimetypes.guess_type(image_path)[0]
    
    # Fallback if mimetypes doesn't recognize the extension
    if not mime_type:
        if file_ext == ".png":
            mime_type = "image/png"
        elif file_ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif file_ext == ".gif":
            mime_type = "image/gif"
        elif file_ext == ".webp":
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"  # Default fallback
    
    # Read and encode the image
    # try:
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
    return base64_image, mime_type


def chat_text(prompt, system=None, model="claude-3-7-sonnet-20250219", client=None, max_tokens=10000, temperature=0.2):
    if "claude" in model:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = response.content[0].text
    elif "openai" in model:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = response.choices[0].message.content
    else:
        raise ValueError(f"Model {model} not supported")
    
    # Parse response
    
    return result_text