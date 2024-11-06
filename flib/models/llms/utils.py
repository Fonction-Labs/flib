import re
import json

def clean_json_output(llm_output):
    """
    Extract and parse JSON from `llm_output` string.
    """
    json_regex = re.compile(r'```json(.*?)```', re.DOTALL)
    match = json_regex.search(llm_output)
    if not match:
        raise ValueError("No JSON found in the provided string.")
    json_str = match.group(1).strip()
    
    try:
        json_output = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON: {e}")

    return json_output