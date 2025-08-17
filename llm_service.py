import os
import json
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = "gpt-4o"
    
    def generate_analysis_code(self, questions, attachments):
        """Generate Python code for data analysis based on questions and attachments"""
        
        # Analyze attachments to understand data structure
        attachment_info = self._analyze_attachments(attachments)
        
        system_prompt = """You are an expert data analyst and Python programmer. Generate complete, working Python code to analyze data and answer questions.

REQUIREMENTS:
1. Return a JSON response (list or dict) matching the question format
2. For visualizations, return base64-encoded data URIs under 100KB
3. Use pandas, numpy, matplotlib, requests, BeautifulSoup as needed
4. Include proper error handling
5. Make the code self-contained and executable
6. For web scraping, use requests and BeautifulSoup
7. For plots, use matplotlib and convert to base64 data URI
8. Include all necessary imports at the top

CODE STRUCTURE:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import base64
import io
import json
# other imports as needed

# Your analysis code here
# ...

# Return result as JSON-serializable format
result = [...]  # or {...}
print(json.dumps(result))
```"""
        
        # Create user message with questions and attachment info
        user_prompt = f"""
QUESTIONS/TASKS:
{questions}

AVAILABLE DATA FILES:
{attachment_info}

Generate Python code to answer the questions using the available data. The code should:
1. Import all necessary libraries
2. Process the data files appropriately
3. Answer each question accurately
4. Return results in the requested JSON format
5. For plots, convert to base64 data URI under 100KB

Remember to use the exact filenames provided in the attachment info.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            generated_code = response.choices[0].message.content
            
            # Extract code from markdown if present
            if "```python" in generated_code:
                code_start = generated_code.find("```python") + 9
                code_end = generated_code.find("```", code_start)
                if code_end != -1:
                    generated_code = generated_code[code_start:code_end].strip()
            elif "```" in generated_code:
                code_start = generated_code.find("```") + 3
                code_end = generated_code.rfind("```")
                if code_end != -1 and code_end > code_start:
                    generated_code = generated_code[code_start:code_end].strip()
            
            logger.debug(f"Generated code: {generated_code[:500]}...")
            return generated_code
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            raise Exception(f"Failed to generate analysis code: {str(e)}")
    
    def fix_code(self, broken_code, error_message, questions, attachments):
        """Fix broken code based on error message"""
        
        attachment_info = self._analyze_attachments(attachments)
        
        fix_prompt = f"""
The following Python code failed with an error. Please fix the code to resolve the issue:

ORIGINAL CODE:
```python
{broken_code}
```

ERROR MESSAGE:
{error_message}

QUESTIONS/TASKS:
{questions}

AVAILABLE DATA FILES:
{attachment_info}

Please provide the corrected Python code that:
1. Fixes the specific error mentioned
2. Still answers all the original questions
3. Uses proper error handling
4. Returns JSON-formatted results
5. Converts plots to base64 data URIs under 100KB

Return only the corrected Python code.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            fixed_code = response.choices[0].message.content
            
            # Extract code from markdown if present
            if "```python" in fixed_code:
                code_start = fixed_code.find("```python") + 9
                code_end = fixed_code.find("```", code_start)
                if code_end != -1:
                    fixed_code = fixed_code[code_start:code_end].strip()
            elif "```" in fixed_code:
                code_start = fixed_code.find("```") + 3
                code_end = fixed_code.rfind("```")
                if code_end != -1 and code_end > code_start:
                    fixed_code = fixed_code[code_start:code_end].strip()
            
            logger.debug(f"Fixed code: {fixed_code[:500]}...")
            return fixed_code
            
        except Exception as e:
            logger.error(f"Error fixing code: {str(e)}")
            raise Exception(f"Failed to fix code: {str(e)}")
    
    def _analyze_attachments(self, attachments):
        """Analyze attachment files to understand their structure"""
        
        if not attachments:
            return "No data files provided."
        
        info_parts = []
        
        for key, filepath in attachments.items():
            try:
                filename = os.path.basename(filepath)
                file_size = os.path.getsize(filepath)
                
                info = f"- {key}: {filename} ({file_size} bytes)"
                
                # Try to get more info based on file type
                if filename.lower().endswith('.csv'):
                    try:
                        import pandas as pd
                        df = pd.read_csv(filepath, nrows=5)  # Read first 5 rows
                        info += f"\n  CSV with {len(df.columns)} columns: {list(df.columns)}"
                        info += f"\n  Sample data available for analysis"
                    except Exception as e:
                        info += f"\n  CSV file (could not preview: {str(e)})"
                
                elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    info += "\n  Image file for analysis"
                
                elif filename.lower().endswith('.txt'):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read(200)  # First 200 chars
                        info += f"\n  Text file preview: {content[:100]}..."
                    except Exception as e:
                        info += f"\n  Text file (could not preview: {str(e)})"
                
                info_parts.append(info)
                
            except Exception as e:
                info_parts.append(f"- {key}: Error analyzing file - {str(e)}")
        
        return "\n".join(info_parts)
