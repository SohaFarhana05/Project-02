import subprocess
import json
import tempfile
import os
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr
import io

logger = logging.getLogger(__name__)

class CodeExecutor:
    def __init__(self):
        self.timeout = 180  # 3 minutes timeout
    
    def execute_code(self, code, attachments):
        """Execute Python code safely and return the result"""
        
        try:
            # Create a temporary directory for execution
            with tempfile.TemporaryDirectory() as temp_dir:
                
                # Write code to a temporary file
                code_file = os.path.join(temp_dir, "analysis.py")
                
                # Prepare code with proper imports and attachment paths
                full_code = self._prepare_code(code, attachments, temp_dir)
                
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(full_code)
                
                logger.debug(f"Executing code in {temp_dir}")
                
                # Execute the code using subprocess for security
                result = subprocess.run(
                    [sys.executable, code_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=temp_dir
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr or "Unknown execution error"
                    logger.error(f"Code execution failed: {error_msg}")
                    raise Exception(f"Execution error: {error_msg}")
                
                # Parse the output as JSON
                output = result.stdout.strip()
                logger.debug(f"Code output: {output[:500]}...")
                
                if not output:
                    raise Exception("No output generated")
                
                # Try to parse as JSON
                try:
                    parsed_result = json.loads(output)
                    return parsed_result
                except json.JSONDecodeError as e:
                    # If not valid JSON, try to extract JSON from output
                    lines = output.split('\n')
                    for line in reversed(lines):
                        line = line.strip()
                        if line.startswith('[') or line.startswith('{'):
                            try:
                                parsed_result = json.loads(line)
                                return parsed_result
                            except:
                                continue
                    
                    raise Exception(f"Output is not valid JSON: {str(e)}")
                
        except subprocess.TimeoutExpired:
            raise Exception("Code execution timed out (3 minutes)")
        except Exception as e:
            logger.error(f"Code execution error: {str(e)}")
            raise
    
    def _prepare_code(self, code, attachments, temp_dir):
        """Prepare code with proper imports and file paths"""
        
        # Copy attachment files to temp directory and update paths
        attachment_vars = {}
        for key, original_path in attachments.items():
            filename = os.path.basename(original_path)
            new_path = os.path.join(temp_dir, filename)
            
            # Copy file to temp directory
            import shutil
            shutil.copy2(original_path, new_path)
            
            # Create variable name from key
            var_name = key.replace('.', '_').replace('-', '_')
            attachment_vars[var_name] = filename
        
        # Prepare imports and attachment variables
        imports = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import base64
import io
import json
import sqlite3
import scipy.stats as stats
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')
plt.style.use('default')
"""
        
        # Add attachment file variables
        file_vars = ""
        for var_name, filename in attachment_vars.items():
            file_vars += f"{var_name}_path = '{filename}'\n"
        
        # Combine everything
        full_code = imports + "\n" + file_vars + "\n" + code
        
        # Ensure the code prints JSON output
        if "print(json.dumps(" not in full_code and "print(" not in full_code:
            full_code += "\n\n# Ensure JSON output\nif 'result' in locals():\n    print(json.dumps(result))\n"
        
        return full_code
