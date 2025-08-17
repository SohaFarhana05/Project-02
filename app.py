import os
import logging
import json
import base64
import tempfile
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from llm_service import LLMService
from code_executor import CodeExecutor
from data_analyzer import DataAnalyzer
from fallback_analyzer import FallbackAnalyzer

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure file uploads
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize services
llm_service = LLMService()
code_executor = CodeExecutor()
data_analyzer = DataAnalyzer()
fallback_analyzer = FallbackAnalyzer()

@app.route('/')
def index():
    """Main page with API documentation"""
    return render_template('index.html')

@app.route('/test')
def test_page():
    """Test page for trying the API"""
    return render_template('test.html')

@app.route('/api/', methods=['POST'])
def analyze_data():
    """Main API endpoint for data analysis"""
    try:
        logger.debug("Received POST request to /api/")
        
        # Check if questions.txt is provided
        if 'questions.txt' not in request.files:
            return jsonify({'error': 'questions.txt file is required'}), 400
        
        questions_file = request.files['questions.txt']
        if questions_file.filename == '':
            return jsonify({'error': 'questions.txt file is required'}), 400
        
        # Read questions
        questions = questions_file.read().decode('utf-8')
        logger.debug(f"Questions: {questions[:200]}...")
        
        # Process additional files
        attachments = {}
        for key, file in request.files.items():
            if key != 'questions.txt' and file.filename and file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                attachments[key] = filepath
                logger.debug(f"Saved attachment: {key} -> {filepath}")
        
        # Try to generate code using LLM, fallback to local analysis if needed
        logger.debug("Attempting analysis...")
        
        try:
            # Try LLM-based analysis first
            logger.debug("Generating code with LLM...")
            initial_code = llm_service.generate_analysis_code(questions, attachments)
            
            # Execute code with retry mechanism
            logger.debug("Executing code...")
            result = execute_with_retry(initial_code, questions, attachments)
            
        except Exception as llm_error:
            # Check if it's an API quota/rate limit error
            error_str = str(llm_error).lower()
            if any(keyword in error_str for keyword in ['quota', 'rate limit', '429', 'insufficient_quota']):
                logger.warning("LLM API quota exceeded, using fallback analyzer...")
                
                # Use fallback analyzer for basic tasks
                result = fallback_analyzer.analyze_questions(questions, attachments)
                
                # Add helpful message for quota errors
                if isinstance(result, dict) and 'error' not in result:
                    logger.info("Fallback analysis completed successfully")
                elif isinstance(result, dict) and 'error' in result:
                    # Enhance error message with quota info
                    result['quota_info'] = "OpenAI API quota exceeded. Add credits to your OpenAI account for full AI-powered analysis."
            else:
                # Re-raise other types of errors
                raise llm_error
        
        # Clean up temporary files
        cleanup_files(attachments)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in analyze_data: {str(e)}")
        # Clean up files if they exist
        if 'attachments' in locals():
            cleanup_files(attachments)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

def execute_with_retry(code, questions, attachments, max_retries=3):
    """Execute code with error handling and retry mechanism"""
    
    for attempt in range(max_retries):
        logger.debug(f"Execution attempt {attempt + 1}/{max_retries}")
        
        try:
            # Execute the code
            result = code_executor.execute_code(code, attachments)
            
            # Validate result format
            if validate_result(result):
                logger.debug("Code executed successfully")
                return result
            else:
                raise Exception("Invalid result format")
                
        except Exception as e:
            logger.warning(f"Execution attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                # Try to fix the code using LLM
                logger.debug("Attempting to fix code...")
                error_message = str(e)
                fixed_code = llm_service.fix_code(code, error_message, questions, attachments)
                code = fixed_code
            else:
                # Final attempt failed
                return {'error': f'Code execution failed after {max_retries} attempts: {str(e)}'}
    
    return {'error': 'Maximum retry attempts exceeded'}

def validate_result(result):
    """Validate that the result is in the correct format"""
    if isinstance(result, dict) and 'error' in result:
        return False
    
    # Result should be a list or dict for JSON response
    return isinstance(result, (list, dict))

def cleanup_files(attachments):
    """Clean up temporary files"""
    for filepath in attachments.values():
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"Cleaned up file: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to clean up file {filepath}: {str(e)}")

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
