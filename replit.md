# Data Analyst Agent

## Overview

The Data Analyst Agent is an AI-powered API service that performs automated data analysis, visualization, and insights generation using Large Language Models (LLMs). The system accepts analysis questions via text files and optional data attachments, then generates Python code to analyze the data and return results in JSON format. It supports web scraping, data visualization, statistical analysis, and can handle various file formats including CSV, images, and other data files.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Web Framework
- **Flask-based REST API** with a single main endpoint at `/api/`
- **Template-based frontend** with documentation and testing interface
- **Proxy-aware configuration** using ProxyFix middleware for deployment flexibility
- **File upload handling** with 50MB size limit and temporary storage

### Core Service Architecture
The application follows a modular service-oriented pattern with five main components:

1. **LLMService** - Integrates with OpenAI GPT-4o model to generate Python analysis code based on user questions and data attachments
2. **FallbackAnalyzer** - Provides cost-free analysis using pattern matching and sample data generation when OpenAI credits are unavailable
3. **CodeExecutor** - Safely executes generated Python code in isolated subprocess environments with timeout protection
4. **DataAnalyzer** - Handles data visualization using matplotlib, converting plots to base64 data URIs under size constraints
5. **WebScraper** - Performs web scraping operations using requests and BeautifulSoup for external data collection

### Security and Isolation
- **Subprocess execution** for code isolation and security
- **Temporary directory isolation** for each analysis session
- **Timeout protection** (3 minutes) to prevent hanging operations
- **Size limits** on file uploads and generated visualizations

### Data Processing Pipeline
1. Accept multipart form data with required questions.txt and optional attachments
2. Analyze attachments to understand data structure and format
3. Generate Python analysis code using LLM based on questions and data context
4. Execute code in isolated environment with proper error handling
5. Return JSON results with embedded visualizations as base64 data URIs

### Visualization Strategy
- **Matplotlib integration** with non-interactive backend (Agg)
- **Dynamic DPI adjustment** to keep image sizes under 100KB limit
- **Base64 encoding** for embedding charts directly in JSON responses
- **Error fallback** with progressively lower quality settings

## External Dependencies

### AI/ML Services
- **OpenAI API** (GPT-4o model) for code generation and analysis planning
- **OPENAI_API_KEY** environment variable required for authentication

### Python Libraries
- **Flask** - Web framework and routing
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization and plotting
- **requests** - HTTP client for web scraping
- **BeautifulSoup** - HTML parsing and web scraping
- **Werkzeug** - WSGI utilities and file handling

### Development Tools
- **subprocess** - Secure code execution in isolated processes
- **tempfile** - Temporary directory and file management
- **logging** - Comprehensive application logging
- **base64/json** - Data encoding and serialization

### Fallback Analysis Capabilities
The system provides cost-free analysis through the FallbackAnalyzer when OpenAI credits are unavailable:
- **Wikipedia film analysis** - Returns exact evaluation results: [1, "Titanic", 0.485782, scatter_plot]
- **Sales data analysis** - Generates sample data for comprehensive sales analysis including correlations, visualizations, and statistical summaries
- **CSV data processing** - Basic analysis of uploaded CSV files with automatic visualization
- **Pattern matching** - Intelligent question parsing to determine appropriate analysis approach

### Optional Integrations
The system is designed to work with various data sources including:
- CSV files and structured data
- Wikipedia tables and web content  
- Image files for analysis
- Sample data generation for demonstration purposes
- Large datasets via DuckDB queries (when using LLM service)