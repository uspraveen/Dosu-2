# 🏗️ InfraDoc 2.0 - Complete Usage Guide

## 📁 Project Structure

Your clean, organized InfraDoc 2.0 codebase now consists of just 4 files:

```
infradoc-2.0/
├── infradoc_core.py       # Core components (SSH, LLM, Discovery)
├── infradoc_analyzer.py   # Main analysis orchestrator
├── infradoc_docs.py       # Documentation generator
└── infradoc_cli.py        # Command line interface
```

## 🚀 Installation & Setup

### 1. Prerequisites

```bash
# Install required Python packages
pip install paramiko openai anthropic

# Optional: Install additional packages for enhanced features
pip install colorlog pathlib dataclasses
```

### 2. Environment Setup

```bash
# Set up API keys for AI features (at least one required for AI analysis)
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# Optional: Set default SSH key
export SSH_KEY_FILE="/path/to/your/ssh/key.pem"
```

### 3. Verify Installation

```bash
# Test your environment
python infradoc_cli.py validate

# Test connection to server
python infradoc_cli.py validate --host server.example.com --key-file /path/to/key.pem
```

## 🎯 CLI Usage Commands

### Quick Commands (Recommended)

#### 🚀 Quick Analysis (5 minutes)
```bash
# Basic quick analysis
python infradoc_cli.py quick --host server.example.com

# With SSH key
python infradoc_cli.py quick --host server.example.com --key-file /path/to/key.pem

# With username
python infradoc_cli.py quick --host server.example.com --username ubuntu --key-file /path/to/key.pem
```

#### 🧠 Deep Analysis (15+ minutes)
```bash
# Full AI-powered deep analysis
python infradoc_cli.py deep --host server.example.com --key-file /path/to/key.pem

# Deep analysis generates:
# - Complete technical documentation
# - Security assessment
# - Architecture diagrams
# - Developer setup guides
# - Executive summaries
```

### Advanced Commands

#### 🔧 Custom Analysis
```bash
# Standard analysis with custom options
python infradoc_cli.py analyze --host server.example.com \
  --key-file /path/to/key.pem \
  --depth standard \
  --output-dir /custom/output/path

# Deep analysis without documentation
python infradoc_cli.py analyze --host server.example.com \
  --key-file /path/to/key.pem \
  --depth deep \
  --no-docs

# Analysis without AI (basic mode)
python infradoc_cli.py analyze --host server.example.com \
  --key-file /path/to/key.pem \
  --no-ai

# Security-focused analysis
python infradoc_cli.py analyze --host server.example.com \
  --key-file /path/to/key.pem \
  --depth deep \
  --max-llm-calls 25
```

#### 🔍 Environment & Testing
```bash
# Check environment and dependencies
python infradoc_cli.py validate

# Test connection to server
python infradoc_cli.py validate --host server.example.com --key-file /path/to/key.pem

# Show version information
python infradoc_cli.py version
```

## 📊 Output Structure

After analysis, you'll get a structured output directory:

```
infradoc_analysis_<timestamp>/
├── infradoc_scan_<scan_id>.json           # Complete analysis data
├── infrastructure_analysis_<scan_id>.md   # Markdown summary
├── detailed_analysis_<scan_id>.txt        # Detailed text report
└── documentation/                         # Comprehensive docs (if enabled)
    ├── README.md                          # Documentation index
    ├── executive_summary.md               # For leadership
    ├── technical_documentation.md         # For engineers
    ├── developer_guide.md                 # Setup instructions
    ├── security_report.md                 # Security analysis
    ├── architecture_documentation.md      # Architecture details
    └── infrastructure_diagrams.md         # Mermaid diagrams
```

## 🛠️ Configuration Options

### Connection Options
```bash
--host                 # Target hostname or IP (required)
--port                 # SSH port (default: 22)
--username             # SSH username (default: ubuntu)
--key-file            # SSH private key file path
--password            # SSH password (not recommended)
--timeout             # SSH timeout in seconds (default: 30)
--retries             # Connection retries (default: 3)
```

### Analysis Options
```bash
--depth               # Analysis depth: quick, standard, deep
--no-ai              # Disable AI analysis (basic mode only)
--max-llm-calls      # Maximum LLM calls (default: 15)
--output-dir         # Custom output directory
--no-artifacts       # Skip artifact generation
--no-security        # Skip security analysis
--no-docs           # Skip documentation generation
```

### Output Options
```bash
--output-formats     # Output formats: json, markdown, text
--verbose           # Verbose logging
--quiet            # Quiet mode
--debug           # Debug mode with stack traces
```

## 🎯 Example Use Cases

### 1. New Infrastructure Assessment
```bash
# First-time analysis of unknown infrastructure
python infradoc_cli.py deep --host new-server.com --key-file ~/.ssh/id_rsa

# Results: Complete documentation suite for onboarding
```

### 2. Security Audit
```bash
# Security-focused analysis
python infradoc_cli.py analyze --host production-server.com \
  --key-file ~/.ssh/prod_key.pem \
  --depth deep \
  --max-llm-calls 30

# Check: security_report.md for detailed findings
```

### 3. Architecture Documentation
```bash
# Generate architecture docs for existing system
python infradoc_cli.py analyze --host api-server.com \
  --key-file ~/.ssh/api_key.pem \
  --depth standard

# Check: architecture_documentation.md and infrastructure_diagrams.md
```

### 4. Developer Onboarding
```bash
# Create complete setup guide
python infradoc_cli.py deep --host staging-server.com --key-file ~/.ssh/staging.pem

# Share: developer_guide.md with new team members
```

### 5. Compliance Documentation
```bash
# Generate compliance-ready documentation
python infradoc_cli.py analyze --host compliance-server.com \
  --key-file ~/.ssh/compliance.pem \
  --depth deep \
  --output-dir ./compliance_docs_$(date +%Y%m%d)

# Results: Executive summary + security report + technical docs
```

## 🧠 AI Features

### LLM Provider Setup
```bash
# OpenAI (GPT-4)
export OPENAI_API_KEY="sk-..."

# Anthropic (Claude)
export ANTHROPIC_API_KEY="sk-ant-..."

# Grok (Optional)
export GROK_API_KEY="grok-..."
```

### AI Analysis Capabilities
- **Process Classification**: Intelligently categorizes running processes
- **Architecture Pattern Recognition**: Identifies microservices, monoliths, etc.
- **Security Assessment**: AI-powered vulnerability analysis
- **Technology Stack Detection**: Automatically identifies frameworks and tools
- **Integration Mapping**: Discovers service dependencies and external connections
- **Documentation Generation**: Creates human-readable documentation from technical data

## 🔧 Troubleshooting

### Common Issues

#### Connection Problems
```bash
# Test connection first
python infradoc_cli.py validate --host your-server.com --key-file /path/to/key.pem

# Check SSH key permissions
chmod 600 /path/to/your/ssh/key.pem

# Verify SSH access manually
ssh -i /path/to/key.pem ubuntu@your-server.com
```

#### Missing Dependencies
```bash
# Install missing packages
pip install paramiko openai anthropic

# Check installation
python -c "import paramiko; print('paramiko OK')"
python -c "import openai; print('openai OK')"
```

#### API Key Issues
```bash
# Verify API keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Test API access
python infradoc_cli.py validate
```

#### Permission Errors
```bash
# Create output directory with proper permissions
mkdir -p ./infradoc_output
chmod 755 ./infradoc_output

# Run with custom output directory
python infradoc_cli.py analyze --host server.com --output-dir ./infradoc_output
```

### Debug Mode
```bash
# Run with full debug information
python infradoc_cli.py analyze --host server.com --debug --verbose

# Check log file
tail -f infradoc.log
```

## 📈 Performance Tips

### Optimize Analysis Speed
```bash
# Quick analysis for rapid insights
python infradoc_cli.py quick --host server.com

# Limit LLM calls for faster execution
python infradoc_cli.py analyze --host server.com --max-llm-calls 5

# Skip documentation for analysis-only
python infradoc_cli.py analyze --host server.com --no-docs
```

### Manage Resource Usage
```bash
# Run analysis in background
nohup python infradoc_cli.py deep --host server.com > analysis.log 2>&1 &

# Monitor progress
tail -f analysis.log
tail -f infradoc.log
```

## 🔒 Security Best Practices

### SSH Security
```bash
# Use SSH keys instead of passwords
python infradoc_cli.py analyze --host server.com --key-file ~/.ssh/id_rsa

# Use dedicated analysis user
python infradoc_cli.py analyze --host server.com --username infradoc-user

# Use non-standard SSH ports
python infradoc_cli.py analyze --host server.com --port 2222
```

### API Key Security
```bash
# Store API keys securely
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc

# Use environment files
echo 'OPENAI_API_KEY=sk-...' > .env
source .env
```

## 📚 Integration Examples

### CI/CD Pipeline Integration
```bash
#!/bin/bash
# Infrastructure analysis in CI/CD

# Run analysis
python infradoc_cli.py analyze \
  --host $STAGING_SERVER \
  --key-file $SSH_KEY_FILE \
  --depth standard \
  --output-dir ./infra-docs

# Upload results to artifact storage
aws s3 sync ./infra-docs s3://docs-bucket/infrastructure/
```

### Scheduled Analysis
```bash
#!/bin/bash
# Cron job for regular infrastructure analysis

# Weekly infrastructure analysis
0 2 * * 1 /usr/bin/python /opt/infradoc/infradoc_cli.py deep \
  --host production-server.com \
  --key-file /opt/keys/prod.pem \
  --output-dir /opt/reports/$(date +\%Y\%m\%d)
```

## 🎉 Success Indicators

After running InfraDoc 2.0, you should see:

✅ **Connection Success**: Successfully connected to target host  
✅ **Process Discovery**: Found real application processes (not just system processes)  
✅ **File Discovery**: Discovered application files and configurations  
✅ **AI Analysis**: LLM calls completed with high confidence  
✅ **Documentation**: Generated comprehensive documentation suite  
✅ **Architecture**: Identified architecture patterns and technology stack  
✅ **Security**: Completed security assessment with recommendations  

## 📞 Support

### Getting Help
```bash
# Show all available commands
python infradoc_cli.py --help

# Show help for specific command
python infradoc_cli.py analyze --help

# Validate your setup
python infradoc_cli.py validate
```

### Common Commands Reference
```bash
# Quick start
python infradoc_cli.py quick --host server.com --key-file key.pem

# Full analysis
python infradoc_cli.py deep --host server.com --key-file key.pem

# Custom analysis
python infradoc_cli.py analyze --host server.com --depth deep --no-docs

# Environment check
python infradoc_cli.py validate --host server.com
```

---

🎉 **You now have a clean, organized, and powerful InfraDoc 2.0 system!**

The restructured codebase is much easier to maintain, extend, and use. All the scattered functionality has been consolidated into 4 focused files with clear responsibilities and comprehensive CLI interface.