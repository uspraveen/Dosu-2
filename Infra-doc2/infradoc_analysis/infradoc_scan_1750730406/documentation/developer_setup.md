# 🔧 Developer Setup Guide

> **Complete setup instructions for \**

## 📋 Prerequisites

### System Requirements
- Linux, macOS, or Windows with WSL2
- Git for version control

### Python Environment
- Python 3.8+ ([Download](https://python.org))
- pip package manager
- virtualenv or venv for environment isolation

### Web Server
- Nginx (for local testing)

### Development Tools
- Code editor (VS Code recommended)
- Terminal/Command prompt
- Browser for testing


## 🚀 Quick Setup

### 1. Environment Preparation

```bash
# Clone the repository
git clone <repository-url>
cd \

# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

```

### 2. Install Dependencies

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nginx

# macOS
brew install nginx
```


### 3. Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit with your values
# API key for accessing LlamaParse services.
LLAMA_CLOUD_API_KEY_2=None
# Flag to disable image extraction.
DISABLE_IMG=true
# AWS access key for S3 operations
AWS_ACCESS_KEY_ID=None
# AWS secret key for S3 operations
AWS_SECRET_ACCESS_KEY=None
# AWS region for S3
AWS_DEFAULT_REGION=us-east-2
# API key for OpenAI services
OPENAI_API_KEY=None
# URI for Neo4j database connection
NEO4J_URI=None
# Username for Neo4j
NEO4J_USER=neo4j
# Password for Neo4j
NEO4J_PASSWORD=None
```


### 4. Database Setup

*Database setup will be documented if database models are detected.*

### 5. Start Services

### Start Application Services

```bash
# Start web server
sudo systemctl start nginx

# Start application
python app.py  # or your main application file

# Start background workers
python worker.py &

# Verify services are running
ps aux | grep python
```

## 🧪 Development Workflow


### Typical Development Workflow

1. **Pull Latest Changes**
   ```bash
   git pull origin main
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Edit code following existing patterns
   - Add tests for new functionality
   - Update documentation if needed

4. **Test Changes**
   ```bash
   # Run tests
   python -m pytest
   
   # Check code quality
   flake8 .
   black --check .
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Open PR for code review
   - Address review feedback
   - Merge after approval


## 🧩 Testing


### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage report
python -m pytest --cov=./ --cov-report=html

# Run specific test file
python -m pytest tests/test_specific.py

# Run with verbose output
python -m pytest -v
```

### Test Types

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **API Tests**: Test API endpoints (if applicable)
- **Performance Tests**: Test system performance

### Writing Tests

Follow the existing test patterns:
- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names
- Include docstrings for complex tests


## 🐛 Debugging


### Debugging Techniques

#### Application Logs
```bash
# View application logs
tail -f /var/log/app/application.log

# Check system logs
journalctl -u your-service-name -f
```

#### Process Monitoring
```bash
# Check running processes
ps aux | grep python

# Monitor system resources
htop

# Check network connections
netstat -tlnp
```

#### Common Debug Commands
```bash
# Python debugging
python -m pdb your_script.py

# Check Python path and modules
python -c "import sys; print(sys.path)"

# Validate configuration
python -c "from config import settings; print(settings)"
```

#### IDE Setup
- Set breakpoints in your code editor
- Use built-in debugger
- Configure remote debugging if needed


## 🔄 Common Development Tasks

### Common Development Tasks

#### Adding New API Endpoint
1. Define endpoint in appropriate module
2. Add route handler function
3. Update API documentation
4. Add tests for new endpoint

#### Database Changes
1. Update model definitions
2. Create migration files
3. Test migration on development database
4. Update API documentation if schema changes

#### Adding Background Jobs
1. Define job function
2. Add job to worker queue
3. Test job execution
4. Monitor job performance


## 🔍 Code Structure

### Code Organization

The codebase follows these patterns:

- **learnchain/**: worker.py, parsing_adapter.py, worker-2.py
  and 1 more files

### Coding Standards
- Follow PEP 8 for Python code
- Use descriptive variable and function names
- Add docstrings to functions and classes
- Keep functions focused and small
- Use type hints where appropriate

## 📝 Contributing


### Contributing Guidelines

#### Before Contributing
- Read through existing code to understand patterns
- Check for existing issues or feature requests
- Discuss major changes before implementation

#### Code Style
- Run `black .` to format code
- Run `flake8 .` to check for issues
- Follow existing naming conventions
- Add tests for new features

#### Pull Request Process
1. Create feature branch from `main`
2. Make changes with descriptive commits
3. Ensure all tests pass
4. Update documentation if needed
5. Submit pull request with clear description

#### Getting Help
- Check existing documentation
- Ask questions in team chat
- Review similar implementations in codebase


---

*🔧 Developer guide generated from actual infrastructure analysis*
