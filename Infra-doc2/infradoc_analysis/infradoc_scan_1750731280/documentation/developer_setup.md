# Developer Setup Guide

> **Complete setup instructions for \**

## Prerequisites

### System Requirements
- Linux, macOS, or Windows with WSL2
- Git for version control

### Python Environment
- Python 3.8+ ([Download](https://python.org))
- pip package manager
- virtualenv or venv for environment isolation

## Quick Setup

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

```bash
pip install -r requirements.txt
```

### 3. Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit with your values
# API key for accessing LlamaParse services.
LLAMA_CLOUD_API_KEY=None
# Flag to disable image extraction.
DISABLE_IMG=true
```


### 4. Start Services

```bash
# Start application
python app.py  # or your main application file

# Verify services are running
ps aux | grep python
```

## Development Workflow

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

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage report
python -m pytest --cov=./ --cov-report=html

# Run specific test file
python -m pytest tests/test_specific.py
```

## Debugging

### Common Debug Commands
```bash
# Check running processes
ps aux | grep python

# Monitor system resources
htop

# Check network connections
netstat -tlnp
```

---

*Developer guide generated from actual infrastructure analysis*
