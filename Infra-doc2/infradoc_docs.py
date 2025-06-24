#!/usr/bin/env python3
"""
InfraDoc 2.0 - Intelligent Documentation Generator
Generates beautiful, comprehensive documentation that developers actually want to read.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from datetime import datetime

# Configure logging with Windows-safe format (no emojis)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntelligentDocumentationGenerator:
    """Generate intelligent, beautiful documentation from enhanced analysis."""
    
    def __init__(self, scan_report, output_dir: str):
        """Initialize the intelligent documentation generator."""
        self.scan_report = scan_report
        self.output_dir = Path(output_dir)
        self.docs_dir = self.output_dir / "documentation"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract intelligence for better documentation
        self.project_intelligence = self._analyze_project_intelligence()
        self.api_intelligence = self._extract_api_intelligence()
        self.business_intelligence = self._extract_business_intelligence()
        self.deployment_intelligence = self._extract_deployment_intelligence()
        
        logger.info(f"[DOCS] Intelligent Documentation Generator initialized: {self.docs_dir}")
    
    def generate_intelligent_documentation(self) -> bool:
        """Generate complete intelligent documentation suite."""
        try:
            logger.info("[DOCS] Generating intelligent documentation suite")
            
            # Generate comprehensive documentation
            self._generate_intelligent_readme()
            self._generate_api_documentation()
            self._generate_developer_setup_guide()
            self._generate_architecture_deep_dive()
            self._generate_business_logic_guide()
            self._generate_deployment_runbook()
            self._generate_security_assessment()
            self._generate_troubleshooting_guide()
            self._generate_monitoring_guide()
            self._generate_performance_guide()
            self._generate_enhanced_executive_summary()
            self._generate_documentation_index()
            
            logger.info("[DOCS] Intelligent documentation generated successfully")
            return True
            
        except Exception as e:
            logger.error(f"[DOCS] Intelligent documentation generation failed: {e}")
            return False
    
    def _analyze_project_intelligence(self) -> Dict:
        """Analyze project to extract intelligence for documentation."""
        # Determine project type and characteristics
        has_web_api = bool(self.scan_report.api_documentation and self.scan_report.api_documentation.get('endpoints'))
        has_business_logic = bool(self.scan_report.business_intelligence)
        has_workers = any('worker' in p.command.lower() for p in self.scan_report.processes)
        has_web_server = any('nginx' in p.command.lower() for p in self.scan_report.processes)
        
        # Determine project name from file paths
        common_paths = [f.path for f in self.scan_report.application_files]
        project_name = "Application"
        if common_paths:
            # Extract most common directory name
            path_parts = []
            for path in common_paths:
                parts = Path(path).parts
                for part in parts:
                    if part not in ['opt', 'srv', 'var', 'www', 'home', 'usr', 'etc']:
                        path_parts.append(part)
            
            if path_parts:
                from collections import Counter
                most_common = Counter(path_parts).most_common(1)
                if most_common:
                    project_name = most_common[0][0].replace('_', ' ').title()
        
        # Determine project type
        if has_web_api and has_workers:
            project_type = "Full-Stack Web Service"
            description = "Web API service with background processing capabilities"
        elif has_web_api:
            project_type = "Web API Service"
            description = "RESTful API service for web and mobile applications"
        elif has_workers:
            project_type = "Background Processing Service"
            description = "Background job processing and task management system"
        elif has_web_server:
            project_type = "Web Application"
            description = "Web application with server-side rendering"
        else:
            project_type = "Application Service"
            description = "Custom application service"
        
        # Extract technology stack
        tech_stack = self.scan_report.infrastructure_insights.technology_stack or []
        
        # Add detected technologies from files
        languages = set()
        frameworks = set()
        for file_info in self.scan_report.application_files:
            if file_info.language and file_info.language != 'Unknown':
                languages.add(file_info.language)
            
            # Detect frameworks from imports
            if hasattr(file_info, 'imports') and file_info.imports:
                for imp in file_info.imports:
                    if any(fw in imp.lower() for fw in ['flask', 'django', 'fastapi']):
                        frameworks.add(imp)
        
        tech_stack.extend(list(languages))
        tech_stack.extend(list(frameworks))
        tech_stack = list(set(tech_stack))  # Remove duplicates
        
        return {
            'project_name': project_name,
            'project_type': project_type,
            'description': description,
            'tech_stack': tech_stack,
            'has_web_api': has_web_api,
            'has_business_logic': has_business_logic,
            'has_workers': has_workers,
            'has_web_server': has_web_server
        }
    
    def _extract_api_intelligence(self) -> Dict:
        """Extract API intelligence for documentation."""
        if not hasattr(self.scan_report, 'api_documentation') or not self.scan_report.api_documentation:
            return {}
        
        api_docs = self.scan_report.api_documentation
        endpoints = api_docs.get('endpoints', [])
        
        # Group endpoints by functionality
        endpoint_groups = {}
        for endpoint in endpoints:
            path = endpoint.get('path', '')
            # Extract resource from path (e.g., /api/users -> users)
            path_parts = [p for p in path.split('/') if p and p != 'api']
            resource = path_parts[0] if path_parts else 'general'
            
            if resource not in endpoint_groups:
                endpoint_groups[resource] = []
            endpoint_groups[resource].append(endpoint)
        
        return {
            'endpoint_groups': endpoint_groups,
            'total_endpoints': len(endpoints),
            'base_url': api_docs.get('base_url', 'http://localhost'),
            'models': api_docs.get('models', [])
        }
    
    def _extract_business_intelligence(self) -> Dict:
        """Extract business intelligence for documentation."""
        if not hasattr(self.scan_report, 'business_intelligence') or not self.scan_report.business_intelligence:
            return {}
        
        bi = self.scan_report.business_intelligence
        return {
            'domain': bi.get('business_domain', 'Unknown'),
            'purpose': bi.get('application_purpose', ''),
            'functions': bi.get('primary_business_functions', []),
            'workflows': bi.get('critical_workflows', []),
            'data_flows': bi.get('data_flows', []),
            'integrations': bi.get('integration_architecture', {})
        }
    
    def _extract_deployment_intelligence(self) -> Dict:
        """Extract deployment intelligence for documentation."""
        deployment_info = {
            'processes': len(self.scan_report.processes),
            'files': len(self.scan_report.application_files),
            'services': [],
            'config_files': [],
            'environment_vars': [],
            'security_notes': []
        }
        
        # Extract services from processes
        for process in self.scan_report.processes:
            if process.service_classification in ['web_server', 'application', 'background_worker']:
                deployment_info['services'].append({
                    'name': process.name,
                    'type': process.service_classification,
                    'user': process.user,
                    'command': process.command
                })
        
        # Extract environment variables from enhanced files
        for file_info in self.scan_report.application_files:
            if hasattr(file_info, 'environment_variables') and file_info.environment_variables:
                deployment_info['environment_vars'].extend(file_info.environment_variables)
        
        # Extract security concerns
        for file_info in self.scan_report.application_files:
            if hasattr(file_info, 'security_concerns') and file_info.security_concerns:
                deployment_info['security_notes'].extend(file_info.security_concerns)
        
        return deployment_info
    
    def _generate_intelligent_readme(self):
        """Generate an intelligent README that tells the complete story."""
        logger.info("[DOCS] Generating intelligent README")
        
        pi = self.project_intelligence
        
        content = f"""# {pi['project_name']}

> **{pi['project_type']}** - {pi['description']}

## Quick Start

```bash
{self._generate_quick_start_commands()}
```

## What This System Does

{self._generate_system_overview()}

## Architecture Overview

{self._generate_architecture_overview_table()}

{self._generate_api_section()}

{self._generate_business_logic_section()}

## Project Structure

```
{self._generate_intelligent_project_structure()}
```

## Configuration

{self._generate_configuration_table()}

## Health & Monitoring

{self._generate_health_monitoring_table()}

## Development

### Quick Links
- [Developer Setup Guide](./developer_setup.md) - Complete setup instructions
- [API Documentation](./api_documentation.md) - Full API reference  
- [Business Logic Guide](./business_logic.md) - Understanding the business
- [Deployment Guide](./deployment_runbook.md) - Production deployment
- [Security Assessment](./security_assessment.md) - Security considerations
- [Troubleshooting Guide](./troubleshooting.md) - Common issues and solutions

### Development Workflow
1. **Setup**: Follow the [Developer Setup Guide](./developer_setup.md)
2. **Code**: Make your changes following the established patterns
3. **Test**: Run the test suite (see Testing section below)
4. **Deploy**: Use the deployment runbook for production

### Testing
```bash
{self._generate_testing_commands()}
```

## Common Issues

{self._generate_troubleshooting_quick_reference()}

## Performance

{self._generate_performance_overview()}

---

**Auto-generated by InfraDoc 2.0** - *"Developers just develop, we'll document"*  
*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

*This documentation was automatically generated by analyzing the actual running infrastructure. 
It reflects the real system architecture, APIs, and deployment patterns.*
"""
        
        file_path = self.docs_dir / "README.md"
        file_path.write_text(content, encoding='utf-8')
        logger.info(f"[DOCS] Intelligent README: {file_path}")
    
    def _generate_quick_start_commands(self) -> str:
        """Generate intelligent quick start commands based on actual deployment."""
        commands = []
        
        # Check for Python virtual environment
        has_python = any('python' in f.path.lower() for f in self.scan_report.application_files)
        has_requirements = any('requirements' in f.path.lower() for f in self.scan_report.application_files)
        
        if has_python:
            commands.extend([
                "# Clone and setup",
                "git clone <repository-url>",
                "cd " + self.project_intelligence['project_name'].lower().replace(' ', '-'),
                ""
            ])
            
            if has_requirements:
                commands.extend([
                    "# Setup Python environment",
                    "python3 -m venv venv",
                    "source venv/bin/activate  # On Windows: venv\\Scripts\\activate",
                    "pip install -r requirements.txt",
                    ""
                ])
        
        # Check for Docker
        has_docker = any('docker' in f.path.lower() for f in self.scan_report.application_files)
        if has_docker:
            commands.extend([
                "# Or start with Docker",
                "docker-compose up -d",
                ""
            ])
        
        # Add service startup commands
        if self.deployment_intelligence['services']:
            commands.extend([
                "# Start services",
                "sudo systemctl start application-service",
                ""
            ])
        
        commands.extend([
            "# Verify everything is running",
            "curl http://localhost:8000/health",
            "",
            "# View logs",
            "tail -f /var/log/app/application.log"
        ])
        
        return "\n".join(commands) if commands else "# Setup commands will be added based on project analysis"
    
    def _generate_system_overview(self) -> str:
        """Generate system overview based on business intelligence."""
        if self.business_intelligence:
            overview = f"""
### Business Purpose
{self.business_intelligence['purpose']}

### Key Capabilities
"""
            for func in self.business_intelligence['functions'][:5]:
                overview += f"- **{func}**\n"
            
            if self.business_intelligence['workflows']:
                overview += f"""
### Critical Workflows
"""
                for workflow in self.business_intelligence['workflows'][:3]:
                    overview += f"- {workflow}\n"
            
            return overview
        else:
            return f"""
This {self.project_intelligence['project_type'].lower()} provides core functionality for the application ecosystem.
The system handles {len(self.scan_report.processes)} processes and manages {len(self.scan_report.application_files)} application files.

*Business intelligence analysis will be added in future scans.*
"""
    
    def _generate_architecture_overview_table(self) -> str:
        """Generate architecture overview in table format."""
        insights = self.scan_report.infrastructure_insights
        
        table = f"""
| Component | Details |
|-----------|---------|
| **Architecture Pattern** | {insights.architecture_pattern} |
| **Deployment Model** | {insights.deployment_model} |
| **Technology Stack** | {', '.join(insights.technology_stack)} |
| **Processes Running** | {len(self.scan_report.processes)} active processes |
| **Application Files** | {len(self.scan_report.application_files)} files analyzed |
| **Scalability** | {insights.scalability_assessment} |
| **Security Posture** | {insights.security_posture} |
| **Operational Complexity** | {insights.operational_complexity} |
"""
        return table
    
    def _generate_api_section(self) -> str:
        """Generate API section if APIs are available."""
        if not self.api_intelligence:
            return ""
        
        api_section = f"""
## API Reference

This service exposes **{self.api_intelligence['total_endpoints']} API endpoints** organized by functionality:

### Quick API Reference
| Method | Endpoint | Description |
|--------|----------|-------------|
"""
        
        # Show top endpoints from each group
        for group_name, endpoints in self.api_intelligence['endpoint_groups'].items():
            for endpoint in endpoints[:2]:  # Show first 2 from each group
                method = endpoint.get('method', 'GET')
                path = endpoint.get('path', '/')
                description = endpoint.get('description', f'{group_name} operations')
                api_section += f"| `{method}` | `{path}` | {description} |\n"
        
        api_section += f"""
**Base URL**: `{self.api_intelligence['base_url']}`

**[Complete API Documentation →](./api_documentation.md)**
"""
        
        return api_section
    
    def _generate_business_logic_section(self) -> str:
        """Generate business logic section if available."""
        if not self.business_intelligence:
            return ""
        
        section = f"""
## Business Logic

**Domain**: {self.business_intelligence['domain']}

### Core Business Functions
"""
        for func in self.business_intelligence['functions'][:3]:
            section += f"- {func}\n"
        
        if self.business_intelligence['data_flows']:
            section += f"""
### Data Flow
"""
            for flow in self.business_intelligence['data_flows'][:3]:
                from_src = flow.get('from', 'Unknown')
                to_dest = flow.get('to', 'Unknown')
                description = flow.get('description', '')
                section += f"- **{from_src}** → **{to_dest}**: {description}\n"
        
        section += f"""
**[Detailed Business Logic Guide →](./business_logic.md)**
"""
        
        return section
    
    def _generate_intelligent_project_structure(self) -> str:
        """Generate intelligent project structure based on actual files."""
        structure = {}
        
        # Build tree structure from actual files
        for file_info in self.scan_report.application_files:
            path_parts = Path(file_info.path).parts
            current = structure
            
            # Skip system directories
            if any(sys_dir in path_parts for sys_dir in ['/usr', '/var/lib', '/proc']):
                continue
            
            for part in path_parts[:-1]:  # Exclude filename
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Add file with metadata
            filename = Path(file_info.path).name
            business_summary = ""
            if hasattr(file_info, 'business_logic_summary'):
                business_summary = file_info.business_logic_summary[:50] + "..." if len(file_info.business_logic_summary) > 50 else file_info.business_logic_summary
            
            current[filename] = {
                'type': 'file',
                'language': file_info.language,
                'purpose': business_summary
            }
        
        # Convert to string representation
        return self._structure_to_string(structure, max_depth=4)
    
    def _structure_to_string(self, structure: Dict, indent: int = 0, max_depth: int = 10) -> str:
        """Convert structure dict to tree string with depth limit."""
        if indent > max_depth:
            return "  " * indent + "├── ..."
        
        result = []
        items = list(structure.items())
        
        for i, (name, content) in enumerate(items):
            is_last = i == len(items) - 1
            prefix = "  " * indent + ("└── " if is_last else "├── ")
            
            if isinstance(content, dict):
                if content.get('type') == 'file':
                    # File with metadata
                    purpose = f" # {content['purpose']}" if content['purpose'] else ""
                    result.append(f"{prefix}{name} ({content['language']}){purpose}")
                else:
                    # Directory
                    result.append(f"{prefix}{name}/")
                    if indent < max_depth:
                        child_structure = self._structure_to_string(content, indent + 1, max_depth)
                        if child_structure:
                            result.append(child_structure)
            else:
                result.append(f"{prefix}{name}")
        
        return "\n".join(filter(None, result))
    
    def _generate_configuration_table(self) -> str:
        """Generate configuration table with environment variables."""
        env_vars = self.deployment_intelligence.get('environment_vars', [])
        
        if not env_vars:
            return "*No configuration variables detected. Configuration will be added based on deployment analysis.*"
        
        table = """
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
"""
        
        seen_vars = set()
        for var in env_vars:
            var_name = var.get('variable_name', 'Unknown')
            if var_name in seen_vars:
                continue
            seen_vars.add(var_name)
            
            description = var.get('usage_context', var.get('description', ''))
            required = 'Yes' if var.get('is_required', True) else 'No'
            default = var.get('default_value', 'None')
            
            table += f"| `{var_name}` | {description} | {required} | `{default}` |\n"
        
        return table
    
    def _generate_health_monitoring_table(self) -> str:
        """Generate health and monitoring information table."""
        table = """
| Check | Endpoint/Command | Expected Response |
|-------|------------------|-------------------|
"""
        
        # Add API health check if available
        if self.api_intelligence:
            table += f"| **API Health** | `GET {self.api_intelligence['base_url']}/health` | `200 OK` |\n"
        
        # Add service checks
        for service in self.deployment_intelligence.get('services', []):
            service_name = service.get('name', 'unknown')
            table += f"| **{service_name} Service** | `systemctl status {service_name}` | `active (running)` |\n"
        
        # Add basic system checks
        table += "| **Disk Space** | `df -h` | < 80% usage |\n"
        table += "| **Memory** | `free -h` | Available memory > 1GB |\n"
        table += "| **Processes** | `ps aux | grep python` | Application processes running |\n"
        
        return table
    
    def _generate_testing_commands(self) -> str:
        """Generate testing commands based on project type."""
        commands = []
        
        # Check for Python testing frameworks
        has_python = any('python' in f.path.lower() for f in self.scan_report.application_files)
        
        if has_python:
            commands.extend([
                "# Run tests",
                "python -m pytest",
                "",
                "# Run with coverage",
                "python -m pytest --cov=./",
                ""
            ])
        
        # API testing if APIs are available
        if self.api_intelligence:
            commands.extend([
                "# Test API endpoints",
                f"curl -X GET {self.api_intelligence['base_url']}/health",
                ""
            ])
        
        # Add linting
        if has_python:
            commands.extend([
                "# Code quality checks",
                "flake8 .",
                "black --check .",
                ""
            ])
        
        return "\n".join(commands) if commands else "# Testing commands will be added based on project analysis"
    
    def _generate_troubleshooting_quick_reference(self) -> str:
        """Generate quick troubleshooting reference."""
        issues = [
            "**Service Won't Start**: Check `systemctl status service-name` and logs in `/var/log/`",
            "**API Not Responding**: Verify process is running with `ps aux | grep python`",
            "**Permission Errors**: Check file ownership and permissions",
            "**Port Already in Use**: Find process with `netstat -tlnp | grep :port`"
        ]
        
        # Add specific issues from security concerns
        security_notes = self.deployment_intelligence.get('security_notes', [])
        for note in security_notes[:2]:
            if isinstance(note, dict):
                concern = note.get('description', '')
                recommendation = note.get('recommendation', '')
                if concern and recommendation:
                    issues.append(f"**{concern}**: {recommendation}")
        
        return "\n".join(f"- {issue}" for issue in issues[:5])
    
    def _generate_performance_overview(self) -> str:
        """Generate performance overview."""
        overview = f"""
| Metric | Current Status |
|--------|----------------|
| **Active Processes** | {len(self.scan_report.processes)} processes |
| **Application Files** | {len(self.scan_report.application_files)} files |
| **Architecture Complexity** | {self.scan_report.infrastructure_insights.operational_complexity} |
| **Scalability Assessment** | {self.scan_report.infrastructure_insights.scalability_assessment} |
"""
        
        # Add performance notes from enhanced files
        performance_notes = []
        for file_info in self.scan_report.application_files:
            if hasattr(file_info, 'performance_notes') and file_info.performance_notes:
                performance_notes.extend(file_info.performance_notes)
        
        if performance_notes:
            overview += f"""
### Performance Considerations
"""
            for note in performance_notes[:3]:
                overview += f"- {note}\n"
        
        return overview
    
    def _generate_api_documentation(self):
        """Generate comprehensive API documentation."""
        if not self.api_intelligence:
            logger.info("[DOCS] No API intelligence available, skipping API documentation")
            return
        
        logger.info("[DOCS] Generating API documentation")
        
        content = f"""# API Documentation

## Overview

This service exposes **{self.api_intelligence['total_endpoints']} RESTful API endpoints** for {self.project_intelligence['description'].lower()}.

## Base URL

```
{self.api_intelligence['base_url']}
```

## Authentication

{self._analyze_authentication()}

## API Groups

{self._generate_api_groups_documentation()}

## Data Models

{self._generate_data_models_documentation()}

## Error Handling

{self._generate_error_handling_documentation()}

## Rate Limiting

{self._analyze_rate_limiting()}

## Examples

{self._generate_api_examples()}

---

*Auto-generated API documentation by InfraDoc 2.0*
"""
        
        file_path = self.docs_dir / "api_documentation.md"
        file_path.write_text(content, encoding='utf-8')
        logger.info(f"[DOCS] API documentation: {file_path}")
    
    def _generate_api_groups_documentation(self) -> str:
        """Generate documentation for API endpoint groups."""
        docs = []
        
        for group_name, endpoints in self.api_intelligence['endpoint_groups'].items():
            docs.append(f"### {group_name.title()} Operations")
            docs.append("")
            
            # Create table for this group
            docs.append("| Method | Endpoint | Description | Parameters |")
            docs.append("|--------|----------|-------------|------------|")
            
            for endpoint in endpoints:
                method = endpoint.get('method', 'GET')
                path = endpoint.get('path', '/')
                description = endpoint.get('description', 'No description')
                params = ', '.join(endpoint.get('parameters', []))
                
                docs.append(f"| `{method}` | `{path}` | {description} | {params} |")
            
            docs.append("")
        
        return "\n".join(docs)
    
    def _generate_data_models_documentation(self) -> str:
        """Generate data models documentation."""
        models = self.api_intelligence.get('models', [])
        
        if not models:
            return "*No data models detected in the current analysis.*"
        
        docs = []
        
        for model in models:
            model_name = model.get('model_name', 'Unknown')
            table_name = model.get('table_name', '')
            fields = model.get('fields', [])
            relationships = model.get('relationships', [])
            
            docs.append(f"### {model_name}")
            
            if table_name:
                docs.append(f"**Database Table**: `{table_name}`")
            
            if fields:
                docs.append("")
                docs.append("| Field | Type | Description | Required |")
                docs.append("|-------|------|-------------|----------|")
                
                for field in fields:
                    field_name = field.get('name', 'unknown')
                    field_type = field.get('type', 'unknown')
                    field_desc = field.get('description', '')
                    required = 'No' if field.get('nullable', True) else 'Yes'
                    
                    docs.append(f"| `{field_name}` | `{field_type}` | {field_desc} | {required} |")
            
            if relationships:
                docs.append("")
                docs.append("**Relationships**:")
                for rel in relationships:
                    docs.append(f"- {rel}")
            
            docs.append("")
        
        return "\n".join(docs)
    
    def _analyze_authentication(self) -> str:
        """Analyze authentication methods."""
        return """
*Authentication methods not automatically detected.*

Common authentication patterns to verify:
- JWT tokens in Authorization header
- API keys in headers or query parameters  
- Session-based authentication
- OAuth 2.0 flows

Check the security assessment for authentication details.
"""
    
    def _generate_error_handling_documentation(self) -> str:
        """Generate error handling documentation."""
        return """
### Standard HTTP Status Codes

| Status Code | Meaning | Description |
|-------------|---------|-------------|
| `200` | OK | Request successful |
| `201` | Created | Resource created successfully |
| `400` | Bad Request | Invalid request parameters |
| `401` | Unauthorized | Authentication required |
| `403` | Forbidden | Access denied |
| `404` | Not Found | Resource not found |
| `500` | Internal Server Error | Server error occurred |

### Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": "Additional error details"
  }
}
```
"""
    
    def _analyze_rate_limiting(self) -> str:
        """Analyze rate limiting."""
        return """
*Rate limiting configuration not automatically detected.*

Recommended rate limiting strategy:
- **Public endpoints**: 100 requests per minute per IP
- **Authenticated endpoints**: 1000 requests per minute per user
- **Resource-intensive operations**: 10 requests per minute per user

Monitor API usage and adjust limits based on actual traffic patterns.
"""
    
    def _generate_api_examples(self) -> str:
        """Generate API usage examples."""
        examples = []
        
        # Get first few endpoints for examples
        all_endpoints = []
        for endpoints in self.api_intelligence['endpoint_groups'].values():
            all_endpoints.extend(endpoints)
        
        for endpoint in all_endpoints[:3]:
            method = endpoint.get('method', 'GET')
            path = endpoint.get('path', '/')
            description = endpoint.get('description', '')
            
            examples.append(f"""
### {description or f'{method} {path}'}

**Request**:
```bash
curl -X {method} '{self.api_intelligence['base_url']}{path}' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer YOUR_TOKEN'
```

**Response**:
```json
{{
  "status": "success",
  "data": {{
    "message": "Response data here"
  }}
}}
```
""")
        
        return "\n".join(examples) if examples else "*API examples will be generated based on endpoint analysis.*"
    
    def _generate_developer_setup_guide(self):
        """Generate comprehensive developer setup guide."""
        logger.info("[DOCS] Generating developer setup guide")
        
        content = f"""# Developer Setup Guide

> **Complete setup instructions for {self.project_intelligence['project_name']}**

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
cd {self.project_intelligence['project_name'].lower().replace(' ', '-')}

# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\\Scripts\\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configuration

{self._generate_configuration_setup_guide()}

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
"""
        
        file_path = self.docs_dir / "developer_setup.md"
        file_path.write_text(content, encoding='utf-8')
        logger.info(f"[DOCS] Developer setup guide: {file_path}")
    
    def _generate_configuration_setup_guide(self) -> str:
        """Generate configuration setup guide."""
        env_vars = self.deployment_intelligence.get('environment_vars', [])
        
        if not env_vars:
            return "*Configuration variables will be documented based on code analysis.*"
        
        guide = [
            "### Environment Variables",
            "Create a `.env` file in the project root:",
            "",
            "```bash",
            "# Copy the example file",
            "cp .env.example .env",
            "",
            "# Edit with your values",
        ]
        
        for var in env_vars[:5]:  # Show first 5
            var_name = var.get('variable_name', 'UNKNOWN')
            default = var.get('default_value', 'your_value_here')
            description = var.get('description', var.get('usage_context', ''))
            
            guide.append(f"# {description}")
            guide.append(f"{var_name}={default}")
        
        guide.extend(["```", ""])
        
        return "\n".join(guide)
    
    def _generate_architecture_deep_dive(self):
        """Generate architecture deep dive documentation."""
        logger.info("[DOCS] Generating architecture deep dive")
        
        insights = self.scan_report.infrastructure_insights
        
        content = f"""# Architecture Deep Dive

## System Architecture Overview

**Architecture Pattern**: {insights.architecture_pattern}
**Deployment Model**: {insights.deployment_model}
**Host**: {self.scan_report.host}

## Component Analysis

### Core Services
{self._generate_core_services_analysis()}

### Application Components
{self._generate_application_components_analysis()}

### Infrastructure Services
{self._generate_infrastructure_services_analysis()}

## Technology Stack

### Programming Languages
{self._generate_language_analysis()}

### Frameworks and Libraries
{self._generate_framework_analysis()}

### Infrastructure Components
{self._generate_infrastructure_components()}

## Service Dependencies

{self._generate_dependency_mapping()}

## Scalability Assessment

**Current Assessment**: {insights.scalability_assessment}

### Scalability Factors
{self._generate_scalability_analysis()}

## Operational Complexity

**Current Assessment**: {insights.operational_complexity}

### Complexity Factors
{self._generate_complexity_analysis()}

## Architecture Recommendations

{self._format_architecture_recommendations(insights.recommendations)}

---

*Architecture deep dive generated by InfraDoc 2.0*
"""
        
        file_path = self.docs_dir / "architecture_deep_dive.md"
        file_path.write_text(content, encoding='utf-8')
        logger.info(f"[DOCS] Architecture deep dive: {file_path}")
    
    def _generate_core_services_analysis(self) -> str:
        """Generate core services analysis."""
        core_services = []
        for process in self.scan_report.processes:
            if process.service_classification in ['web_server', 'service']:
                core_services.append(f"- {process.name}: {process.service_purpose}")
        
        return "\n".join(core_services) if core_services else "- No core services identified"
    
    def _generate_application_components_analysis(self) -> str:
        """Generate application components analysis."""
        app_components = []
        for process in self.scan_report.processes:
            if process.service_classification in ['application', 'background_worker']:
                app_components.append(f"- {process.name}: {process.service_purpose}")
        
        return "\n".join(app_components) if app_components else "- No application components identified"
    
    def _generate_infrastructure_services_analysis(self) -> str:
        """Generate infrastructure services analysis."""
        return "- Standard Linux infrastructure services"
    
    def _generate_language_analysis(self) -> str:
        """Generate programming language analysis."""
        languages = {}
        for file_info in self.scan_report.application_files:
            lang = file_info.language
            languages[lang] = languages.get(lang, 0) + 1
        
        return "\n".join(f"- {lang}: {count} files" for lang, count in languages.items())
    
    def _generate_framework_analysis(self) -> str:
        """Generate framework analysis."""
        return "- Framework analysis pending"
    
    def _generate_infrastructure_components(self) -> str:
        """Generate infrastructure components list."""
        components = []
        
        # Detect web server
        if any('nginx' in p.command.lower() for p in self.scan_report.processes):
            components.append("- Nginx web server")
        
        return "\n".join(components) if components else "- Standard Linux infrastructure"
    
    def _generate_dependency_mapping(self) -> str:
        """Generate service dependency mapping."""
        return """
### Service Dependencies

Based on process analysis:
- Web services depend on application workers
- Application workers may depend on external services
- All services depend on system infrastructure (logging, time sync)

**Recommendation**: Map detailed dependencies for better understanding.
"""
    
    def _generate_scalability_analysis(self) -> str:
        """Generate scalability analysis."""
        worker_count = len([p for p in self.scan_report.processes if 'worker' in p.command.lower()])
        
        return f"""
- **Current Workers**: {worker_count} background worker processes detected
- **Horizontal Scaling**: Consider load balancing for web services
- **Vertical Scaling**: Monitor resource usage for capacity planning
- **Bottlenecks**: Identify and address performance bottlenecks
"""
    
    def _generate_complexity_analysis(self) -> str:
        """Generate complexity analysis."""
        total_processes = len(self.scan_report.processes)
        app_processes = len([p for p in self.scan_report.processes if p.service_classification in ['application', 'background_worker']])
        
        return f"""
- **Total Processes**: {total_processes}
- **Application Processes**: {app_processes}
- **Complexity Drivers**: Multiple services, background workers
- **Management**: Consider orchestration tools for complex deployments
"""
    
    def _format_architecture_recommendations(self, recommendations: List[str]) -> str:
        """Format architecture recommendations."""
        return "\n".join(f"- {rec}" for rec in recommendations[:5]) if recommendations else "- Review current architecture for optimization opportunities"
    
    def _generate_business_logic_guide(self):
        """Generate business logic guide."""
        logger.info("[DOCS] Generating business logic guide")
        
        content = """# Business Logic Guide

## Overview

This document explains the business logic and domain understanding of the analyzed system.

## Business Intelligence

*Business intelligence analysis will be enhanced in future scans with deeper code analysis.*

## Key Components

Based on the discovered files and processes, this system appears to handle core application functionality.

## Data Flows

*Data flow analysis will be added based on business intelligence extraction.*

## Integration Points

*Integration points will be documented based on discovered external service connections.*

---

*Business logic guide generated by InfraDoc 2.0*
"""
        
        file_path = self.docs_dir / "business_logic.md"
        file_path.write_text(content, encoding='utf-8')
        logger.info(f"[DOCS] Business logic guide: {file_path}")
    
    def _generate_deployment_runbook(self):
        """Generate deployment runbook."""
        logger.info("[DOCS] Generating deployment runbook")
        
        content = """# Deployment Runbook

## Overview

This runbook provides step-by-step instructions for deploying the analyzed infrastructure.

## Prerequisites

- Target server with appropriate access
- SSH access to deployment environment
- Required environment variables configured

## Deployment Steps

### 1. Environment Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y python3 python3-pip nginx
```

### 2. Application Deployment

```bash
# Clone application code
git clone <repository-url>
cd application

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Service Configuration

```bash
# Start services
sudo systemctl enable nginx
sudo systemctl start nginx

# Start application services
python app.py &
```

### 4. Verification

```bash
# Check service status
sudo systemctl status nginx
ps aux | grep python

# Test connectivity
curl http://localhost/health
```

## Rollback Procedures

1. Stop services
2. Restore previous version
3. Restart services
4. Verify functionality

---

*Deployment runbook generated by InfraDoc 2.0*
"""
        
        file_path = self.docs_dir / "deployment_runbook.md"
        file_path.write_text(content, encoding='utf-8')
        logger.info(f"[DOCS] Deployment runbook: {file_path}")
    
    def _generate_security_assessment(self):
        """Generate security assessment."""
        logger.info("[DOCS] Generating security assessment")
        
        content = f"""# Security Assessment

## Executive Summary

**Infrastructure Security Assessment**
- **Host**: {self.scan_report.host}
- **Analysis Date**: {self.scan_report.timestamp[:10]}
- **Security Posture**: {self.scan_report.infrastructure_insights.security_posture}

## Security Analysis

{self.scan_report.security_analysis.get('analysis', 'Security analysis completed with basic assessment.')}

## Key Security Findings

{self._format_security_findings()}

## Priority Recommendations

{self._format_security_recommendations()}

## Process Security Review

{self._generate_process_security_analysis()}

## Compliance and Governance

### Access Control
- Review user permissions and role-based access
- Implement principle of least privilege
- Regular access audits

### Data Protection
- Encrypt data in transit and at rest
- Implement proper backup strategies
- Ensure data retention policies

### Monitoring and Incident Response
- Implement comprehensive logging
- Set up security monitoring and alerting
- Develop incident response procedures

## Action Items

### Immediate (High Priority)
- Update all system packages
- Review and harden SSH configuration
- Implement proper firewall rules

### Short Term (Medium Priority)
- Implement comprehensive logging and monitoring
- Set up automated security updates
- Conduct security configuration review

### Long Term (Strategic)
- Implement infrastructure as code
- Consider containerization for better isolation
- Regular penetration testing and security audits

---

*Security assessment generated by InfraDoc 2.0*
"""
        
        file_path = self.docs_dir / "security_assessment.md"
        file_path.write_text(content, encoding='utf-8')
        logger.info(f"[DOCS] Security assessment: {file_path}")
    
    def _format_security_findings(self) -> str:
        """Format security findings."""
        return "- Review processes running with elevated privileges\n- Verify network service configurations\n- Check for proper authentication mechanisms"
    
    def _format_security_recommendations(self) -> str:
        """Format security recommendations."""
        return "- Implement regular security updates\n- Configure proper firewall rules\n- Set up monitoring and alerting\n- Review access controls"
    
    def _generate_process_security_analysis(self) -> str:
        """Generate process-level security analysis."""
        root_processes = [p for p in self.scan_report.processes if p.user == 'root']
        
        return f"""
### Processes Running as Root
{len(root_processes)} processes are running with root privileges.

**Risk Level**: Medium to High - Consider running with least privileges where possible.
"""
    
    def _generate_troubleshooting_guide(self):
        """Generate troubleshooting guide."""
        logger.info("[DOCS] Generating troubleshooting guide")
        
        content = """# Troubleshooting Guide

## Common Issues

### Service Won't Start

**Symptoms**: Service fails to start or immediately stops

**Diagnosis**:
```bash
# Check service status
sudo systemctl status service-name

# Check logs
journalctl -u service-name -f

# Check configuration
python -c "from config import settings; print(settings)"
```

**Solutions**:
- Verify configuration files
- Check file permissions
- Review environment variables
- Check available resources (disk, memory)

### API Not Responding

**Symptoms**: HTTP requests timeout or return errors

**Diagnosis**:
```bash
# Check if process is running
ps aux | grep python

# Check network connectivity
netstat -tlnp | grep :8000

# Test locally
curl http://localhost:8000/health
```

**Solutions**:
- Restart application service
- Check firewall configuration
- Verify port binding
- Review application logs

### Performance Issues

**Symptoms**: Slow response times, high resource usage

**Diagnosis**:
```bash
# Monitor resource usage
htop

# Check disk space
df -h

# Monitor network
iftop
```

**Solutions**:
- Scale resources
- Optimize application code
- Add caching layer
- Load balance traffic

### Database Connection Issues

**Symptoms**: Database connection errors

**Diagnosis**:
```bash
# Test database connectivity
ping database-host

# Check database service
sudo systemctl status postgresql
```

**Solutions**:
- Verify database credentials
- Check network connectivity
- Review database configuration
- Check connection limits

## Emergency Procedures

### Service Recovery

1. Stop affected services
2. Check logs for error details
3. Apply necessary fixes
4. Restart services
5. Verify functionality

### Data Recovery

1. Identify affected data
2. Stop write operations
3. Restore from backup
4. Verify data integrity
5. Resume operations

## Contact Information

- System Administrator: [Contact Info]
- Database Administrator: [Contact Info]
- Security Team: [Contact Info]

---

*Troubleshooting guide generated by InfraDoc 2.0*
"""
        
        file_path = self.docs_dir / "troubleshooting.md"
        file_path.write_text(content, encoding='utf-8')
        logger.info(f"[DOCS] Troubleshooting guide: {file_path}")
    
    def _generate_monitoring_guide(self):
        """Generate monitoring guide."""
        logger.info("[DOCS] Generating monitoring guide")
        
        content = """# Monitoring Guide

## Overview

This guide provides monitoring strategies and health checks for the analyzed infrastructure.

## Health Checks

### Application Health
```bash
# Check application status
curl http://localhost:8000/health

# Verify processes
ps aux | grep python
```

### System Health
```bash
# Check system resources
free -h
df -h
uptime

# Check service status
sudo systemctl status nginx
```

## Monitoring Metrics

### Key Performance Indicators
- Response time
- Error rate
- Throughput
- Resource utilization

### System Metrics
- CPU usage
- Memory usage
- Disk space
- Network traffic

## Alerting

### Critical Alerts
- Service down
- High error rate
- Resource exhaustion

### Warning Alerts
- High response time
- Low disk space
- High CPU usage

## Monitoring Tools

### Recommended Tools
- Prometheus for metrics collection
- Grafana for visualization
- AlertManager for alerting

### Log Management
- Centralized logging with ELK stack
- Log rotation and retention
- Error tracking and analysis

---

*Monitoring guide generated by InfraDoc 2.0*
"""
        
        file_path = self.docs_dir / "monitoring.md"
        file_path.write_text(content, encoding='utf-8')
        logger.info(f"[DOCS] Monitoring guide: {file_path}")
    
    def _generate_performance_guide(self):
        """Generate performance guide."""
        logger.info("[DOCS] Generating performance guide")
        
        content = """# Performance Guide

## Overview

Performance optimization strategies and best practices for the analyzed infrastructure.

## Performance Metrics

### Current Performance
- Active Processes: {len(self.scan_report.processes)}
- Application Files: {len(self.scan_report.application_files)}
- Architecture Complexity: {self.scan_report.infrastructure_insights.operational_complexity}

## Optimization Strategies

### Application Level
- Code optimization
- Database query optimization
- Caching implementation
- Connection pooling

### Infrastructure Level
- Resource scaling
- Load balancing
- CDN implementation
- Database optimization

## Performance Testing

### Load Testing
```bash
# Basic load testing with curl
for i in {{1..100}}; do
  curl http://localhost:8000/api/endpoint &
done
```

### Monitoring During Tests
- Monitor CPU and memory usage
- Track response times
- Check error rates

## Capacity Planning

### Current Capacity
- Analyze current resource usage
- Identify bottlenecks
- Plan for growth

### Scaling Strategies
- Horizontal scaling options
- Vertical scaling limits
- Auto-scaling implementation

---

*Performance guide generated by InfraDoc 2.0*
"""
        
        file_path = self.docs_dir / "performance.md"
        file_path.write_text(content, encoding='utf-8')
        logger.info(f"[DOCS] Performance guide: {file_path}")
    
    def _generate_enhanced_executive_summary(self):
        """Generate enhanced executive summary."""
        logger.info("[DOCS] Generating enhanced executive summary")
        
        insights = self.scan_report.infrastructure_insights
        
        content = f"""# Executive Summary

## Infrastructure Analysis Overview

**Analysis Date**: {self.scan_report.timestamp[:10]}  
**Target System**: {self.scan_report.host}  
**Analysis Duration**: {self.scan_report.scan_duration:.2f} seconds

## Key Findings

### System Overview
- **Architecture Pattern**: {insights.architecture_pattern}
- **Deployment Model**: {insights.deployment_model}
- **Processes Analyzed**: {len(self.scan_report.processes)}
- **Application Files**: {len(self.scan_report.application_files)}

### Technology Assessment
- **Technology Stack**: {', '.join(insights.technology_stack)}
- **Scalability**: {insights.scalability_assessment}
- **Security Posture**: {insights.security_posture}
- **Operational Complexity**: {insights.operational_complexity}

## Strategic Recommendations

### Priority Actions
"""
        
        for i, rec in enumerate(insights.recommendations[:5], 1):
            content += f"{i}. {rec}\n"
        
        content += f"""
### Business Impact
- Improved system reliability through proper documentation
- Enhanced developer productivity with clear setup guides
- Reduced operational overhead through automation insights
- Better security posture through identified vulnerabilities

### Next Steps
1. Review and implement priority recommendations
2. Establish regular infrastructure analysis schedule
3. Implement monitoring and alerting based on findings
4. Train team on new documentation and procedures

## Investment Analysis

### Technical Debt
- **Current State**: {insights.operational_complexity} operational complexity
- **Improvement Potential**: {insights.scalability_assessment} scalability readiness

### Risk Assessment
- **Security Risk**: {insights.security_posture}
- **Operational Risk**: Medium (based on complexity)
- **Business Risk**: Low to Medium (based on current state)

---

*Executive summary generated by InfraDoc 2.0*  
*This analysis provides actionable insights for strategic infrastructure decisions.*
"""
        
        file_path = self.docs_dir / "executive_summary.md"
        file_path.write_text(content, encoding='utf-8')
        logger.info(f"[DOCS] Enhanced executive summary: {file_path}")
    
    def _generate_documentation_index(self):
        """Generate documentation index/README."""
        logger.info("[DOCS] Generating documentation index")
        
        content = f"""# Infrastructure Documentation

## Overview

This documentation suite provides comprehensive analysis and guidance for the infrastructure at `{self.scan_report.host}`.

**Analysis Date**: {self.scan_report.timestamp[:10]}  
**Scan ID**: {self.scan_report.scan_id}  
**Duration**: {self.scan_report.scan_duration:.2f} seconds

## Documentation Structure

### Executive Summary
**File**: [executive_summary.md](./executive_summary.md)  
High-level overview of infrastructure analysis, key findings, and strategic recommendations.

### Developer Setup Guide
**File**: [developer_setup.md](./developer_setup.md)  
Complete setup instructions for developers to recreate the environment.

### API Documentation
**File**: [api_documentation.md](./api_documentation.md)  
Comprehensive API reference with endpoints, models, and examples.

### Architecture Deep Dive
**File**: [architecture_deep_dive.md](./architecture_deep_dive.md)  
Detailed architecture analysis including patterns, components, and design decisions.

### Business Logic Guide
**File**: [business_logic.md](./business_logic.md)  
Understanding of business domain and application logic.

### Deployment Runbook
**File**: [deployment_runbook.md](./deployment_runbook.md)  
Step-by-step deployment procedures for production environments.

### Security Assessment
**File**: [security_assessment.md](./security_assessment.md)  
Comprehensive security analysis with vulnerabilities, risks, and remediation recommendations.

### Troubleshooting Guide
**File**: [troubleshooting.md](./troubleshooting.md)  
Common issues, diagnostic procedures, and solutions.

### Monitoring Guide
**File**: [monitoring.md](./monitoring.md)  
Monitoring strategies, health checks, and alerting recommendations.

### Performance Guide
**File**: [performance.md](./performance.md)  
Performance optimization strategies and capacity planning.

## Key Findings

### System Summary
- **Processes Analyzed**: {len(self.scan_report.processes)}
- **Application Files**: {len(self.scan_report.application_files)}
- **Architecture Pattern**: {self.scan_report.infrastructure_insights.architecture_pattern}
- **Security Posture**: {self.scan_report.infrastructure_insights.security_posture}

### Technology Stack
{self._format_technology_stack_index()}

### Critical Services
{self._format_critical_services_index()}

## Quick Start

1. **For Executives**: Start with [Executive Summary](./executive_summary.md)
2. **For Developers**: Review [Developer Setup Guide](./developer_setup.md)
3. **For Operations**: Check [Deployment Runbook](./deployment_runbook.md)
4. **For Security**: Study [Security Assessment](./security_assessment.md)

## Analysis Methodology

This documentation was generated using InfraDoc 2.0's intelligent analysis:

- **LLM Calls**: {self.scan_report.llm_analysis_summary.get('total_llm_calls', 0)}
- **Analysis Stages**: {self.scan_report.llm_analysis_summary.get('analysis_stages', 0)}
- **Overall Confidence**: {int(self.scan_report.llm_analysis_summary.get('overall_confidence', 0) * 100)}%

## Support and Updates

For questions about this documentation or to request updates:
- Re-run InfraDoc analysis for latest state
- Review change logs for infrastructure modifications
- Validate configurations against security baselines

---

*Documentation generated by InfraDoc 2.0 - Intelligent Infrastructure Analysis*  
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        file_path = self.docs_dir / "README.md"
        file_path.write_text(content, encoding='utf-8')
        logger.info(f"[DOCS] Documentation index: {file_path}")
    
    def _format_technology_stack_index(self) -> str:
        """Format technology stack for index."""
        tech_stack = self.scan_report.infrastructure_insights.technology_stack
        return "\n".join(f"- {tech}" for tech in tech_stack) if tech_stack else "- Technology analysis pending"
    
    def _format_critical_services_index(self) -> str:
        """Format critical services for index."""
        critical_services = []
        for process in self.scan_report.processes:
            if process.service_classification in ['web_server', 'background_worker'] or 'worker' in process.command.lower():
                critical_services.append(f"- {process.name} ({process.service_classification})")
        
        return "\n".join(critical_services[:5]) if critical_services else "- No critical services identified"


# Keep the original DocumentationGenerator class for backward compatibility
class DocumentationGenerator(IntelligentDocumentationGenerator):
    """Backward compatibility wrapper for the original DocumentationGenerator."""
    
    def generate_all_documentation(self) -> bool:
        """Generate documentation using the intelligent generator."""
        return self.generate_intelligent_documentation()

# Keep the original DocumentationGenerator class for backward compatibility
class DocumentationGenerator(IntelligentDocumentationGenerator):
    """Backward compatibility wrapper for the original DocumentationGenerator."""
    
    def generate_all_documentation(self) -> bool:
        """Generate documentation using the intelligent generator."""
        return self.generate_intelligent_documentation()