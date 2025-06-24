#!/usr/bin/env python3
"""
InfraDoc 2.0 - Enhanced Command Line Interface
Main CLI for intelligent infrastructure analysis and documentation generation.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional
import sys
import locale
import codecs


# Add these lines at the top after imports in infradoc_cli.py:
import sys
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Import core components
try:
    from infradoc_core import ConnectionConfig
    from infradoc_analyzer import (
        IntelligentInfrastructureAnalyzer, 
        AnalysisConfig, 
        quick_analysis, 
        deep_analysis, 
        intelligent_analysis
    )
except ImportError as e:
    print(f"ERROR: Error importing InfraDoc components: {e}")
    print("Make sure all InfraDoc 2.0 files are in the same directory:")
    print("  - infradoc_core.py")
    print("  - infradoc_analyzer.py") 
    print("  - infradoc_docs.py")
    print("  - infradoc_cli.py")
    sys.exit(1)

# Configure logging
def setup_logging(verbose: bool = False, quiet: bool = False):
    """Setup logging configuration."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup file handler
    file_handler = logging.FileHandler('infradoc.log')
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

def validate_environment():
    """Validate environment and dependencies."""
    errors = []
    warnings = []
    
    # Check for required API keys if AI is enabled
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Grok": os.getenv("GROK_API_KEY")
    }
    
    available_providers = [name for name, key in api_keys.items() if key]
    
    if not available_providers:
        warnings.append("No LLM API keys found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GROK_API_KEY for AI features")
    
    # Check SSH key file if provided
    ssh_key = os.getenv("SSH_KEY_FILE")
    if ssh_key and not os.path.exists(os.path.expanduser(ssh_key)):
        errors.append(f"SSH key file not found: {ssh_key}")
    
    # Check for required Python packages
    try:
        import paramiko
    except ImportError:
        errors.append("paramiko package required. Install with: pip install paramiko")
    
    return errors, warnings, available_providers

def print_banner():
    """Print InfraDoc banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                    InfraDoc 2.0                               ║
║              Intelligent Infrastructure Analysis              ║
╠═══════════════════════════════════════════════════════════════╣
║  🧠 AI-Powered Analysis    📚 Auto-Documentation             ║
║  🔍 Smart Discovery        🛡️  Security Assessment           ║
║  📊 Business Intelligence  🚀 Developer-Ready Guides         ║
║                                                               ║
║           "Developers just develop, we'll document"           ║ 
╚═══════════════════════════════════════════════════════════════╝
""")

def cmd_analyze(args):
    """Execute infrastructure analysis command."""
    print_banner()
    
    logger.info(f"🚀 Starting infrastructure analysis for {args.host}")
    
    # Validate environment
    errors, warnings, available_providers = validate_environment()
    
    if errors:
        for error in errors:
            print(f"❌ ERROR: {error}")
        return 1
    
    for warning in warnings:
        print(f"⚠️  WARNING: {warning}")
    
    print(f"🔑 Available AI Providers: {', '.join(available_providers) if available_providers else 'None'}")
    
    # Configure connection
    connection_config = ConnectionConfig(
        host=args.host,
        port=args.port,
        username=args.username,
        key_file=args.key_file,
        password=args.password,
        timeout=args.timeout,
        max_retries=args.retries
    )
    
    # Configure analysis
    analysis_config = AnalysisConfig(
        scan_depth=args.depth,
        enable_ai=not args.no_ai,
        max_llm_calls=args.max_llm_calls,
        output_formats=args.output_formats,
        export_artifacts=not args.no_artifacts,
        include_security=not args.no_security,
        include_documentation=not args.no_docs,
        enable_business_intelligence=args.depth in ['deep', 'intelligent'],
        enable_enhanced_analysis=args.depth == 'intelligent'
    )
    
    # Determine LLM providers
    llm_providers = []
    if not args.no_ai and available_providers:
        if "OpenAI" in available_providers:
            llm_providers.append({"provider": "openai", "model": "gpt-4o"})
        if "Anthropic" in available_providers:
            llm_providers.append({"provider": "claude", "model": "claude-3-5-sonnet-20241022"})
        if "Grok" in available_providers:
            llm_providers.append({"provider": "grok", "model": "grok-3"})
    
    if not llm_providers and not args.no_ai:
        print("⚠️  WARNING: No LLM providers available, running basic analysis only")
        analysis_config.enable_ai = False
    
    analyzer = None
    try:
        # Initialize analyzer
        analyzer = IntelligentInfrastructureAnalyzer(
            llm_providers=llm_providers if llm_providers else None,
            output_base_dir=args.output_dir
        )
        
        # Run analysis
        result = analyzer.analyze_infrastructure(connection_config, analysis_config)
        
        if result.success:
            print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"📊 Processes analyzed: {len(result.scan_report.processes)}")
            print(f"📁 Files discovered: {len(result.scan_report.application_files)}")
            print(f"🏗️ Architecture: {result.scan_report.infrastructure_insights.architecture_pattern}")
            print(f"📋 Artifacts generated: {len(result.artifacts_generated)}")
            print(f"📚 Documentation: {'YES' if result.documentation_generated else 'NO'}")
            
            # Show enhanced features if available
            if hasattr(result.scan_report, 'business_intelligence') and result.scan_report.business_intelligence:
                domain = result.scan_report.business_intelligence.get('business_domain', 'Unknown')
                print(f"🏢 Business domain: {domain}")
            
            if hasattr(result.scan_report, 'api_documentation') and result.scan_report.api_documentation:
                endpoint_count = len(result.scan_report.api_documentation.get('endpoints', []))
                print(f"🌐 API endpoints: {endpoint_count}")
            
            print(f"📂 Output directory: {result.output_directory}")
            
            # Show generated files
            print(f"\n📋 Generated Files:")
            for artifact in result.artifacts_generated:
                print(f"   ✅ {Path(artifact).name}")
            
            if result.documentation_generated:
                docs_dir = Path(result.output_directory) / "documentation"
                if docs_dir.exists():
                    print(f"\n📚 Documentation Files:")
                    for doc_file in docs_dir.glob("*.md"):
                        print(f"   📄 {doc_file.name}")
            
            return 0
        else:
            print(f"❌ ANALYSIS FAILED: {result.error_message}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⛔ Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # Ensure cleanup
        if analyzer:
            try:
                analyzer._cleanup_resources()
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error: {cleanup_error}")

def cmd_quick(args):
    """Execute quick analysis command."""
    print("🚀 Running QUICK infrastructure analysis...")
    print("⏱️  Expected duration: 2-5 minutes")
    
    try:
        result = quick_analysis(
            host=args.host,
            username=args.username,
            key_file=args.key_file,
            password=args.password
        )
        
        if result.success:
            print(f"✅ Quick analysis completed!")
            print(f"📊 Found {len(result.scan_report.processes)} processes")
            print(f"📁 Found {len(result.scan_report.application_files)} files")
            print(f"📂 Results: {result.output_directory}")
            return 0
        else:
            print(f"❌ Quick analysis failed: {result.error_message}")
            return 1
            
    except Exception as e:
        print(f"💥 Quick analysis error: {e}")
        return 1

def cmd_deep(args):
    """Execute deep analysis command."""
    print("🧠 Running DEEP infrastructure analysis with full AI...")
    print("⏱️  Expected duration: 10-20 minutes")
    print("🔍 Features: Complete code analysis + Business intelligence + Documentation")
    
    try:
        result = deep_analysis(
            host=args.host,
            username=args.username,
            key_file=args.key_file,
            password=args.password
        )
        
        if result.success:
            print(f"🎉 Deep analysis completed!")
            print(f"🧠 LLM calls: {result.scan_report.llm_analysis_summary.get('total_llm_calls', 0)}")
            print(f"📊 Processes: {len(result.scan_report.processes)}")
            print(f"📁 Files: {len(result.scan_report.application_files)}")
            print(f"📚 Documentation: {'YES' if result.documentation_generated else 'NO'}")
            print(f"📂 Results: {result.output_directory}")
            return 0
        else:
            print(f"❌ Deep analysis failed: {result.error_message}")
            return 1
            
    except Exception as e:
        print(f"💥 Deep analysis error: {e}")
        return 1

def cmd_intelligent(args):
    """Execute intelligent analysis command with enhanced business understanding."""
    print("🧠 Running INTELLIGENT infrastructure analysis...")
    print("⏱️  Expected duration: 15-30 minutes")
    print("✨ Features: Complete business intelligence + API documentation + Developer guides")
    print("🎯 Goal: 'Developers just develop, we'll document'")
    
    try:
        result = intelligent_analysis(
            host=args.host,
            username=args.username,
            key_file=args.key_file,
            password=args.password
        )
        
        if result.success:
            print(f"\n🎉 INTELLIGENT ANALYSIS COMPLETED!")
            print("=" * 60)
            print(f"🧠 Business intelligence extracted")
            print(f"📊 {len(result.scan_report.processes)} processes analyzed")
            print(f"📁 {len(result.scan_report.application_files)} files understood")
            
            # Show business intelligence
            if hasattr(result.scan_report, 'business_intelligence') and result.scan_report.business_intelligence:
                bi = result.scan_report.business_intelligence
                print(f"🏢 Business domain: {bi.get('business_domain', 'Unknown')}")
                functions = bi.get('primary_business_functions', [])
                if functions:
                    print(f"⚙️  Key functions: {', '.join(functions[:3])}")
            
            # Show API documentation
            if hasattr(result.scan_report, 'api_documentation') and result.scan_report.api_documentation:
                endpoints = result.scan_report.api_documentation.get('endpoints', [])
                print(f"🌐 API endpoints documented: {len(endpoints)}")
            
            print(f"📚 Intelligent documentation generated")
            print(f"📂 Results: {result.output_directory}")
            
            # Show intelligent documentation files
            docs_dir = Path(result.output_directory) / "documentation"
            if docs_dir.exists():
                print(f"\n📚 Intelligent Documentation:")
                doc_files = [
                    ("README.md", "📖 Intelligent project overview"),
                    ("developer_setup.md", "🔧 Complete setup guide"),
                    ("api_documentation.md", "🌐 Auto-generated API docs"),
                    ("business_logic.md", "🏢 Business intelligence guide"),
                    ("architecture_deep_dive.md", "🏗️ Architecture analysis"),
                    ("security_assessment.md", "🔒 Security assessment"),
                    ("troubleshooting.md", "🐛 Troubleshooting guide")
                ]
                
                for filename, description in doc_files:
                    if (docs_dir / filename).exists():
                        print(f"   {description}")
            
            return 0
        else:
            print(f"❌ Intelligent analysis failed: {result.error_message}")
            return 1
            
    except Exception as e:
        print(f"💥 Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

def cmd_version(args):
    """Show version information."""
    print("InfraDoc 2.0 - Intelligent Infrastructure Analysis")
    print("Version: 2.0.0")
    print("Author: InfraDoc Team")
    print()
    print("🚀 Enhanced Features:")
    print("  ✅ AI-powered analysis with LLM orchestration")
    print("  ✅ Business intelligence extraction")
    print("  ✅ Intelligent documentation generation")
    print("  ✅ API documentation auto-generation") 
    print("  ✅ Developer-ready setup guides")
    print("  ✅ Security assessment and recommendations")
    print("  ✅ Architecture pattern recognition")
    print("  ✅ Beautiful markdown documentation with tables")
    print()
    print("🧠 Supported AI Providers:")
    print("  • OpenAI GPT-4o")
    print("  • Anthropic Claude 3.5 Sonnet")
    print("  • xAI Grok")
    return 0

def cmd_validate(args):
    """Validate environment and configuration."""
    print("🔍 Validating InfraDoc environment...")
    
    errors, warnings, available_providers = validate_environment()
    
    # Check API keys
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Grok": os.getenv("GROK_API_KEY")
    }
    
    print(f"\n🔑 API Keys Status:")
    for provider, key in api_keys.items():
        status = "✅ SET" if key else "❌ NOT SET"
        print(f"   {provider}: {status}")
    
    # Check dependencies
    print(f"\n📦 Dependencies:")
    try:
        import paramiko
        print(f"   ✅ paramiko: {paramiko.__version__}")
    except ImportError:
        print(f"   ❌ paramiko: Not installed")
    
    try:
        import openai
        print(f"   ✅ openai: Available")
    except ImportError:
        print(f"   ⚠️  openai: Not installed (optional for OpenAI)")
    
    try:
        import anthropic
        print(f"   ✅ anthropic: Available")
    except ImportError:
        print(f"   ⚠️  anthropic: Not installed (optional for Claude)")
    
    # Show configuration
    if args.host:
        print(f"\n🔗 Connection Test:")
        print(f"   Host: {args.host}")
        print(f"   Username: {args.username}")
        print(f"   SSH Key: {args.key_file if args.key_file else 'Not specified'}")
        
        # Test connection if possible
        try:
            from infradoc_core import SSHConnector, ConnectionConfig
            connector = SSHConnector()
            config = ConnectionConfig(
                host=args.host,
                username=args.username,
                key_file=args.key_file,
                password=args.password
            )
            
            print(f"   🔄 Testing connection...")
            if connector.connect(config):
                print(f"   ✅ Connection successful")
                connector.close_all_connections()
            else:
                print(f"   ❌ Connection failed")
        except Exception as e:
            print(f"   ⚠️  Connection test error: {e}")
    
    # Show intelligent features status
    print(f"\n🧠 Intelligent Features:")
    if available_providers:
        print(f"   ✅ AI Analysis: Available ({', '.join(available_providers)})")
        print(f"   ✅ Business Intelligence: Available")
        print(f"   ✅ API Documentation: Available")
        print(f"   ✅ Intelligent Documentation: Available")
    else:
        print(f"   ⚠️  AI Analysis: Limited (no API keys)")
        print(f"   ⚠️  Business Intelligence: Limited")
        print(f"   ⚠️  API Documentation: Limited")
        print(f"   ✅ Basic Documentation: Available")
    
    # Summary
    print(f"\n📋 Validation Summary:")
    if errors:
        print(f"   ❌ {len(errors)} errors found")
        for error in errors:
            print(f"      • {error}")
    
    if warnings:
        print(f"   ⚠️  {len(warnings)} warnings")
        for warning in warnings:
            print(f"      • {warning}")
    
    if not errors:
        if available_providers:
            print(f"   🎉 Environment ready for INTELLIGENT InfraDoc analysis")
        else:
            print(f"   ✅ Environment ready for BASIC InfraDoc analysis")
        return 0
    else:
        return 1

def cmd_demo(args):
    """Run demo mode with sample data."""
    print("🎭 Running InfraDoc 2.0 Demo Mode...")
    print()
    print("This demo shows what InfraDoc 2.0 can generate:")
    print()
    
    # Create demo output directory
    demo_dir = Path("infradoc_demo")
    demo_dir.mkdir(exist_ok=True)
    docs_dir = demo_dir / "documentation"
    docs_dir.mkdir(exist_ok=True)
    
    # Generate demo README
    demo_readme = """# Sample E-commerce API

> **Web API Service** - RESTful API service for web and mobile applications

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd sample-ecommerce-api

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start services
python app.py
```

## 🏗️ Architecture Overview

| Component | Details |
|-----------|---------|
| **Architecture Pattern** | Microservices |
| **Deployment Model** | Cloud-native |
| **Technology Stack** | Python, Flask, PostgreSQL, Redis |
| **Processes Running** | 8 active processes |
| **Application Files** | 15 files analyzed |
| **Scalability** | Good |
| **Security Posture** | Needs attention |

## 🌐 API Reference

This service exposes **12 API endpoints** organized by functionality:

### Quick API Reference
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/users` | User management operations |
| `POST` | `/api/users` | Create new user |
| `GET` | `/api/products` | Product catalog operations |
| `POST` | `/api/orders` | Order processing |

**Base URL**: `http://localhost:8000`

## 🏢 Business Logic

**Domain**: E-commerce

### Core Business Functions
- User account management
- Product catalog management  
- Order processing and fulfillment

---

**🤖 Auto-generated by InfraDoc 2.0** - *"Developers just develop, we'll document"*
"""
    
    (docs_dir / "README.md").write_text(demo_readme)
    
    print("📁 Demo documentation generated:")
    print(f"   📄 {docs_dir / 'README.md'}")
    print()
    print("🎯 This example shows how InfraDoc 2.0 generates:")
    print("   ✅ Intelligent project overviews")
    print("   ✅ Architecture analysis with tables")
    print("   ✅ API documentation extraction")
    print("   ✅ Business intelligence insights") 
    print("   ✅ Developer-ready setup guides")
    print()
    print("🚀 To analyze your real infrastructure:")
    print("   python infradoc_cli.py intelligent --host your-server.com")
    
    return 0

def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="InfraDoc 2.0 - Intelligent Infrastructure Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 Examples:
  # Quick analysis (2-5 minutes)
  python infradoc_cli.py quick --host server.example.com
  
  # Deep analysis with AI (10-20 minutes)
  python infradoc_cli.py deep --host server.example.com
  
  # Intelligent analysis with business understanding (15-30 minutes)
  python infradoc_cli.py intelligent --host server.example.com
  
  # Custom analysis with specific options
  python infradoc_cli.py analyze --host server.example.com --depth intelligent --max-llm-calls 40
  
  # Validate environment and test connection
  python infradoc_cli.py validate --host server.example.com
  
  # See demo output
  python infradoc_cli.py demo

🔑 Environment Variables:
  OPENAI_API_KEY     - OpenAI API key for GPT models
  ANTHROPIC_API_KEY  - Anthropic API key for Claude models  
  GROK_API_KEY       - Grok API key for Grok models
  SSH_KEY_FILE       - Default SSH key file path

🧠 Intelligent Features:
  • Business intelligence extraction
  • API documentation auto-generation
  • Developer setup guide creation
  • Security assessment and recommendations
  • Architecture pattern recognition
  • Beautiful documentation with tables and structure

🎯 Goal: "Developers just develop, we'll document"
        """
    )
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')
    parser.add_argument('--debug', action='store_true', help='Debug mode with stack traces')
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command (full customization)
    analyze_parser = subparsers.add_parser('analyze', help='Run comprehensive infrastructure analysis')
    analyze_parser.add_argument('--host', required=True, help='Target hostname or IP address')
    analyze_parser.add_argument('--port', '-p', type=int, default=22, help='SSH port (default: 22)')
    analyze_parser.add_argument('--username', '-u', default='ubuntu', help='SSH username (default: ubuntu)')
    analyze_parser.add_argument('--key-file', '-k', help='SSH private key file path')
    analyze_parser.add_argument('--password', help='SSH password (not recommended)')
    analyze_parser.add_argument('--timeout', type=int, default=30, help='SSH timeout in seconds (default: 30)')
    analyze_parser.add_argument('--retries', type=int, default=3, help='SSH connection retries (default: 3)')
    analyze_parser.add_argument('--depth', choices=['quick', 'standard', 'deep', 'intelligent'], default='standard', help='Analysis depth')
    analyze_parser.add_argument('--output-dir', '-o', default='infradoc_analysis', help='Output directory')
    analyze_parser.add_argument('--no-ai', action='store_true', help='Disable AI analysis')
    analyze_parser.add_argument('--max-llm-calls', type=int, default=25, help='Maximum LLM calls (default: 25)')
    analyze_parser.add_argument('--output-formats', nargs='+', default=['json', 'markdown'], help='Output formats')
    analyze_parser.add_argument('--no-artifacts', action='store_true', help='Skip artifact generation')
    analyze_parser.add_argument('--no-security', action='store_true', help='Skip security analysis')
    analyze_parser.add_argument('--no-docs', action='store_true', help='Skip documentation generation')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Quick command (preset for quick analysis)
    quick_parser = subparsers.add_parser('quick', help='🚀 Run quick analysis (2-5 minutes)')
    quick_parser.add_argument('--host', required=True, help='Target hostname or IP address')
    quick_parser.add_argument('--username', '-u', default='ubuntu', help='SSH username (default: ubuntu)')
    quick_parser.add_argument('--key-file', '-k', help='SSH private key file path')
    quick_parser.add_argument('--password', help='SSH password (not recommended)')
    quick_parser.set_defaults(func=cmd_quick)
    
    # Deep command (preset for deep analysis)
    deep_parser = subparsers.add_parser('deep', help='🧠 Run deep analysis with full AI (10-20 minutes)')
    deep_parser.add_argument('--host', required=True, help='Target hostname or IP address')
    deep_parser.add_argument('--username', '-u', default='ubuntu', help='SSH username (default: ubuntu)')
    deep_parser.add_argument('--key-file', '-k', help='SSH private key file path')
    deep_parser.add_argument('--password', help='SSH password (not recommended)')
    deep_parser.set_defaults(func=cmd_deep)
    
    # Intelligent command (preset for intelligent analysis with business understanding)
    intelligent_parser = subparsers.add_parser('intelligent', help='✨ Run intelligent analysis with business understanding (15-30 minutes)')
    intelligent_parser.add_argument('--host', required=True, help='Target hostname or IP address')
    intelligent_parser.add_argument('--username', '-u', default='ubuntu', help='SSH username (default: ubuntu)')
    intelligent_parser.add_argument('--key-file', '-k', help='SSH private key file path')
    intelligent_parser.add_argument('--password', help='SSH password (not recommended)')
    intelligent_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    intelligent_parser.set_defaults(func=cmd_intelligent)
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    version_parser.set_defaults(func=cmd_version)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='🔍 Validate environment and configuration')
    validate_parser.add_argument('--host', help='Test connection to host')
    validate_parser.add_argument('--username', '-u', default='ubuntu', help='SSH username for connection test')
    validate_parser.add_argument('--key-file', '-k', help='SSH private key file for connection test')
    validate_parser.add_argument('--password', help='SSH password for connection test')
    validate_parser.set_defaults(func=cmd_validate)
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='🎭 Show demo of generated documentation')
    demo_parser.set_defaults(func=cmd_demo)
    
    return parser

def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    
    # Handle no command provided
    if not args.command:
        print_banner()
        print("🎯 Choose your analysis type:")
        print("   quick       - 2-5 minutes, basic analysis")
        print("   deep        - 10-20 minutes, full AI analysis")
        print("   intelligent - 15-30 minutes, business intelligence + documentation")
        print("   demo        - See example output")
        print("   validate    - Check environment setup")
        print()
        print("📚 For full help: python infradoc_cli.py --help")
        sys.exit(0)
    
    # Execute command
    try:
        exit_code = args.func(args)
        if exit_code == 0:
            print(f"\n✅ InfraDoc analysis completed successfully.")
        else:
            print(f"\n❌ InfraDoc analysis completed with errors.")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⛔ Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if hasattr(args, 'debug') and args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()