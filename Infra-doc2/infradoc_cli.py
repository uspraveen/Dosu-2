#!/usr/bin/env python3
"""
InfraDoc 2.0 - Command Line Interface
Main CLI for infrastructure analysis and documentation generation.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

# Import core components
try:
    from infradoc_core import ConnectionConfig
    from infradoc_analyzer import InfrastructureAnalyzer, AnalysisConfig, quick_analysis, deep_analysis
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
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        warnings.append("No LLM API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY for AI features")
    
    # Check SSH key file if provided
    ssh_key = os.getenv("SSH_KEY_FILE")
    if ssh_key and not os.path.exists(os.path.expanduser(ssh_key)):
        errors.append(f"SSH key file not found: {ssh_key}")
    
    # Check for required Python packages
    try:
        import paramiko
    except ImportError:
        errors.append("paramiko package required. Install with: pip install paramiko")
    
    return errors, warnings

def print_banner():
    """Print InfraDoc banner."""
    print("""
InfraDoc 2.0 - Intelligent Infrastructure Analysis
==================================================
    AI-Powered Infrastructure Discovery & Documentation
    
    * LLM-guided analysis    * Comprehensive reports
    * Smart discovery        * Security assessment  
    * Auto-documentation     * Architecture mapping
""")

def cmd_analyze(args):
    """Execute infrastructure analysis command."""
    print_banner()
    
    logger.info(f"🚀 Starting infrastructure analysis for {args.host}")
    
    # Validate environment
    errors, warnings = validate_environment()
    
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    
    for warning in warnings:
        print(f"WARNING: {warning}")
    
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
        include_documentation=not args.no_docs
    )
    
    # Determine LLM providers
    llm_providers = []
    if not args.no_ai:
        if os.getenv("OPENAI_API_KEY"):
            llm_providers.append({"provider": "openai", "model": "gpt-4o"})
        if os.getenv("ANTHROPIC_API_KEY"):
            llm_providers.append({"provider": "claude", "model": "claude-3-5-sonnet-20241022"})
        if os.getenv("GROK_API_KEY"):
            llm_providers.append({"provider": "grok", "model": "grok-3"})
    
    if not llm_providers and not args.no_ai:
        print("WARNING: No LLM providers available, running basic analysis only")
        analysis_config.enable_ai = False
    
    analyzer = None
    try:
        # Initialize analyzer
        analyzer = InfrastructureAnalyzer(
            llm_providers=llm_providers if llm_providers else None,
            output_base_dir=args.output_dir
        )
        
        # Run analysis
        result = analyzer.analyze_infrastructure(connection_config, analysis_config)
        
        if result.success:
            print("\nANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print(f"Processes analyzed: {len(result.scan_report.processes)}")
            print(f"Files discovered: {len(result.scan_report.application_files)}")
            print(f"Architecture: {result.scan_report.infrastructure_insights.architecture_pattern}")
            print(f"Artifacts generated: {len(result.artifacts_generated)}")
            print(f"Documentation: {'YES' if result.documentation_generated else 'NO'}")
            print(f"Output directory: {result.output_directory}")
            
            # Show generated files
            print(f"\nGenerated Files:")
            for artifact in result.artifacts_generated:
                print(f"   * {Path(artifact).name}")
            
            if result.documentation_generated:
                docs_dir = Path(result.output_directory) / "documentation"
                if docs_dir.exists():
                    print(f"\nDocumentation Files:")
                    for doc_file in docs_dir.glob("*.md"):
                        print(f"   * {doc_file.name}")
            
            return 0
        else:
            print(f"ANALYSIS FAILED: {result.error_message}")
            return 1
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
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
    print("Running QUICK infrastructure analysis...")
    
    try:
        result = quick_analysis(
            host=args.host,
            username=args.username,
            key_file=args.key_file,
            password=args.password
        )
        
        if result.success:
            print(f"Quick analysis completed!")
            print(f"Found {len(result.scan_report.processes)} processes")
            print(f"Found {len(result.scan_report.application_files)} files")
            print(f"Results: {result.output_directory}")
            return 0
        else:
            print(f"Quick analysis failed: {result.error_message}")
            return 1
            
    except Exception as e:
        print(f"Quick analysis error: {e}")
        return 1

def cmd_deep(args):
    """Execute deep analysis command."""
    print("Running DEEP infrastructure analysis with full AI...")
    
    try:
        result = deep_analysis(
            host=args.host,
            username=args.username,
            key_file=args.key_file,
            password=args.password
        )
        
        if result.success:
            print(f"Deep analysis completed!")
            print(f"LLM calls: {result.scan_report.llm_analysis_summary.get('total_llm_calls', 0)}")
            print(f"Processes: {len(result.scan_report.processes)}")
            print(f"Files: {len(result.scan_report.application_files)}")
            print(f"Documentation: {'YES' if result.documentation_generated else 'NO'}")
            print(f"Results: {result.output_directory}")
            return 0
        else:
            print(f"Deep analysis failed: {result.error_message}")
            return 1
            
    except Exception as e:
        print(f"Deep analysis error: {e}")
        return 1

def cmd_version(args):
    """Show version information."""
    print("InfraDoc 2.0 - Intelligent Infrastructure Analysis")
    print("Version: 2.0.0")
    print("Author: InfraDoc Team")
    print()
    print("Features:")
    print("  * AI-powered analysis with LLM orchestration")
    print("  * Smart infrastructure discovery")
    print("  * Comprehensive documentation generation")
    print("  * Security assessment and recommendations")
    print("  * Architecture pattern recognition")
    return 0

def cmd_validate(args):
    """Validate environment and configuration."""
    print("Validating InfraDoc environment...")
    
    errors, warnings = validate_environment()
    
    # Check API keys
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Grok": os.getenv("GROK_API_KEY")
    }
    
    print(f"\nAPI Keys Status:")
    for provider, key in api_keys.items():
        status = "SET" if key else "NOT SET"
        print(f"   {provider}: {status}")
    
    # Check dependencies
    print(f"\nDependencies:")
    try:
        import paramiko
        print(f"   paramiko: {paramiko.__version__}")
    except ImportError:
        print(f"   paramiko: Not installed")
    
    try:
        import openai
        print(f"   openai: Available")
    except ImportError:
        print(f"   openai: Not installed (optional for OpenAI)")
    
    try:
        import anthropic
        print(f"   anthropic: Available")
    except ImportError:
        print(f"   anthropic: Not installed (optional for Claude)")
    
    # Show configuration
    if args.host:
        print(f"\nConnection Test:")
        print(f"   Host: {args.host}")
        print(f"   Username: {args.username}")
        print(f"   SSH Key: {args.key_file if args.key_file else 'Not specified'}")
        
        # Test connection if possible
        try:
            from infradoc_core import SSHConnector
            connector = SSHConnector()
            config = ConnectionConfig(
                host=args.host,
                username=args.username,
                key_file=args.key_file,
                password=args.password
            )
            
            print(f"   Testing connection...")
            if connector.connect(config):
                print(f"   Connection successful")
                connector.close_all_connections()
            else:
                print(f"   Connection failed")
        except Exception as e:
            print(f"   Connection test error: {e}")
    
    # Summary
    print(f"\nValidation Summary:")
    if errors:
        print(f"   {len(errors)} errors found")
        for error in errors:
            print(f"      - {error}")
    
    if warnings:
        print(f"   {len(warnings)} warnings")
        for warning in warnings:
            print(f"      - {warning}")
    
    if not errors:
        print(f"   Environment ready for InfraDoc analysis")
        return 0
    else:
        return 1

def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="InfraDoc 2.0 - Intelligent Infrastructure Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick analysis
  python infradoc_cli.py quick --host server.example.com
  
  # Deep analysis with full AI
  python infradoc_cli.py deep --host server.example.com
  
  # Custom analysis
  python infradoc_cli.py analyze --host server.example.com --depth deep --no-docs
  
  # Validate environment
  python infradoc_cli.py validate --host server.example.com
  
Environment Variables:
  OPENAI_API_KEY     - OpenAI API key for GPT models
  ANTHROPIC_API_KEY  - Anthropic API key for Claude models  
  GROK_API_KEY       - Grok API key for Grok models
  SSH_KEY_FILE       - Default SSH key file path
        """
    )
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')
    parser.add_argument('--debug', action='store_true', help='Debug mode with stack traces')
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run comprehensive infrastructure analysis')
    analyze_parser.add_argument('--host', required=True, help='Target hostname or IP address')
    analyze_parser.add_argument('--port', '-p', type=int, default=22, help='SSH port (default: 22)')
    analyze_parser.add_argument('--username', '-u', default='ubuntu', help='SSH username (default: ubuntu)')
    analyze_parser.add_argument('--key-file', '-k', help='SSH private key file path')
    analyze_parser.add_argument('--password', help='SSH password (not recommended)')
    analyze_parser.add_argument('--timeout', type=int, default=30, help='SSH timeout in seconds (default: 30)')
    analyze_parser.add_argument('--retries', type=int, default=3, help='SSH connection retries (default: 3)')
    analyze_parser.add_argument('--depth', choices=['quick', 'standard', 'deep'], default='standard', help='Analysis depth')
    analyze_parser.add_argument('--output-dir', '-o', default='infradoc_analysis', help='Output directory')
    analyze_parser.add_argument('--no-ai', action='store_true', help='Disable AI analysis')
    analyze_parser.add_argument('--max-llm-calls', type=int, default=15, help='Maximum LLM calls (default: 15)')
    analyze_parser.add_argument('--output-formats', nargs='+', default=['json', 'markdown'], help='Output formats')
    analyze_parser.add_argument('--no-artifacts', action='store_true', help='Skip artifact generation')
    analyze_parser.add_argument('--no-security', action='store_true', help='Skip security analysis')
    analyze_parser.add_argument('--no-docs', action='store_true', help='Skip documentation generation')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Quick command
    quick_parser = subparsers.add_parser('quick', help='Run quick analysis (5 minutes)')
    quick_parser.add_argument('--host', required=True, help='Target hostname or IP address')
    quick_parser.add_argument('--username', '-u', default='ubuntu', help='SSH username (default: ubuntu)')
    quick_parser.add_argument('--key-file', '-k', help='SSH private key file path')
    quick_parser.add_argument('--password', help='SSH password (not recommended)')
    quick_parser.set_defaults(func=cmd_quick)
    
    # Deep command  
    deep_parser = subparsers.add_parser('deep', help='Run deep analysis with full AI (15+ minutes)')
    deep_parser.add_argument('--host', required=True, help='Target hostname or IP address')
    deep_parser.add_argument('--username', '-u', default='ubuntu', help='SSH username (default: ubuntu)')
    deep_parser.add_argument('--key-file', '-k', help='SSH private key file path')
    deep_parser.add_argument('--password', help='SSH password (not recommended)')
    deep_parser.set_defaults(func=cmd_deep)
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    version_parser.set_defaults(func=cmd_version)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate environment and configuration')
    validate_parser.add_argument('--host', help='Test connection to host')
    validate_parser.add_argument('--username', '-u', default='ubuntu', help='SSH username for connection test')
    validate_parser.add_argument('--key-file', '-k', help='SSH private key file for connection test')
    validate_parser.add_argument('--password', help='SSH password for connection test')
    validate_parser.set_defaults(func=cmd_validate)
    
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
        parser.print_help()
        sys.exit(0)
    
    # Execute command
    try:
        exit_code = args.func(args)
        print(f"\nInfraDoc analysis completed.")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()