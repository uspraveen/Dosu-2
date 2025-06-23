#!/usr/bin/env python3
"""
InfraDoc 2.0 - Main Infrastructure Analyzer
Progressive analysis orchestrator that coordinates discovery, intelligence, and documentation.
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from infradoc_core import (
    SSHConnector, LLMOrchestrator, SmartDiscovery, 
    ConnectionConfig, ProcessInfo, ApplicationFile, InfrastructureInsights
)

logger = logging.getLogger(__name__)

# ================================================================================
# ANALYSIS CONFIGURATION AND RESULTS
# ================================================================================

@dataclass
class AnalysisConfig:
    """Configuration for infrastructure analysis."""
    scan_depth: str = "standard"  # standard, deep, quick
    enable_ai: bool = True
    max_llm_calls: int = 15
    output_formats: List[str] = None
    export_artifacts: bool = True
    include_security: bool = True
    include_documentation: bool = True

@dataclass
class ScanReport:
    """Complete scan report with all analysis results."""
    host: str
    scan_id: str
    timestamp: str
    scan_duration: float
    
    # Core results
    processes: List[ProcessInfo]
    application_files: List[ApplicationFile]
    infrastructure_insights: InfrastructureInsights
    security_analysis: Dict[str, Any]
    
    # Metadata
    llm_analysis_summary: Dict[str, Any]
    scan_statistics: Dict[str, Any]
    analysis_stages: List[Dict[str, Any]]

@dataclass
class AnalysisResult:
    """Complete analysis result with all generated artifacts."""
    scan_report: ScanReport
    artifacts_generated: List[str]
    output_directory: str
    documentation_generated: bool
    success: bool
    error_message: Optional[str] = None

class InfrastructureAnalyzer:
    """
    Main orchestrator for progressive infrastructure analysis.
    Coordinates discovery, intelligence, and documentation generation.
    """
    
    def __init__(self, llm_providers: List[Dict] = None, output_base_dir: str = "infradoc_analysis"):
        """
        Initialize the Infrastructure Analyzer.
        
        Args:
            llm_providers: List of LLM provider configurations
            output_base_dir: Base directory for output files
        """
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
        
        # Initialize core components
        self.ssh_connector = SSHConnector()
        
        # Initialize LLM orchestrator if providers available
        self.llm_orchestrator = None
        if llm_providers:
            try:
                self.llm_orchestrator = LLMOrchestrator(llm_providers)
                logger.info("[ANALYZER] LLM orchestrator initialized")
            except Exception as e:
                logger.warning(f"⚠️ LLM initialization failed: {e}")
        
        # Initialize smart discovery
        self.smart_discovery = None
        if self.llm_orchestrator:
            self.smart_discovery = SmartDiscovery(self.ssh_connector, self.llm_orchestrator)
            logger.info("[ANALYZER] Smart discovery initialized")
        
        logger.info("[ANALYZER] Infrastructure Analyzer initialized")
    
    def analyze_infrastructure(self, connection_config: ConnectionConfig, 
                             analysis_config: AnalysisConfig = None) -> AnalysisResult:
        """
        Perform complete infrastructure analysis.
        
        Args:
            connection_config: SSH connection configuration
            analysis_config: Analysis configuration options
            
        Returns:
            Complete analysis result with all artifacts
        """
        if analysis_config is None:
            analysis_config = AnalysisConfig()
        
        scan_id = f"scan_{int(time.time())}"
        start_time = time.time()
        timestamp = datetime.now()
        
        # Create output directory for this scan
        output_dir = self.output_base_dir / f"infradoc_{scan_id}"
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"[ANALYZER] Starting infrastructure analysis: {scan_id}")
        logger.info(f"[ANALYZER] Target: {connection_config.host}")
        logger.info(f"[ANALYZER] Output: {output_dir}")
        
        try:
            # Stage 1: Establish Connection
            if not self._establish_connection(connection_config):
                return AnalysisResult(
                    scan_report=None,
                    artifacts_generated=[],
                    output_directory=str(output_dir),
                    documentation_generated=False,
                    success=False,
                    error_message="Failed to establish SSH connection"
                )
            
            # Stage 2: Perform Discovery
            if analysis_config.enable_ai and self.smart_discovery:
                discovery_results = self._perform_smart_discovery(connection_config.host, analysis_config)
            else:
                discovery_results = self._perform_basic_discovery(connection_config.host)
            
            # Stage 3: Create Scan Report
            scan_duration = time.time() - start_time
            scan_report = self._create_scan_report(
                host=connection_config.host,
                scan_id=scan_id,
                timestamp=timestamp.isoformat(),
                scan_duration=scan_duration,
                discovery_results=discovery_results,
                analysis_config=analysis_config
            )
            
            # Stage 4: Generate Artifacts
            artifacts = self._generate_artifacts(scan_report, output_dir, analysis_config)
            
            # Stage 5: Generate Documentation
            documentation_generated = False
            if analysis_config.include_documentation:
                documentation_generated = self._generate_documentation(scan_report, output_dir)
            
            # Create final result
            result = AnalysisResult(
                scan_report=scan_report,
                artifacts_generated=artifacts,
                output_directory=str(output_dir),
                documentation_generated=documentation_generated,
                success=True
            )
            
            # Log summary
            self._log_analysis_summary(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return AnalysisResult(
                scan_report=None,
                artifacts_generated=[],
                output_directory=str(output_dir),
                documentation_generated=False,
                success=False,
                error_message=str(e)
            )
        finally:
            # Ensure cleanup always happens
            try:
                self._cleanup_resources()
            except Exception as cleanup_error:
                logger.warning(f"Cleanup warning: {cleanup_error}")
    
    def _cleanup_resources(self):
        """Clean up all resources."""
        try:
            # Close SSH connections
            if hasattr(self, 'ssh_connector') and self.ssh_connector:
                self.ssh_connector.close_all_connections()
            
            # Clean up smart discovery
            if hasattr(self, 'smart_discovery') and self.smart_discovery:
                self.smart_discovery.cleanup()
                
            logger.info("[ANALYZER] Resources cleaned up successfully")
        except Exception as e:
            logger.warning(f"[ANALYZER] Error during cleanup: {e}")
    
    def _establish_connection(self, config: ConnectionConfig) -> bool:
        """Establish SSH connection to target host."""
        logger.info(f"[ANALYZER] Establishing connection to {config.host}")
        
        success = self.ssh_connector.connect(config)
        if success:
            logger.info(f"[ANALYZER] Connected to {config.host}")
        else:
            logger.error(f"[ANALYZER] Failed to connect to {config.host}")
        
        return success
    
    def _perform_smart_discovery(self, host: str, config: AnalysisConfig) -> Dict[str, Any]:
        """Perform AI-powered smart discovery."""
        logger.info("[ANALYZER] Performing smart discovery with AI")
        
        try:
            discovery_results = self.smart_discovery.discover_infrastructure(host)
            logger.info("[ANALYZER] Smart discovery completed")
            return discovery_results
        except Exception as e:
            logger.error(f"[ANALYZER] Smart discovery failed: {e}")
            logger.info("[ANALYZER] Falling back to basic discovery")
            return self._perform_basic_discovery(host)
    
    def _perform_basic_discovery(self, host: str) -> Dict[str, Any]:
        """Perform basic discovery without AI."""
        logger.info("[ANALYZER] Performing basic discovery")
        
        # Get basic process list
        stdout, stderr, exit_code, execution = self.ssh_connector.execute_command(
            host, "ps aux --no-headers", "basic_process_discovery"
        )
        
        processes = []
        if stdout:
            for line in stdout.split('\n')[:20]:  # Limit to 20 processes
                if line.strip():
                    parts = line.split(None, 10)
                    if len(parts) >= 11:
                        process = ProcessInfo(
                            pid=int(parts[1]),
                            name=parts[10].split()[0],
                            user=parts[0],
                            cpu_percent=parts[2],
                            memory_percent=parts[3],
                            command=parts[10],
                            service_classification="unknown",
                            service_purpose="Basic process discovery"
                        )
                        processes.append(process)
        
        # Get basic file listing
        stdout, stderr, exit_code, execution = self.ssh_connector.execute_command(
            host, "find /opt /srv /var/www -name '*.py' -o -name '*.js' 2>/dev/null | head -10", 
            "basic_file_discovery"
        )
        
        files = []
        if stdout:
            for file_path in stdout.strip().split('\n'):
                if file_path.strip():
                    file_obj = ApplicationFile(
                        path=file_path,
                        language=self._detect_language_simple(file_path),
                        size=0,
                        last_modified="unknown"
                    )
                    files.append(file_obj)
        
        # Basic insights
        insights = InfrastructureInsights(
            architecture_pattern="Standard deployment",
            technology_stack=["Linux"],
            deployment_model="Server-based",
            scalability_assessment="Unknown",
            security_posture="Needs assessment",
            operational_complexity="Standard",
            recommendations=["Enable AI analysis for detailed insights"]
        )
        
        return {
            'processes': processes,
            'files': files,
            'infrastructure_insights': insights,
            'security_analysis': {"analysis": "Basic security scan completed"},
            'llm_analysis_summary': {"total_llm_calls": 0, "analysis_stages": 0, "overall_confidence": 0.5}
        }
    
    def _detect_language_simple(self, file_path: str) -> str:
        """Simple language detection from file extension."""
        extension = Path(file_path).suffix.lower()
        lang_map = {'.py': 'Python', '.js': 'JavaScript', '.java': 'Java'}
        return lang_map.get(extension, 'Unknown')
    
    def _create_scan_report(self, host: str, scan_id: str, timestamp: str, 
                           scan_duration: float, discovery_results: Dict[str, Any],
                           analysis_config: AnalysisConfig) -> ScanReport:
        """Create comprehensive scan report."""
        logger.info("[ANALYZER] Creating scan report")
        
        # Extract infrastructure insights
        insights_data = discovery_results.get('infrastructure_insights', {})
        if isinstance(insights_data, dict):
            infrastructure_insights = InfrastructureInsights(
                architecture_pattern=insights_data.get('architecture_pattern', 'Unknown'),
                technology_stack=insights_data.get('technology_stack', []),
                deployment_model=insights_data.get('deployment_model', 'Unknown'),
                scalability_assessment=insights_data.get('scalability_assessment', 'Unknown'),
                security_posture=insights_data.get('security_posture', 'Unknown'),
                operational_complexity=insights_data.get('operational_complexity', 'Unknown'),
                recommendations=insights_data.get('recommendations', [])
            )
        else:
            infrastructure_insights = insights_data
        
        # Create scan statistics
        scan_statistics = {
            "processes_analyzed": len(discovery_results.get('processes', [])),
            "files_analyzed": len(discovery_results.get('files', [])),
            "analysis_depth": analysis_config.scan_depth,
            "ai_enabled": analysis_config.enable_ai,
            "total_commands_executed": len(self.ssh_connector.command_history)
        }
        
        # Create analysis stages summary
        analysis_stages = [
            {"stage": "connection", "status": "completed", "timestamp": timestamp},
            {"stage": "discovery", "status": "completed", "timestamp": timestamp},
            {"stage": "analysis", "status": "completed", "timestamp": timestamp}
        ]
        
        scan_report = ScanReport(
            host=host,
            scan_id=scan_id,
            timestamp=timestamp,
            scan_duration=scan_duration,
            processes=discovery_results.get('processes', []),
            application_files=discovery_results.get('files', []),
            infrastructure_insights=infrastructure_insights,
            security_analysis=discovery_results.get('security_analysis', {}),
            llm_analysis_summary=discovery_results.get('llm_analysis_summary', {}),
            scan_statistics=scan_statistics,
            analysis_stages=analysis_stages
        )
        
        logger.info("[ANALYZER] Scan report created")
        return scan_report
    
    def _generate_artifacts(self, scan_report: ScanReport, output_dir: Path, 
                           config: AnalysisConfig) -> List[str]:
        """Generate analysis artifacts."""
        logger.info("[ANALYZER] Generating analysis artifacts")
        
        artifacts = []
        
        # Generate JSON report
        json_file = output_dir / f"infradoc_scan_{scan_report.scan_id}.json"
        with open(json_file, 'w') as f:
            # Convert dataclasses to dicts for JSON serialization
            report_dict = self._scan_report_to_dict(scan_report)
            json.dump(report_dict, f, indent=2, default=str)
        artifacts.append(str(json_file))
        
        # Generate markdown summary
        if "markdown" in (config.output_formats or ["json", "markdown"]):
            md_file = output_dir / f"infrastructure_analysis_{scan_report.scan_id}.md"
            markdown_content = self._generate_markdown_report(scan_report)
            with open(md_file, 'w') as f:
                f.write(markdown_content)
            artifacts.append(str(md_file))
        
        # Generate detailed text report
        txt_file = output_dir / f"detailed_analysis_{scan_report.scan_id}.txt"
        text_content = self._generate_text_report(scan_report)
        with open(txt_file, 'w') as f:
            f.write(text_content)
        artifacts.append(str(txt_file))
        
        logger.info(f"[ANALYZER] Generated {len(artifacts)} artifacts")
        return artifacts
    
    def _scan_report_to_dict(self, scan_report: ScanReport) -> Dict[str, Any]:
        """Convert scan report to dictionary for JSON serialization."""
        return {
            "host": scan_report.host,
            "scan_id": scan_report.scan_id,
            "timestamp": scan_report.timestamp,
            "scan_duration": scan_report.scan_duration,
            "processes": [asdict(p) for p in scan_report.processes],
            "application_files": [asdict(f) for f in scan_report.application_files],
            "infrastructure_insights": asdict(scan_report.infrastructure_insights),
            "security_analysis": scan_report.security_analysis,
            "llm_analysis_summary": scan_report.llm_analysis_summary,
            "scan_statistics": scan_report.scan_statistics,
            "analysis_stages": scan_report.analysis_stages
        }
    
    def _generate_markdown_report(self, scan_report: ScanReport) -> str:
        """Generate markdown analysis report."""
        content = f"""# Infrastructure Analysis Report

## Overview
- **Host**: {scan_report.host}
- **Scan ID**: {scan_report.scan_id}
- **Analysis Date**: {scan_report.timestamp[:10]}
- **Duration**: {scan_report.scan_duration:.2f} seconds

## Summary
- **Processes Analyzed**: {len(scan_report.processes)}
- **Files Discovered**: {len(scan_report.application_files)}
- **Architecture Pattern**: {scan_report.infrastructure_insights.architecture_pattern}
- **Security Posture**: {scan_report.infrastructure_insights.security_posture}

## Technology Stack
"""
        for tech in scan_report.infrastructure_insights.technology_stack:
            content += f"- {tech}\n"
        
        content += f"""
## Key Processes
"""
        for proc in scan_report.processes[:10]:
            content += f"- **PID {proc.pid}**: {proc.name} ({proc.service_classification})\n"
        
        content += f"""
## Application Files
"""
        file_types = {}
        for file in scan_report.application_files:
            if file.language not in file_types:
                file_types[file.language] = 0
            file_types[file.language] += 1
        
        for lang, count in file_types.items():
            content += f"- **{lang}**: {count} files\n"
        
        content += f"""
## Recommendations
"""
        for rec in scan_report.infrastructure_insights.recommendations:
            content += f"- {rec}\n"
        
        content += f"""

---
*Generated by InfraDoc 2.0 - Intelligent Infrastructure Analysis*
"""
        return content
    
    def _generate_text_report(self, scan_report: ScanReport) -> str:
        """Generate detailed text report."""
        content = f"""
INFRADOC 2.0 - DETAILED INFRASTRUCTURE ANALYSIS REPORT
======================================================

SCAN INFORMATION
================
Host: {scan_report.host}
Scan ID: {scan_report.scan_id}
Timestamp: {scan_report.timestamp}
Duration: {scan_report.scan_duration:.2f} seconds

ANALYSIS SUMMARY
================
Processes Analyzed: {len(scan_report.processes)}
Files Discovered: {len(scan_report.application_files)}
Architecture Pattern: {scan_report.infrastructure_insights.architecture_pattern}
Deployment Model: {scan_report.infrastructure_insights.deployment_model}
Security Posture: {scan_report.infrastructure_insights.security_posture}
Operational Complexity: {scan_report.infrastructure_insights.operational_complexity}

TECHNOLOGY STACK
================"""
        
        for i, tech in enumerate(scan_report.infrastructure_insights.technology_stack, 1):
            content += f"\n{i}. {tech}"
        
        content += f"""

DISCOVERED PROCESSES
===================="""
        
        for proc in scan_report.processes:
            content += f"""
PID: {proc.pid}
Name: {proc.name}
User: {proc.user}
Classification: {proc.service_classification}
Purpose: {proc.service_purpose}
Command: {proc.command[:100]}...
"""
        
        content += f"""

DISCOVERED FILES
================"""
        
        for file in scan_report.application_files[:20]:
            content += f"""
Path: {file.path}
Language: {file.language}
Size: {file.size} bytes
Modified: {file.last_modified}
"""
        
        content += f"""

SECURITY ANALYSIS
=================
{scan_report.security_analysis.get('analysis', 'No detailed security analysis available')}

RECOMMENDATIONS
==============="""
        
        for i, rec in enumerate(scan_report.infrastructure_insights.recommendations, 1):
            content += f"\n{i}. {rec}"
        
        content += f"""

LLM ANALYSIS SUMMARY
====================
Total LLM Calls: {scan_report.llm_analysis_summary.get('total_llm_calls', 0)}
Analysis Stages: {scan_report.llm_analysis_summary.get('analysis_stages', 0)}
Overall Confidence: {scan_report.llm_analysis_summary.get('overall_confidence', 0):.0%}

SCAN STATISTICS
===============
Commands Executed: {scan_report.scan_statistics.get('total_commands_executed', 0)}
Analysis Depth: {scan_report.scan_statistics.get('analysis_depth', 'standard')}
AI Enabled: {scan_report.scan_statistics.get('ai_enabled', False)}

---
Generated by InfraDoc 2.0 - Intelligent Infrastructure Analysis
Analysis completed at {datetime.now().isoformat()}
"""
        return content
    
    def _generate_documentation(self, scan_report: ScanReport, output_dir: Path) -> bool:
        """Generate comprehensive documentation."""
        logger.info("[ANALYZER] Generating comprehensive documentation")
        
        try:
            # Import documentation generator
            from infradoc_docs import DocumentationGenerator
            
            doc_generator = DocumentationGenerator(scan_report, str(output_dir))
            success = doc_generator.generate_all_documentation()
            
            if success:
                logger.info("[ANALYZER] Documentation generated successfully")
            else:
                logger.warning("[ANALYZER] Documentation generation had issues")
            
            return success
            
        except ImportError:
            logger.warning("[ANALYZER] Documentation generator not available")
            return False
        except Exception as e:
            logger.error(f"[ANALYZER] Documentation generation failed: {e}")
            return False
    
    def _log_analysis_summary(self, result: AnalysisResult):
        """Log analysis completion summary."""
        if result.success:
            logger.info("[ANALYZER] ANALYSIS COMPLETED SUCCESSFULLY")
            logger.info(f"[ANALYZER] Processes: {len(result.scan_report.processes)}")
            logger.info(f"[ANALYZER] Files: {len(result.scan_report.application_files)}")
            logger.info(f"[ANALYZER] Architecture: {result.scan_report.infrastructure_insights.architecture_pattern}")
            logger.info(f"[ANALYZER] Artifacts: {len(result.artifacts_generated)}")
            logger.info(f"[ANALYZER] Documentation: {'YES' if result.documentation_generated else 'NO'}")
            logger.info(f"[ANALYZER] Output: {result.output_directory}")
        else:
            logger.error(f"[ANALYZER] ANALYSIS FAILED: {result.error_message}")

# ================================================================================
# CONVENIENCE FUNCTIONS
# ================================================================================

def quick_analysis(host: str, username: str = "ubuntu", key_file: str = None, 
                  password: str = None) -> AnalysisResult:
    """
    Perform quick infrastructure analysis with minimal configuration.
    
    Args:
        host: Target hostname or IP
        username: SSH username
        key_file: Path to SSH private key
        password: SSH password (if not using key)
        
    Returns:
        Analysis result
    """
    # Configure connection
    connection_config = ConnectionConfig(
        host=host,
        username=username,
        key_file=key_file,
        password=password
    )
    
    # Configure analysis for quick scan
    analysis_config = AnalysisConfig(
        scan_depth="quick",
        enable_ai=True,
        max_llm_calls=5,
        include_documentation=False
    )
    
    # Determine LLM providers
    llm_providers = []
    if os.getenv("OPENAI_API_KEY"):
        llm_providers.append({"provider": "openai", "model": "gpt-4o"})
    if os.getenv("ANTHROPIC_API_KEY"):
        llm_providers.append({"provider": "claude", "model": "claude-3-5-sonnet-20241022"})
    
    # Initialize and run analyzer
    analyzer = InfrastructureAnalyzer(llm_providers)
    try:
        return analyzer.analyze_infrastructure(connection_config, analysis_config)
    finally:
        # Ensure cleanup
        analyzer._cleanup_resources()

def deep_analysis(host: str, username: str = "ubuntu", key_file: str = None, 
                 password: str = None) -> AnalysisResult:
    """
    Perform comprehensive deep infrastructure analysis.
    
    Args:
        host: Target hostname or IP
        username: SSH username
        key_file: Path to SSH private key
        password: SSH password (if not using key)
        
    Returns:
        Analysis result
    """
    # Configure connection
    connection_config = ConnectionConfig(
        host=host,
        username=username,
        key_file=key_file,
        password=password
    )
    
    # Configure analysis for deep scan
    analysis_config = AnalysisConfig(
        scan_depth="deep",
        enable_ai=True,
        max_llm_calls=25,
        include_documentation=True,
        include_security=True
    )
    
    # Determine LLM providers
    llm_providers = []
    if os.getenv("GROK_API_KEY"):
        llm_providers.append({"provider": "grok", "model": "grok-3"})
    if os.getenv("OPENAI_API_KEY"):
        llm_providers.append({"provider": "openai", "model": "gpt-4o-mini"})
    if os.getenv("ANTHROPIC_API_KEY"):
        llm_providers.append({"provider": "claude", "model": "claude-3-5-sonnet-20241022"})
    
    
    # Initialize and run analyzer
    analyzer = InfrastructureAnalyzer(llm_providers)
    try:
        return analyzer.analyze_infrastructure(connection_config, analysis_config)
    finally:
        # Ensure cleanup
        analyzer._cleanup_resources()