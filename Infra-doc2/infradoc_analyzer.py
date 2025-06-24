#!/usr/bin/env python3
"""
InfraDoc 2.0 - Enhanced Infrastructure Analyzer
Progressive analysis orchestrator with intelligent discovery and business understanding.
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
    ConnectionConfig, ProcessInfo, ApplicationFile, EnhancedApplicationFile, InfrastructureInsights
)

logger = logging.getLogger(__name__)

# ================================================================================
# ENHANCED ANALYSIS CONFIGURATION AND RESULTS
# ================================================================================

@dataclass
class AnalysisConfig:
    """Configuration for infrastructure analysis."""
    scan_depth: str = "standard"  # standard, deep, quick, intelligent
    enable_ai: bool = True
    max_llm_calls: int = 15
    output_formats: List[str] = None
    export_artifacts: bool = True
    include_security: bool = True
    include_documentation: bool = True
    enable_business_intelligence: bool = False
    enable_enhanced_analysis: bool = False

@dataclass
class EnhancedScanReport:
    """Enhanced scan report with intelligence and business understanding."""
    host: str
    scan_id: str
    timestamp: str
    scan_duration: float
    
    # Core results (enhanced)
    processes: List[ProcessInfo]
    application_files: List[EnhancedApplicationFile]
    infrastructure_insights: InfrastructureInsights
    security_analysis: Dict[str, Any]
    
    # Enhanced intelligence
    business_intelligence: Dict[str, Any] = None
    api_documentation: Dict[str, Any] = None
    deployment_intelligence: Dict[str, Any] = None
    
    # Metadata
    llm_analysis_summary: Dict[str, Any] = None
    scan_statistics: Dict[str, Any] = None
    analysis_stages: List[Dict[str, Any]] = None

@dataclass
class AnalysisResult:
    """Complete analysis result with all generated artifacts."""
    scan_report: EnhancedScanReport
    artifacts_generated: List[str]
    output_directory: str
    documentation_generated: bool
    success: bool
    error_message: Optional[str] = None

class IntelligentInfrastructureAnalyzer:
    """
    Enhanced orchestrator for intelligent infrastructure analysis.
    Coordinates discovery, intelligence extraction, and documentation generation.
    """
    
    def __init__(self, llm_providers: List[Dict] = None, output_base_dir: str = "infradoc_analysis"):
        """
        Initialize the Intelligent Infrastructure Analyzer.
        
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
        
        logger.info("[ANALYZER] Intelligent Infrastructure Analyzer initialized")
    
    def analyze_infrastructure(self, connection_config: ConnectionConfig, 
                             analysis_config: AnalysisConfig = None) -> AnalysisResult:
        """
        Perform complete intelligent infrastructure analysis.
        
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
        
        logger.info(f"[ANALYZER] Starting intelligent infrastructure analysis: {scan_id}")
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
            
            # Stage 2: Perform Enhanced Discovery
            if analysis_config.enable_ai and self.smart_discovery:
                discovery_results = self._perform_intelligent_discovery(connection_config.host, analysis_config)
            else:
                discovery_results = self._perform_basic_discovery(connection_config.host)
            
            # Stage 3: Create Enhanced Scan Report
            scan_duration = time.time() - start_time
            scan_report = self._create_enhanced_scan_report(
                host=connection_config.host,
                scan_id=scan_id,
                timestamp=timestamp.isoformat(),
                scan_duration=scan_duration,
                discovery_results=discovery_results,
                analysis_config=analysis_config
            )
            
            # Stage 4: Generate Artifacts
            artifacts = self._generate_enhanced_artifacts(scan_report, output_dir, analysis_config)
            
            # Stage 5: Generate Intelligent Documentation
            documentation_generated = False
            if analysis_config.include_documentation:
                documentation_generated = self._generate_intelligent_documentation(scan_report, output_dir)
            
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
    
    def _perform_intelligent_discovery(self, host: str, config: AnalysisConfig) -> Dict[str, Any]:
        """Perform AI-powered intelligent discovery with business understanding."""
        logger.info("[ANALYZER] Performing intelligent discovery with business intelligence")
        
        try:
            discovery_results = self.smart_discovery.discover_infrastructure(host)
            
            # Extract business intelligence if available
            if 'business_intelligence' in discovery_results:
                logger.info("[ANALYZER] Business intelligence extracted successfully")
            
            # Extract API documentation if available  
            api_docs = self._extract_api_documentation(discovery_results)
            if api_docs:
                discovery_results['api_documentation'] = api_docs
                logger.info("[ANALYZER] API documentation extracted")
            
            # Extract deployment intelligence
            deployment_intel = self._extract_deployment_intelligence(discovery_results)
            if deployment_intel:
                discovery_results['deployment_intelligence'] = deployment_intel
                logger.info("[ANALYZER] Deployment intelligence extracted")
            
            logger.info("[ANALYZER] Intelligent discovery completed")
            return discovery_results
        except Exception as e:
            logger.error(f"[ANALYZER] Intelligent discovery failed: {e}")
            logger.info("[ANALYZER] Falling back to basic discovery")
            return self._perform_basic_discovery(host)
    
    def _extract_api_documentation(self, discovery_results: Dict) -> Dict:
        """Extract API documentation from enhanced files."""
        api_docs = {
            'endpoints': [],
            'models': [],
            'authentication': [],
            'base_url': 'http://localhost'
        }
        
        enhanced_files = discovery_results.get('files', [])
        for file_info in enhanced_files:
            if hasattr(file_info, 'api_endpoints') and file_info.api_endpoints:
                api_docs['endpoints'].extend(file_info.api_endpoints)
            
            if hasattr(file_info, 'database_models') and file_info.database_models:
                api_docs['models'].extend(file_info.database_models)
        
        return api_docs if api_docs['endpoints'] or api_docs['models'] else None
    
    def _extract_deployment_intelligence(self, discovery_results: Dict) -> Dict:
        """Extract deployment intelligence from processes and files."""
        deployment_intel = {
            'deployment_type': 'unknown',
            'service_management': 'unknown',
            'containerization': False,
            'web_server': None,
            'process_manager': None,
            'environment_setup': []
        }
        
        processes = discovery_results.get('processes', [])
        
        # Detect deployment patterns
        for process in processes:
            if 'nginx' in process.command.lower():
                deployment_intel['web_server'] = 'nginx'
            elif 'systemd' in process.command.lower():
                deployment_intel['service_management'] = 'systemd'
            elif 'docker' in process.command.lower():
                deployment_intel['containerization'] = True
        
        # Extract environment variables
        enhanced_files = discovery_results.get('files', [])
        for file_info in enhanced_files:
            if hasattr(file_info, 'environment_variables') and file_info.environment_variables:
                deployment_intel['environment_setup'].extend(file_info.environment_variables)
        
        return deployment_intel
    
    def _perform_basic_discovery(self, host: str) -> Dict[str, Any]:
        """Perform basic discovery without AI (fallback)."""
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
    
    def _create_enhanced_scan_report(self, host: str, scan_id: str, timestamp: str, 
                                   scan_duration: float, discovery_results: Dict[str, Any],
                                   analysis_config: AnalysisConfig) -> EnhancedScanReport:
        """Create enhanced scan report with business intelligence."""
        logger.info("[ANALYZER] Creating enhanced scan report")
        
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
            "business_intelligence_enabled": analysis_config.enable_business_intelligence,
            "total_commands_executed": len(self.ssh_connector.command_history)
        }
        
        # Create analysis stages summary
        analysis_stages = [
            {"stage": "connection", "status": "completed", "timestamp": timestamp},
            {"stage": "discovery", "status": "completed", "timestamp": timestamp},
            {"stage": "analysis", "status": "completed", "timestamp": timestamp}
        ]
        
        if discovery_results.get('business_intelligence'):
            analysis_stages.append({"stage": "business_intelligence", "status": "completed", "timestamp": timestamp})
        
        scan_report = EnhancedScanReport(
            host=host,
            scan_id=scan_id,
            timestamp=timestamp,
            scan_duration=scan_duration,
            processes=discovery_results.get('processes', []),
            application_files=discovery_results.get('files', []),
            infrastructure_insights=infrastructure_insights,
            security_analysis=discovery_results.get('security_analysis', {}),
            business_intelligence=discovery_results.get('business_intelligence'),
            api_documentation=discovery_results.get('api_documentation'),
            deployment_intelligence=discovery_results.get('deployment_intelligence'),
            llm_analysis_summary=discovery_results.get('llm_analysis_summary', {}),
            scan_statistics=scan_statistics,
            analysis_stages=analysis_stages
        )
        
        logger.info("[ANALYZER] Enhanced scan report created")
        return scan_report
    
    def _generate_enhanced_artifacts(self, scan_report: EnhancedScanReport, output_dir: Path, 
                                   config: AnalysisConfig) -> List[str]:
        """Generate enhanced analysis artifacts."""
        logger.info("[ANALYZER] Generating enhanced analysis artifacts")
        
        artifacts = []
        
        # Generate Enhanced JSON report
        json_file = output_dir / f"infradoc_scan_{scan_report.scan_id}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            # Convert dataclasses to dicts for JSON serialization
            report_dict = self._enhanced_scan_report_to_dict(scan_report)
            json.dump(report_dict, f, indent=2, default=str)
        artifacts.append(str(json_file))
        
        # Generate enhanced markdown summary
        if "markdown" in (config.output_formats or ["json", "markdown"]):
            md_file = output_dir / f"infrastructure_analysis_{scan_report.scan_id}.md"
            markdown_content = self._generate_enhanced_markdown_report(scan_report)
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            artifacts.append(str(md_file))
        
        # Generate detailed text report
        txt_file = output_dir / f"detailed_analysis_{scan_report.scan_id}.txt"
        text_content = self._generate_enhanced_text_report(scan_report)
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        artifacts.append(str(txt_file))
        
        # Generate API documentation if available
        if scan_report.api_documentation:
            api_file = output_dir / f"api_documentation_{scan_report.scan_id}.md"
            api_content = self._generate_api_documentation_file(scan_report)
            with open(api_file, 'w', encoding='utf-8') as f:
                f.write(api_content)
            artifacts.append(str(api_file))
        
        # Generate business intelligence report if available
        if scan_report.business_intelligence:
            business_file = output_dir / f"business_intelligence_{scan_report.scan_id}.md"
            business_content = self._generate_business_intelligence_report(scan_report)
            with open(business_file, 'w', encoding='utf-8') as f:
                f.write(business_content)
            artifacts.append(str(business_file))
        
        logger.info(f"[ANALYZER] Generated {len(artifacts)} enhanced artifacts")
        return artifacts
    
    def _enhanced_scan_report_to_dict(self, scan_report: EnhancedScanReport) -> Dict[str, Any]:
        """Convert enhanced scan report to dictionary for JSON serialization."""
        return {
            "host": scan_report.host,
            "scan_id": scan_report.scan_id,
            "timestamp": scan_report.timestamp,
            "scan_duration": scan_report.scan_duration,
            "processes": [asdict(p) for p in scan_report.processes],
            "application_files": [asdict(f) for f in scan_report.application_files],
            "infrastructure_insights": asdict(scan_report.infrastructure_insights),
            "security_analysis": scan_report.security_analysis,
            "business_intelligence": scan_report.business_intelligence,
            "api_documentation": scan_report.api_documentation,
            "deployment_intelligence": scan_report.deployment_intelligence,
            "llm_analysis_summary": scan_report.llm_analysis_summary,
            "scan_statistics": scan_report.scan_statistics,
            "analysis_stages": scan_report.analysis_stages
        }
    
    def _generate_enhanced_markdown_report(self, scan_report: EnhancedScanReport) -> str:
        """Generate enhanced markdown analysis report."""
        content = f"""# 🧠 Intelligent Infrastructure Analysis Report

## 📊 Overview
- **Host**: {scan_report.host}
- **Scan ID**: {scan_report.scan_id}
- **Analysis Date**: {scan_report.timestamp[:10]}
- **Duration**: {scan_report.scan_duration:.2f} seconds

## 🎯 Executive Summary
- **Processes Analyzed**: {len(scan_report.processes)}
- **Files Discovered**: {len(scan_report.application_files)}
- **Architecture Pattern**: {scan_report.infrastructure_insights.architecture_pattern}
- **Security Posture**: {scan_report.infrastructure_insights.security_posture}

## 🔧 Technology Stack
"""
        for tech in scan_report.infrastructure_insights.technology_stack:
            content += f"- {tech}\n"
        
        # Add business intelligence section if available
        if scan_report.business_intelligence:
            content += f"""
## 🏢 Business Intelligence
- **Domain**: {scan_report.business_intelligence.get('business_domain', 'Unknown')}
- **Primary Purpose**: {scan_report.business_intelligence.get('application_purpose', 'Not determined')}

### Key Business Functions
"""
            for func in scan_report.business_intelligence.get('primary_business_functions', []):
                content += f"- {func}\n"
        
        # Add API documentation section if available
        if scan_report.api_documentation and scan_report.api_documentation.get('endpoints'):
            content += f"""
## 🌐 API Endpoints
| Method | Endpoint | Handler | Description |
|--------|----------|---------|-------------|
"""
            for endpoint in scan_report.api_documentation['endpoints'][:10]:  # Top 10
                method = endpoint.get('method', 'GET')
                path = endpoint.get('path', '/')
                handler = endpoint.get('handler_function', 'Unknown')
                description = endpoint.get('description', '')
                content += f"| `{method}` | `{path}` | `{handler}` | {description} |\n"
        
        content += f"""
## 🔍 Key Processes
"""
        for proc in scan_report.processes[:10]:
            content += f"- **PID {proc.pid}**: {proc.name} ({proc.service_classification})\n"
        
        content += f"""
## 📁 Application Files
"""
        file_types = {}
        for file in scan_report.application_files:
            lang = file.language
            file_types[lang] = file_types.get(lang, 0) + 1
        
        for lang, count in file_types.items():
            content += f"- **{lang}**: {count} files\n"
        
        content += f"""
## 💡 Recommendations
"""
        for rec in scan_report.infrastructure_insights.recommendations:
            content += f"- {rec}\n"
        
        content += f"""

---
*🤖 Generated by InfraDoc 2.0 - Intelligent Infrastructure Analysis*  
*"Developers just develop, we'll document"*
"""
        return content
    
    def _generate_enhanced_text_report(self, scan_report: EnhancedScanReport) -> str:
        """Generate enhanced detailed text report."""
        content = f"""
INFRADOC 2.0 - INTELLIGENT INFRASTRUCTURE ANALYSIS REPORT
=========================================================

SCAN INFORMATION
================
Host: {scan_report.host}
Scan ID: {scan_report.scan_id}
Timestamp: {scan_report.timestamp}
Duration: {scan_report.scan_duration:.2f} seconds

EXECUTIVE SUMMARY
=================
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
        
        # Add business intelligence section if available
        if scan_report.business_intelligence:
            content += f"""

BUSINESS INTELLIGENCE
=====================
Business Domain: {scan_report.business_intelligence.get('business_domain', 'Unknown')}
Application Purpose: {scan_report.business_intelligence.get('application_purpose', 'Not determined')}

Primary Business Functions:"""
            for func in scan_report.business_intelligence.get('primary_business_functions', []):
                content += f"\n- {func}"
            
            content += f"""

Critical Workflows:"""
            for workflow in scan_report.business_intelligence.get('critical_workflows', []):
                content += f"\n- {workflow}"
        
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

ENHANCED APPLICATION FILES
==========================="""
        
        for file in scan_report.application_files[:20]:
            content += f"""
Path: {file.path}
Language: {file.language}
Size: {file.size} bytes
Modified: {file.last_modified}
Business Logic: {getattr(file, 'business_logic_summary', 'Not analyzed')[:200]}
API Endpoints: {len(getattr(file, 'api_endpoints', []))}
Security Concerns: {len(getattr(file, 'security_concerns', []))}
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
Business Intelligence: {scan_report.scan_statistics.get('business_intelligence_enabled', False)}

---
Generated by InfraDoc 2.0 - Intelligent Infrastructure Analysis
Analysis completed at {datetime.now().isoformat()}
"""
        return content
    
    def _generate_api_documentation_file(self, scan_report: EnhancedScanReport) -> str:
        """Generate API documentation file."""
        api_docs = scan_report.api_documentation
        
        content = f"""# 🌐 API Documentation

## Overview
This document describes the APIs discovered in the analyzed infrastructure.

## Base URL
```
{api_docs.get('base_url', 'http://localhost')}
```

## Endpoints

"""
        
        for endpoint in api_docs.get('endpoints', []):
            method = endpoint.get('method', 'GET')
            path = endpoint.get('path', '/')
            handler = endpoint.get('handler_function', 'Unknown')
            description = endpoint.get('description', 'No description available')
            parameters = endpoint.get('parameters', [])
            
            content += f"""### `{method} {path}`
**Handler**: `{handler}`  
**Description**: {description}

"""
            if parameters:
                content += "**Parameters**:\n"
                for param in parameters:
                    content += f"- `{param}`\n"
                content += "\n"
            
            content += f"""**Example Request**:
```bash
curl -X {method} '{api_docs.get('base_url', 'http://localhost')}{path}'
```

"""
        
        # Add data models if available
        if api_docs.get('models'):
            content += "## Data Models\n\n"
            for model in api_docs['models']:
                model_name = model.get('model_name', 'Unknown')
                fields = model.get('fields', [])
                
                content += f"### {model_name}\n"
                if fields:
                    content += "| Field | Type | Description |\n"
                    content += "|-------|------|-------------|\n"
                    for field in fields:
                        field_name = field.get('name', 'unknown')
                        field_type = field.get('type', 'unknown')
                        field_desc = field.get('description', '')
                        content += f"| `{field_name}` | `{field_type}` | {field_desc} |\n"
                content += "\n"
        
        content += f"""
---
*Auto-generated API documentation by InfraDoc 2.0*
"""
        return content
    
    def _generate_business_intelligence_report(self, scan_report: EnhancedScanReport) -> str:
        """Generate business intelligence report."""
        bi = scan_report.business_intelligence
        
        content = f"""# 🏢 Business Intelligence Report

## Executive Summary
- **Business Domain**: {bi.get('business_domain', 'Unknown')}
- **Application Purpose**: {bi.get('application_purpose', 'Not determined')}

## Business Analysis

### Primary Business Functions
"""
        for func in bi.get('primary_business_functions', []):
            content += f"- {func}\n"
        
        content += f"""
### Critical Business Workflows
"""
        for workflow in bi.get('critical_workflows', []):
            content += f"- {workflow}\n"
        
        # Add data flows if available
        if bi.get('data_flows'):
            content += f"""
## Data Flow Analysis
| From | To | Data Type | Description |
|------|-----|-----------|-------------|
"""
            for flow in bi['data_flows']:
                from_src = flow.get('from', 'Unknown')
                to_dest = flow.get('to', 'Unknown')
                data_type = flow.get('data_type', 'Unknown')
                description = flow.get('description', '')
                content += f"| {from_src} | {to_dest} | {data_type} | {description} |\n"
        
        # Add integration architecture if available
        if bi.get('integration_architecture'):
            ia = bi['integration_architecture']
            content += f"""
## Integration Architecture

### External APIs
"""
            for api in ia.get('external_apis', []):
                content += f"- {api}\n"
            
            content += f"""
### Databases
"""
            for db in ia.get('databases', []):
                content += f"- {db}\n"
            
            content += f"""
### Message Queues
"""
            for queue in ia.get('message_queues', []):
                content += f"- {queue}\n"
        
        # Add scaling characteristics if available
        if bi.get('scaling_characteristics'):
            sc = bi['scaling_characteristics']
            content += f"""
## Scaling Characteristics

### Potential Bottlenecks
"""
            for bottleneck in sc.get('bottlenecks', []):
                content += f"- {bottleneck}\n"
            
            content += f"""
### Scaling Strategy
{sc.get('scaling_strategy', 'Not determined')}

### Resource Intensive Operations
"""
            for operation in sc.get('resource_intensive_operations', []):
                content += f"- {operation}\n"
        
        content += f"""
---
*Business intelligence extracted by InfraDoc 2.0*
"""
        return content
    
    def _generate_intelligent_documentation(self, scan_report: EnhancedScanReport, output_dir: Path) -> bool:
        """Generate intelligent comprehensive documentation."""
        logger.info("[ANALYZER] Generating intelligent comprehensive documentation")
        
        try:
            # Import intelligent documentation generator
            from infradoc_docs import IntelligentDocumentationGenerator
            
            doc_generator = IntelligentDocumentationGenerator(scan_report, str(output_dir))
            success = doc_generator.generate_intelligent_documentation()
            
            if success:
                logger.info("[ANALYZER] Intelligent documentation generated successfully")
            else:
                logger.warning("[ANALYZER] Intelligent documentation generation had issues")
            
            return success
            
        except ImportError:
            logger.warning("[ANALYZER] Intelligent documentation generator not available, using basic generator")
            try:
                from infradoc_docs import DocumentationGenerator
                doc_generator = DocumentationGenerator(scan_report, str(output_dir))
                return doc_generator.generate_all_documentation()
            except ImportError:
                logger.warning("[ANALYZER] No documentation generator available")
                return False
        except Exception as e:
            logger.error(f"[ANALYZER] Intelligent documentation generation failed: {e}")
            return False
    
    def _log_analysis_summary(self, result: AnalysisResult):
        """Log intelligent analysis completion summary."""
        if result.success:
            logger.info("[ANALYZER] INTELLIGENT ANALYSIS COMPLETED SUCCESSFULLY")
            logger.info(f"[ANALYZER] Processes: {len(result.scan_report.processes)}")
            logger.info(f"[ANALYZER] Files: {len(result.scan_report.application_files)}")
            logger.info(f"[ANALYZER] Architecture: {result.scan_report.infrastructure_insights.architecture_pattern}")
            logger.info(f"[ANALYZER] Artifacts: {len(result.artifacts_generated)}")
            logger.info(f"[ANALYZER] Documentation: {'YES' if result.documentation_generated else 'NO'}")
            
            if result.scan_report.business_intelligence:
                domain = result.scan_report.business_intelligence.get('business_domain', 'Unknown')
                logger.info(f"[ANALYZER] Business Domain: {domain}")
            
            if result.scan_report.api_documentation:
                endpoint_count = len(result.scan_report.api_documentation.get('endpoints', []))
                logger.info(f"[ANALYZER] API Endpoints: {endpoint_count}")
            
            logger.info(f"[ANALYZER] Output: {result.output_directory}")
        else:
            logger.error(f"[ANALYZER] INTELLIGENT ANALYSIS FAILED: {result.error_message}")

# ================================================================================
# COMPATIBILITY LAYER FOR EXISTING CODE
# ================================================================================

# Keep original class for backward compatibility
InfrastructureAnalyzer = IntelligentInfrastructureAnalyzer

# ================================================================================
# CONVENIENCE FUNCTIONS
# ================================================================================

def quick_analysis(host: str, username: str = "ubuntu", key_file: str = None, 
                  password: str = None) -> AnalysisResult:
    """
    Perform quick infrastructure analysis with minimal configuration.
    """
    connection_config = ConnectionConfig(
        host=host,
        username=username,
        key_file=key_file,
        password=password
    )
    
    analysis_config = AnalysisConfig(
        scan_depth="quick",
        enable_ai=True,
        max_llm_calls=5,
        include_documentation=False
    )
    
    llm_providers = []
    if os.getenv("OPENAI_API_KEY"):
        llm_providers.append({"provider": "openai", "model": "gpt-4o"})
    if os.getenv("ANTHROPIC_API_KEY"):
        llm_providers.append({"provider": "claude", "model": "claude-3-5-sonnet-20241022"})
    
    analyzer = IntelligentInfrastructureAnalyzer(llm_providers)
    try:
        return analyzer.analyze_infrastructure(connection_config, analysis_config)
    finally:
        analyzer._cleanup_resources()

def deep_analysis(host: str, username: str = "ubuntu", key_file: str = None, 
                 password: str = None) -> AnalysisResult:
    """
    Perform comprehensive deep infrastructure analysis.
    """
    connection_config = ConnectionConfig(
        host=host,
        username=username,
        key_file=key_file,
        password=password
    )
    
    analysis_config = AnalysisConfig(
        scan_depth="deep",
        enable_ai=True,
        max_llm_calls=25,
        include_documentation=True,
        include_security=True
    )
    
    llm_providers = []
    if os.getenv("OPENAI_API_KEY"):
        llm_providers.append({"provider": "openai", "model": "gpt-4o"})
    if os.getenv("ANTHROPIC_API_KEY"):
        llm_providers.append({"provider": "claude", "model": "claude-3-5-sonnet-20241022"})
    if os.getenv("GROK_API_KEY"):
        llm_providers.append({"provider": "grok", "model": "grok-3"})
    
    analyzer = IntelligentInfrastructureAnalyzer(llm_providers)
    try:
        return analyzer.analyze_infrastructure(connection_config, analysis_config)
    finally:
        analyzer._cleanup_resources()

def intelligent_analysis(host: str, username: str = "ubuntu", key_file: str = None, 
                        password: str = None) -> AnalysisResult:
    """
    Perform intelligent infrastructure analysis with business understanding.
    """
    connection_config = ConnectionConfig(
        host=host,
        username=username,
        key_file=key_file,
        password=password
    )
    
    analysis_config = AnalysisConfig(
        scan_depth="intelligent",
        enable_ai=True,
        max_llm_calls=35,
        include_documentation=True,
        include_security=True,
        enable_business_intelligence=True,
        enable_enhanced_analysis=True
    )
    
    llm_providers = []
    if os.getenv("OPENAI_API_KEY"):
        llm_providers.append({"provider": "openai", "model": "gpt-4o"})
    if os.getenv("ANTHROPIC_API_KEY"):
        llm_providers.append({"provider": "claude", "model": "claude-3-5-sonnet-20241022"})
    if os.getenv("GROK_API_KEY"):
        llm_providers.append({"provider": "grok", "model": "grok-3"})
    
    analyzer = IntelligentInfrastructureAnalyzer(llm_providers)
    try:
        return analyzer.analyze_infrastructure(connection_config, analysis_config)
    finally:
        analyzer._cleanup_resources()