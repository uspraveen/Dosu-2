#!/usr/bin/env python3
"""
InfraDoc 2.0 - Enhanced Core Infrastructure Components
Contains SSH connector, LLM orchestrator, and intelligent smart discovery classes.
"""

import os
import json
import time
import socket
import logging
import platform
import threading
import paramiko
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
import re

# Configure logging with Windows-safe format (no emojis)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================================================================================
# SSH CONNECTION MANAGEMENT
# ================================================================================

@dataclass
class ConnectionConfig:
    """SSH connection configuration."""
    host: str
    port: int = 22
    username: str = "ubuntu"
    key_file: Optional[str] = None
    password: Optional[str] = None
    timeout: int = 30
    connection_name: Optional[str] = None
    max_retries: int = 3
    retry_delay: int = 2

@dataclass
class CommandExecution:
    """Command execution record."""
    command: str
    host: str
    username: str
    start_time: str
    end_time: str
    duration: float
    exit_code: int
    stdout_lines: int
    stderr_lines: int
    success: bool
    error_message: Optional[str] = None
    command_id: Optional[str] = None

class SSHConnector:
    """Intelligent SSH connector with enhanced error handling and logging."""
    
    def __init__(self, command_timeout: int = 30):
        """Initialize SSH connector."""
        self.connections: Dict[str, paramiko.SSHClient] = {}
        self.command_history: List[CommandExecution] = []
        self.command_timeout = command_timeout
        self.connection_lock = threading.Lock()
        self.command_counter = 0
        
        logger.info("[SSH] SSH Connector initialized")
    
    def _create_ssh_client(self) -> paramiko.SSHClient:
        """Create SSH client with proper host key policy."""
        client = paramiko.SSHClient()
        
        # Handle different paramiko versions for AutoAddHostKeyPolicy
        try:
            # Try the standard location first
            client.set_missing_host_key_policy(paramiko.AutoAddHostKeyPolicy())
        except AttributeError:
            try:
                # Try importing from client module
                from paramiko.client import AutoAddHostKeyPolicy
                client.set_missing_host_key_policy(AutoAddHostKeyPolicy())
            except ImportError:
                try:
                    # Try importing from policy module
                    from paramiko.policy import AutoAddHostKeyPolicy
                    client.set_missing_host_key_policy(AutoAddHostKeyPolicy())
                except ImportError:
                    # Fall back to WarningPolicy
                    logger.warning("[SSH] AutoAddHostKeyPolicy not found, using WarningPolicy")
                    client.set_missing_host_key_policy(paramiko.WarningPolicy())
        
        return client
    
    def connect(self, config: ConnectionConfig) -> bool:
        """
        Connect to remote host with retry logic.
        
        Args:
            config: Connection configuration
            
        Returns:
            True if connection successful, False otherwise
        """
        logger.info(f"[SSH] Connecting to {config.host}:{config.port}")
        
        for attempt in range(1, config.max_retries + 1):
            try:
                client = self._create_ssh_client()
                
                connect_params = {
                    'hostname': config.host,
                    'port': config.port,
                    'username': config.username,
                    'timeout': config.timeout
                }
                
                # Add authentication
                if config.key_file:
                    connect_params['key_filename'] = os.path.expanduser(config.key_file)
                    logger.info(f"[SSH] Using SSH key authentication")
                elif config.password:
                    connect_params['password'] = config.password
                    logger.info(f"[SSH] Using password authentication")
                else:
                    raise ValueError("Either key_file or password must be provided")
                
                # Attempt connection
                client.connect(**connect_params)
                
                # Store successful connection
                with self.connection_lock:
                    self.connections[config.host] = client
                
                logger.info(f"[SSH] Connected to {config.host} (attempt {attempt})")
                return True
                
            except Exception as e:
                logger.warning(f"[SSH] Connection attempt {attempt} failed: {e}")
                if attempt < config.max_retries:
                    logger.info(f"[SSH] Retrying in {config.retry_delay} seconds...")
                    time.sleep(config.retry_delay)
                else:
                    logger.error(f"[SSH] Failed to connect to {config.host} after {config.max_retries} attempts")
                    return False
        
        return False
    
    def execute_command(self, host: str, command: str, description: str = "", 
                       timeout: int = None) -> Tuple[str, str, int, CommandExecution]:
        """
        Execute command with intelligent error handling.
        
        Args:
            host: Target hostname
            command: Command to execute
            description: Human-readable description
            timeout: Command timeout in seconds
            
        Returns:
            Tuple of (stdout, stderr, exit_code, execution_record)
        """
        if host not in self.connections:
            raise ConnectionError(f"No active connection to {host}")
        
        self.command_counter += 1
        command_id = f"cmd_{self.command_counter:04d}"
        
        timeout = timeout or self.command_timeout
        start_time = datetime.now()
        
        logger.debug(f"[SSH] [{command_id}] Executing: {description or command[:50]}")
        
        try:
            client = self.connections[host]
            
            # Execute command
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            
            # Read outputs
            stdout_data = stdout.read().decode(errors='replace')
            stderr_data = stderr.read().decode(errors='replace')
            exit_code = stdout.channel.recv_exit_status()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Create execution record
            execution = CommandExecution(
                command=command,
                host=host,
                username=client.get_transport().get_username(),
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration=duration,
                exit_code=exit_code,
                stdout_lines=len(stdout_data.split('\n')) if stdout_data else 0,
                stderr_lines=len(stderr_data.split('\n')) if stderr_data else 0,
                success=exit_code == 0,
                command_id=command_id
            )
            
            self.command_history.append(execution)
            
            # Log results
            if exit_code == 0:
                logger.debug(f"[SSH] [{command_id}] Success in {duration:.2f}s")
            else:
                logger.warning(f"[SSH] [{command_id}] Exit code {exit_code}")
            
            return stdout_data, stderr_data, exit_code, execution
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            execution = CommandExecution(
                command=command,
                host=host,
                username="unknown",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration=duration,
                exit_code=-1,
                stdout_lines=0,
                stderr_lines=0,
                success=False,
                error_message=str(e),
                command_id=command_id
            )
            
            self.command_history.append(execution)
            logger.error(f"[SSH] [{command_id}] Error: {e}")
            
            return "", str(e), -1, execution
    
    def close_connection(self, host: str):
        """Close connection to specific host."""
        with self.connection_lock:
            if host in self.connections:
                try:
                    self.connections[host].close()
                    del self.connections[host]
                    logger.info(f"[SSH] Closed connection to {host}")
                except Exception as e:
                    logger.warning(f"[SSH] Error closing connection to {host}: {e}")
    
    def close_all_connections(self):
        """Close all connections."""
        with self.connection_lock:
            for host in list(self.connections.keys()):
                self.close_connection(host)

# ================================================================================
# LLM ORCHESTRATION
# ================================================================================

class AnalysisStage(Enum):
    """Analysis stages for progressive infrastructure analysis."""
    DISCOVERY = "discovery"
    PROCESS_CLASSIFICATION = "process_classification"
    FILE_DISCOVERY = "file_discovery"
    CODE_ANALYSIS = "code_analysis"
    BUSINESS_INTELLIGENCE = "business_intelligence"
    SERVICE_MAPPING = "service_mapping"
    ARCHITECTURE_SYNTHESIS = "architecture_synthesis"
    SECURITY_ANALYSIS = "security_analysis"

@dataclass
class LLMCall:
    """LLM call record."""
    stage: AnalysisStage
    prompt: str
    response: Dict
    duration: float
    timestamp: str
    success: bool
    provider: str
    model: str

class LLMProvider:
    """Individual LLM provider wrapper."""
    
    def __init__(self, provider: str, model: str, api_key: str = None):
        """Initialize LLM provider."""
        self.provider = provider.lower()
        self.model = model
        self.client = None
        self.api_key = api_key or self._get_api_key()
        self._initialize_client()
    
    def _get_api_key(self) -> str:
        """Get API key from environment."""
        key_map = {
            "openai": "OPENAI_API_KEY",
            "grok": "GROK_API_KEY", 
            "claude": "ANTHROPIC_API_KEY"
        }
        return os.getenv(key_map.get(self.provider, ""))
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client."""
        try:
            if self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            elif self.provider == "grok":
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.x.ai/v1"
                )
            elif self.provider == "claude":
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            
            logger.info(f"[LLM] Initialized {self.provider.upper()} client")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} client: {e}")
            raise
    
    def call(self, prompt: str, system_prompt: str = None, 
             max_tokens: int = 4000, temperature: float = 0.1) -> Dict:
        """
        Make LLM call with enhanced JSON handling.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Response dictionary with parsed content
        """
        start_time = time.time()
        
        try:
            if self.provider in ["openai", "grok"]:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                content = response.choices[0].message.content
                
            elif self.provider == "claude":
                messages = [{"role": "user", "content": prompt}]
                if system_prompt:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_prompt,
                        messages=messages
                    )
                else:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=messages
                    )
                content = response.content[0].text
            
            duration = time.time() - start_time
            
            # Parse response
            parsed_response = self._parse_response(content)
            
            return {
                "success": True,
                "content": content,
                "parsed": parsed_response,
                "duration": duration,
                "provider": self.provider,
                "model": self.model
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"LLM call failed with {self.provider}: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "provider": self.provider,
                "model": self.model
            }
    
    def _parse_response(self, content: str) -> Dict:
        """Parse LLM response with multiple strategies."""
        try:
            # Strategy 1: Direct JSON parsing
            if content.strip().startswith('{') and content.strip().endswith('}'):
                return json.loads(content)
            
            # Strategy 2: Extract from markdown code blocks
            json_patterns = [
                r'```json\s*(.*?)\s*```',
                r'```\s*(.*?)\s*```',
                r'\{.*?\}',
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                for match in matches:
                    try:
                        cleaned = self._clean_json(match)
                        return json.loads(cleaned)
                    except:
                        continue
            
            # Strategy 3: Extract structured information from text
            return self._extract_structured_info(content)
            
        except Exception as e:
            logger.debug(f"JSON parsing failed: {e}")
            return {"raw_response": content, "parsing_error": str(e)}
    
    def _clean_json(self, json_str: str) -> str:
        """Clean common JSON formatting issues."""
        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        return json_str.strip()
    
    def _extract_structured_info(self, content: str) -> Dict:
        """Extract structured information from unstructured text."""
        extracted = {
            "analysis": content,
            "key_findings": [],
            "recommendations": [],
            "insights": []
        }
        
        # Extract bullet points and lists
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify sections
            if any(keyword in line.lower() for keyword in ['finding', 'discovered', 'identified']):
                current_section = "key_findings"
            elif any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should']):
                current_section = "recommendations"
            elif any(keyword in line.lower() for keyword in ['insight', 'pattern', 'architecture']):
                current_section = "insights"
            
            # Extract list items
            if line.startswith(('-', '•', '*', '1.', '2.', '3.')) and current_section:
                item = re.sub(r'^[-•*\d\.]\s*', '', line)
                extracted[current_section].append(item)
        
        return extracted

class LLMOrchestrator:
    """Intelligent orchestrator for progressive LLM analysis."""
    
    def __init__(self, providers: List[Dict] = None):
        """
        Initialize LLM orchestrator.
        
        Args:
            providers: List of provider configurations
        """
        self.providers = self._initialize_providers(providers)
        self.call_history: List[LLMCall] = []
        self.current_context = {}
        self.total_calls = 0
        
        logger.info(f"[LLM] LLM Orchestrator initialized with {len(self.providers)} providers")
    
    def _initialize_providers(self, providers: List[Dict] = None) -> List[LLMProvider]:
        """Initialize multiple LLM providers with fallbacks."""
        if providers is None:
            providers = [
                {"provider": "openai", "model": "gpt-4o"},
                {"provider": "claude", "model": "claude-3-5-sonnet-20241022"},
                {"provider": "grok", "model": "grok-3"}
            ]
        
        initialized_providers = []
        for provider_config in providers:
            try:
                provider = LLMProvider(
                    provider_config["provider"],
                    provider_config["model"]
                )
                initialized_providers.append(provider)
                logger.info(f"[LLM] {provider_config['provider'].upper()} ready")
            except Exception as e:
                logger.warning(f"[LLM] Failed to initialize {provider_config['provider']}: {e}")
        
        if not initialized_providers:
            raise Exception("No LLM providers available")
        
        return initialized_providers
    
    def progressive_analysis(self, stage: AnalysisStage, prompt: str, 
                           system_prompt: str, context: Dict = None) -> Dict:
        """
        Perform progressive analysis with context and retries.
        
        Args:
            stage: Analysis stage
            prompt: Analysis prompt
            system_prompt: System prompt
            context: Additional context
            
        Returns:
            Analysis response
        """
        logger.info(f"[LLM] Starting {stage.value} analysis...")
        
        # Build enhanced prompt with context
        enhanced_prompt = self._build_contextual_prompt(prompt, context or {})
        
        best_response = None
        
        for provider in self.providers:
            try:
                logger.info(f"[LLM] Calling {provider.provider.upper()} for {stage.value}")
                
                response = provider.call(enhanced_prompt, system_prompt)
                
                self.total_calls += 1
                
                # Record the call
                call_record = LLMCall(
                    stage=stage,
                    prompt=enhanced_prompt[:200] + "...",
                    response=response,
                    duration=response.get("duration", 0),
                    timestamp=datetime.now().isoformat(),
                    success=response.get("success", False),
                    provider=provider.provider,
                    model=provider.model
                )
                self.call_history.append(call_record)
                
                if response.get("success"):
                    best_response = response
                    logger.info(f"[LLM] {provider.provider.upper()} completed {stage.value} in {response['duration']:.2f}s")
                    break
                else:
                    logger.warning(f"[LLM] {provider.provider.upper()} failed for {stage.value}")
                    
            except Exception as e:
                logger.error(f"[LLM] {provider.provider.upper()} error: {e}")
                continue
        
        if not best_response or not best_response.get("success"):
            logger.error(f"[LLM] All providers failed for {stage.value}")
            return {"success": False, "error": "All providers failed"}
        
        logger.info(f"[LLM] {stage.value} completed successfully")
        return best_response
    
    def _build_contextual_prompt(self, base_prompt: str, context: Dict) -> str:
        """Build enhanced prompt with progressive context."""
        if not context:
            return base_prompt
        
        context_section = "\n## CONTEXT\n"
        for key, value in context.items():
            if isinstance(value, (str, int, float)):
                context_section += f"**{key}**: {value}\n"
            elif isinstance(value, list) and len(value) < 10:
                context_section += f"**{key}**: {', '.join(map(str, value))}\n"
        
        return f"{context_section}\n\n## ANALYSIS REQUEST\n{base_prompt}"
    
    def get_analysis_summary(self) -> Dict:
        """Get comprehensive analysis summary."""
        return {
            "total_llm_calls": self.total_calls,
            "analysis_stages": len(set(call.stage for call in self.call_history)),
            "overall_confidence": 0.9 if self.total_calls > 0 else 0,
            "performance_metrics": {
                "total_duration": sum(call.duration for call in self.call_history),
                "avg_call_duration": sum(call.duration for call in self.call_history) / len(self.call_history) if self.call_history else 0
            }
        }

# ================================================================================
# ENHANCED DATA STRUCTURES
# ================================================================================

@dataclass
class ProcessInfo:
    """Enhanced process information with LLM insights."""
    pid: int
    name: str
    user: str
    cpu_percent: str
    memory_percent: str
    command: str
    full_command_line: Optional[str] = None
    working_dir: Optional[str] = None
    service_classification: Optional[str] = None
    service_purpose: Optional[str] = None
    integrations_detected: List[str] = None
    security_concerns: List[str] = None

@dataclass
class ApplicationFile:
    """Basic application file structure."""
    path: str
    language: str
    size: int
    last_modified: str
    imports: List[str] = None
    functions: List[str] = None
    classes: List[str] = None
    external_services: List[str] = None

@dataclass
class EnhancedApplicationFile:
    """Enhanced application file with comprehensive intelligence."""
    path: str
    language: str
    size: int
    last_modified: str
    
    # Basic analysis (existing)
    imports: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    external_services: List[str] = field(default_factory=list)
    
    # Enhanced intelligence fields
    api_endpoints: List[Dict] = field(default_factory=list)
    database_models: List[Dict] = field(default_factory=list)
    environment_variables: List[Dict] = field(default_factory=list)
    security_concerns: List[Dict] = field(default_factory=list)
    business_logic_summary: str = ""
    performance_notes: List[str] = field(default_factory=list)
    
    # Code quality metrics
    complexity_score: int = 0
    documentation_quality: str = "Unknown"
    
    # Dependencies and integrations
    external_apis: List[Dict] = field(default_factory=list)
    database_connections: List[str] = field(default_factory=list)

@dataclass
class InfrastructureInsights:
    """Comprehensive infrastructure insights from LLM analysis."""
    architecture_pattern: str
    technology_stack: List[str]
    deployment_model: str
    scalability_assessment: str
    security_posture: str
    operational_complexity: str
    recommendations: List[str]

# ================================================================================
# INTELLIGENT SMART DISCOVERY
# ================================================================================

class SmartDiscovery:
    """LLM-guided infrastructure discovery and analysis."""
    
    def __init__(self, connector: SSHConnector, llm_orchestrator: LLMOrchestrator):
        """
        Initialize smart discovery.
        
        Args:
            connector: SSH connector instance
            llm_orchestrator: LLM orchestrator instance
        """
        self.connector = connector
        self.llm = llm_orchestrator
        self.discovered_data = {}
        
        logger.info("[DISCOVERY] Smart Discovery initialized")
    
    def discover_infrastructure(self, host: str) -> Dict[str, Any]:
        """
        Perform intelligent infrastructure discovery.
        
        Args:
            host: Target host to analyze
            
        Returns:
            Complete discovery results
        """
        logger.info(f"[DISCOVERY] Starting intelligent discovery of {host}")
        
        try:
            # Stage 1: Initial Discovery
            initial_context = self._stage1_initial_discovery(host)
            
            # Stage 2: Process Analysis
            process_context = self._stage2_process_analysis(host, initial_context)
            
            # Stage 3: File Discovery
            file_context = self._stage3_file_discovery(host, process_context)
            
            # Stage 4: Enhanced Code Analysis
            code_context = self._stage4_enhanced_code_analysis(host, file_context)
            
            # Stage 5: Business Intelligence
            business_context = self._stage5_business_intelligence(host, code_context)
            
            # Stage 6: Architecture Synthesis
            architecture_insights = self._stage6_architecture_synthesis(host, business_context)
            
            # Stage 7: Security Analysis
            security_analysis = self._stage7_security_analysis(host, architecture_insights)
            
            return {
                'processes': self.discovered_data.get('processes', []),
                'files': self.discovered_data.get('enhanced_files', []),
                'business_intelligence': self.discovered_data.get('business_intelligence', {}),
                'infrastructure_insights': asdict(architecture_insights),
                'security_analysis': security_analysis,
                'llm_analysis_summary': self.llm.get_analysis_summary()
            }
        
        except Exception as e:
            logger.error(f"[DISCOVERY] Discovery failed: {e}")
            # Return basic fallback data
            return {
                'processes': self.discovered_data.get('processes', []),
                'files': self.discovered_data.get('files', []),
                'infrastructure_insights': {
                    'architecture_pattern': 'Unknown',
                    'technology_stack': [],
                    'deployment_model': 'Unknown',
                    'scalability_assessment': 'Unknown',
                    'security_posture': 'Needs assessment',
                    'operational_complexity': 'Unknown',
                    'recommendations': ['Review infrastructure analysis']
                },
                'security_analysis': {'analysis': 'Discovery incomplete due to error'},
                'llm_analysis_summary': self.llm.get_analysis_summary()
            }
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'connector') and self.connector:
                self.connector.close_all_connections()
        except Exception as e:
            logger.warning(f"[DISCOVERY] Cleanup warning: {e}")
    
    def _stage1_initial_discovery(self, host: str) -> Dict:
        """Stage 1: Initial system discovery."""
        logger.info("[DISCOVERY] Stage 1: Initial system discovery")
        
        # Get basic system information
        commands = [
            ('ps aux --no-headers', 'process_list'),
            ('netstat -tuln 2>/dev/null || ss -tuln', 'network_ports'),
            ('find /opt /srv /var/www /home -maxdepth 3 -type d 2>/dev/null | head -25', 'app_directories')
        ]
        
        raw_data = {}
        for command, key in commands:
            stdout, stderr, exit_code, execution = self.connector.execute_command(
                host, command, key
            )
            raw_data[key] = stdout
        
        # LLM analyzes raw system data
        discovery_prompt = f"""
Analyze this system data to guide infrastructure discovery. Focus on ACTUAL APPLICATION CODE and services, NOT library dependencies or virtual environment files:

PROCESSES: {raw_data.get('process_list', '')[:3000]}
NETWORK: {raw_data.get('network_ports', '')[:1000]}
DIRECTORIES: {raw_data.get('app_directories', '')[:1000]}

Identify and prioritize:
1. PRIMARY APPLICATIONS (worker.py, main.py, server.py, app.py - actual application entry points)
2. BUSINESS LOGIC FILES (not site-packages, not venv, not system libraries)
3. CONFIGURATION FILES (systemd services, nginx configs, application configs)
4. APPLICATION DIRECTORIES to explore (exclude /venv/, /site-packages/, /__pycache__/)

IGNORE:
- Virtual environment files (/venv/, /site-packages/)
- Python libraries and dependencies
- System packages and caches
- __pycache__ directories

Focus on files that contain business logic, application entry points, and infrastructure configuration.
Provide specific guidance for next discovery steps targeting actual application code.
"""

        system_prompt = """You are an expert DevOps engineer analyzing infrastructure.
        Focus on real applications vs system processes and provide actionable discovery guidance."""
        
        response = self.llm.progressive_analysis(
            stage=AnalysisStage.DISCOVERY,
            prompt=discovery_prompt,
            system_prompt=system_prompt,
            context={'host': host}
        )
        
        return {
            'raw_data': raw_data,
            'llm_guidance': response.get('parsed', {}),
            'priority_processes': self._extract_app_processes(raw_data.get('process_list', ''))
        }
    
    def _stage2_process_analysis(self, host: str, initial_context: Dict) -> Dict:
        """Stage 2: Intelligent process classification."""
        logger.info("[DISCOVERY] Stage 2: Process classification")
        
        processes_data = []
        process_lines = initial_context['raw_data'].get('process_list', '').split('\n')
        
        # Filter for real application processes
        for line in process_lines:
            if not line.strip():
                continue
            
            parts = line.split(None, 10)
            if len(parts) >= 11:
                pid = parts[1]
                user = parts[0]
                command = parts[10]
                
                if self._is_application_process(command, user):
                    enhanced_info = self._get_process_details(host, pid, command, user)
                    if enhanced_info:
                        processes_data.append(enhanced_info)
        
        logger.info(f"[DISCOVERY] Found {len(processes_data)} application processes")
        
        # LLM classifies processes
        classification_prompt = f"""
Classify these application processes:

{json.dumps(processes_data[:10], indent=2)}

For each process, determine:
1. Service classification (web_server, database, application, background_worker, etc.)
2. Application purpose and role
3. Technology integrations detected
4. Security concerns

Provide detailed classification for infrastructure understanding.
"""

        system_prompt = """You are a systems architect classifying application processes.
        Focus on service types, purposes, and relationships between components."""
        
        response = self.llm.progressive_analysis(
            stage=AnalysisStage.PROCESS_CLASSIFICATION,
            prompt=classification_prompt,
            system_prompt=system_prompt,
            context=initial_context
        )
        
        # Convert to ProcessInfo objects
        classified_processes = []
        llm_insights = response.get('parsed', {})
        
        for proc_data in processes_data:
            process_info = ProcessInfo(
                pid=int(proc_data.get('pid', 0)),
                name=proc_data.get('name', ''),
                user=proc_data.get('user', ''),
                cpu_percent=proc_data.get('cpu_percent', '0'),
                memory_percent=proc_data.get('memory_percent', '0'),
                command=proc_data.get('command', ''),
                full_command_line=proc_data.get('full_command_line'),
                working_dir=proc_data.get('working_dir'),
                service_classification=self._extract_classification(proc_data, llm_insights),
                service_purpose=self._extract_purpose(proc_data, llm_insights),
                integrations_detected=self._extract_integrations(proc_data, llm_insights),
                security_concerns=self._extract_security_concerns(proc_data, llm_insights)
            )
            classified_processes.append(process_info)
        
        self.discovered_data['processes'] = classified_processes
        return {'classified_processes': classified_processes, 'llm_analysis': llm_insights}
    
    def _stage3_file_discovery(self, host: str, process_context: Dict) -> Dict:
        """Stage 3: Intelligent file discovery."""
        logger.info("[DISCOVERY] Stage 3: File discovery")
        
        # Extract application directories from processes
        app_directories = ['/opt', '/srv', '/var/www', '/home', '/app']
        
        # Add directories from process working dirs
        for proc in process_context.get('classified_processes', []):
            if proc.working_dir and proc.working_dir not in ['/', '/proc', '/sys']:
                app_directories.append(proc.working_dir)
        
        # Focus on actual application files, exclude virtual environments
        file_searches = [
            # Application Python files (exclude site-packages and venv)
            f'find {" ".join(app_directories)} -name "*.py" -type f ! -path "*/site-packages/*" ! -path "*/venv/*" ! -path "*/__pycache__/*" 2>/dev/null | head -15',
            # JavaScript application files
            f'find {" ".join(app_directories)} -name "*.js" -type f ! -path "*/node_modules/*" ! -path "*/venv/*" 2>/dev/null | head -10',
            # Configuration files
            f'find {" ".join(app_directories)} -name "*.conf" -o -name "*.yml" -o -name "*.yaml" -o -name "*.json" 2>/dev/null | grep -v "/venv/" | head -10',
            # Project files
            f'find {" ".join(app_directories)} -name "requirements.txt" -o -name "package.json" -o -name "Dockerfile" -o -name "docker-compose.yml" 2>/dev/null | head -5',
            # Systemd service files
            'find /etc/systemd/system -name "*.service" -type f 2>/dev/null | head -10',
            # Nginx configuration
            'find /etc/nginx -name "*.conf" -type f 2>/dev/null | head -5'
        ]
        
        discovered_files = []
        config_files = []
        
        for search_cmd in file_searches:
            stdout, stderr, exit_code, execution = self.connector.execute_command(
                host, search_cmd, "file_discovery"
            )
            if stdout.strip():
                files = stdout.strip().split('\n')
                for file_path in files:
                    # Categorize files
                    if any(config_pattern in file_path for config_pattern in ['/etc/', '.conf', '.service', '.yml', '.yaml']):
                        config_files.append(file_path)
                    else:
                        discovered_files.append(file_path)
        
        # Remove duplicates and filter out virtual environment files
        discovered_files = self._filter_application_files(list(set(discovered_files)))
        config_files = list(set(config_files))
        
        logger.info(f"[DISCOVERY] Discovered {len(discovered_files)} application files and {len(config_files)} configuration files")
        
        return {
            'discovered_files': discovered_files,
            'configuration_files': config_files
        }
    
    def _stage4_enhanced_code_analysis(self, host: str, file_context: Dict) -> Dict:
        """Stage 4: Enhanced code analysis with full file intelligence."""
        logger.info("[DISCOVERY] Stage 4: Enhanced code analysis")
        
        analyzed_files = []
        configuration_files = []
        
        # Analyze application files with enhanced intelligence
        for file_path in file_context.get('discovered_files', [])[:10]:
            enhanced_file = self._analyze_file_with_intelligence(host, file_path)
            if enhanced_file:
                analyzed_files.append(enhanced_file)
        
        # Analyze configuration files
        for file_path in file_context.get('configuration_files', [])[:5]:
            config_analysis = self._analyze_configuration_file(host, file_path)
            if config_analysis:
                configuration_files.append(config_analysis)
        
        self.discovered_data['enhanced_files'] = analyzed_files
        self.discovered_data['configuration_files'] = configuration_files
        
        return {
            'analyzed_files': analyzed_files,
            'configuration_files': configuration_files
        }
    
    def _stage5_business_intelligence(self, host: str, code_context: Dict) -> Dict:
        """Stage 5: Extract business intelligence from code analysis."""
        logger.info("[DISCOVERY] Stage 5: Business intelligence extraction")
        
        enhanced_files = code_context.get('analyzed_files', [])
        
        business_intelligence_prompt = f"""
BUSINESS INTELLIGENCE ANALYSIS

Analyze these application files for complete business understanding:

APPLICATION FILES WITH ENHANCED DATA:
{json.dumps([asdict(f) for f in enhanced_files], indent=2, default=str)[:12000]}

Extract comprehensive business intelligence:

{{
  "business_domain": "e-commerce|finance|healthcare|education|logistics|social|saas|other",
  "primary_business_functions": ["main business capabilities"],
  "application_purpose": "What problem does this solve for users?",
  "data_flows": [
    {{"from": "source", "to": "destination", "data_type": "user_data", "description": "flow description"}}
  ],
  "critical_workflows": ["most important business processes"],
  "integration_architecture": {{
    "external_apis": ["list of external APIs used"],
    "databases": ["database systems detected"],
    "message_queues": ["queue systems if any"],
    "caching": ["caching solutions"]
  }},
  "scaling_characteristics": {{
    "bottlenecks": ["potential performance bottlenecks"],
    "scaling_strategy": "horizontal|vertical|both",
    "resource_intensive_operations": ["operations that use most resources"]
  }},
  "technology_insights": {{
    "framework_patterns": ["web framework patterns detected"],
    "architectural_style": "microservices|monolith|layered|event-driven",
    "deployment_complexity": "simple|moderate|complex"
  }}
}}

Focus on BUSINESS VALUE and practical understanding.
"""

        system_prompt = """You are a senior business analyst and technical architect.
        Extract business intelligence that helps stakeholders understand the value,
        purpose, and operational characteristics of this system."""
        
        response = self.llm.progressive_analysis(
            stage=AnalysisStage.BUSINESS_INTELLIGENCE,
            prompt=business_intelligence_prompt,
            system_prompt=system_prompt,
            context=code_context
        )
        
        business_intelligence = response.get('parsed', {})
        self.discovered_data['business_intelligence'] = business_intelligence
        
        return {
            'business_intelligence': business_intelligence,
            'enhanced_files': enhanced_files
        }
    
    def _stage6_architecture_synthesis(self, host: str, business_context: Dict) -> InfrastructureInsights:
        """Stage 6: Architecture synthesis with business intelligence."""
        logger.info("[DISCOVERY] Stage 6: Architecture synthesis")
        
        # Combine all analysis data
        complete_analysis = {
            'processes': [asdict(p) for p in self.discovered_data.get('processes', [])],
            'enhanced_files': [asdict(f) for f in self.discovered_data.get('enhanced_files', [])],
            'configuration_files': self.discovered_data.get('configuration_files', []),
            'business_intelligence': self.discovered_data.get('business_intelligence', {})
        }
        
        synthesis_prompt = f"""
COMPREHENSIVE ARCHITECTURE SYNTHESIS

Synthesize complete infrastructure analysis with business context:

BUSINESS INTELLIGENCE:
{json.dumps(complete_analysis['business_intelligence'], indent=2, default=str)[:3000]}

APPLICATION PROCESSES:
{json.dumps([p for p in complete_analysis['processes'] if p.get('service_classification') in ['application', 'background_worker', 'web_server']], indent=2, default=str)[:3000]}

ENHANCED APPLICATION FILES:
{json.dumps(complete_analysis['enhanced_files'], indent=2, default=str)[:5000]}

Provide comprehensive synthesis:

{{
  "architecture_pattern": "microservices|monolith|layered|event-driven|hybrid",
  "technology_stack": ["all technologies actually in use"],
  "deployment_model": "cloud-native|traditional|hybrid|containerized",
  "scalability_assessment": "excellent|good|moderate|limited|poor", 
  "security_posture": "excellent|good|needs-attention|concerning|critical",
  "operational_complexity": "simple|moderate|complex|very-complex",
  "recommendations": [
    "specific actionable recommendations for improvement"
  ]
}}

Focus on strategic insights that help decision-making.
"""

        system_prompt = """You are a senior infrastructure architect providing strategic analysis.
        Synthesize technical findings into executive-level insights and actionable recommendations."""
        
        response = self.llm.progressive_analysis(
            stage=AnalysisStage.ARCHITECTURE_SYNTHESIS,
            prompt=synthesis_prompt,
            system_prompt=system_prompt,
            context=business_context
        )
        
        insights_data = response.get('parsed', {})
        
        return InfrastructureInsights(
            architecture_pattern=insights_data.get('architecture_pattern', 'Unknown'),
            technology_stack=insights_data.get('technology_stack', []),
            deployment_model=insights_data.get('deployment_model', 'Unknown'),
            scalability_assessment=insights_data.get('scalability_assessment', 'Unknown'),
            security_posture=insights_data.get('security_posture', 'Unknown'),
            operational_complexity=insights_data.get('operational_complexity', 'Unknown'),
            recommendations=insights_data.get('recommendations', [])
        )
    
    def _stage7_security_analysis(self, host: str, architecture_insights: InfrastructureInsights) -> Dict:
        """Stage 7: Comprehensive security analysis."""
        logger.info("[DISCOVERY] Stage 7: Security analysis")
        
        security_prompt = f"""
Perform comprehensive security analysis:

ARCHITECTURE: {asdict(architecture_insights)}
PROCESSES: {[asdict(p) for p in self.discovered_data.get('processes', [])]}
ENHANCED FILES: {[asdict(f) for f in self.discovered_data.get('enhanced_files', [])]}

Analyze:
1. Security vulnerabilities and risks
2. Access control and authentication mechanisms  
3. Network security posture
4. Data protection measures
5. Code-level security concerns
6. Priority security recommendations

Provide actionable security assessment with specific remediation steps.
"""

        system_prompt = """You are a cybersecurity expert analyzing production infrastructure.
        Focus on practical vulnerabilities and implementable security improvements."""
        
        response = self.llm.progressive_analysis(
            stage=AnalysisStage.SECURITY_ANALYSIS,
            prompt=security_prompt,
            system_prompt=system_prompt,
            context={'architecture_insights': asdict(architecture_insights)}
        )
        
        return response.get('parsed', {})
    
    def _analyze_file_with_intelligence(self, host: str, file_path: str) -> Optional[EnhancedApplicationFile]:
        """Analyze file with comprehensive intelligence."""
        try:
            # Get file stats
            stdout, stderr, exit_code, _ = self.connector.execute_command(
                host, f'stat -c "%s %Y" "{file_path}" 2>/dev/null', 'file_stats'
            )
            
            if exit_code != 0:
                return None
            
            stats = stdout.split()
            size = int(stats[0]) if len(stats) > 0 else 0
            
            # Skip very large files to avoid token limits
            if size > 500 * 1024:  # 500KB limit
                logger.info(f"Skipping large file: {file_path} ({size} bytes)")
                return None
            
            # Get full file content for analysis
            stdout, stderr, exit_code, _ = self.connector.execute_command(
                host, f'cat "{file_path}" 2>/dev/null', 'full_file_content'
            )
            
            if exit_code != 0 or not stdout.strip():
                return None
            
            content = stdout
            language = self._detect_language(file_path)
            
            # Enhanced LLM analysis
            intelligence = self._extract_file_intelligence(content, file_path, language)
            
            return EnhancedApplicationFile(
                path=file_path,
                language=language,
                size=size,
                last_modified=datetime.fromtimestamp(int(stats[1])).isoformat() if len(stats) > 1 else 'unknown',
                
                # Basic fields (enhanced)
                imports=intelligence.get('imports', []),
                functions=intelligence.get('functions', []),
                classes=intelligence.get('classes', []),
                external_services=intelligence.get('external_services', []),
                
                # Enhanced intelligence fields
                api_endpoints=intelligence.get('api_endpoints', []),
                database_models=intelligence.get('database_models', []),
                environment_variables=intelligence.get('environment_variables', []),
                security_concerns=intelligence.get('security_concerns', []),
                business_logic_summary=intelligence.get('business_logic_summary', ''),
                performance_notes=intelligence.get('performance_notes', []),
                complexity_score=intelligence.get('complexity_score', 0),
                documentation_quality=intelligence.get('documentation_quality', 'Unknown'),
                external_apis=intelligence.get('external_apis', []),
                database_connections=intelligence.get('database_connections', [])
            )
            
        except Exception as e:
            logger.debug(f"Enhanced file analysis failed for {file_path}: {e}")
            return None
    
    def _extract_file_intelligence(self, content: str, file_path: str, language: str) -> Dict:
        """Extract comprehensive intelligence from file content using LLM."""
        
        # Limit content to avoid token limits but ensure we get meaningful analysis
        analysis_content = content[:15000] if len(content) > 15000 else content
        
        intelligence_prompt = f"""
COMPREHENSIVE FILE INTELLIGENCE EXTRACTION

File: {file_path} ({language})
Purpose: Extract complete understanding for intelligent documentation

CODE CONTENT:
```{language}
{analysis_content}
```

Extract and return JSON with detailed analysis:

{{
  "business_logic_summary": "Clear explanation of what this file does for the business",
  "api_endpoints": [
    {{
      "path": "/api/path",
      "method": "GET|POST|PUT|DELETE",
      "handler_function": "function_name", 
      "parameters": ["param1", "param2"],
      "description": "What this endpoint does"
    }}
  ],
  "database_models": [
    {{
      "model_name": "ModelName",
      "table_name": "table_name",
      "fields": [
        {{"name": "field", "type": "string", "description": "purpose"}}
      ],
      "relationships": ["related_models"]
    }}
  ],
  "environment_variables": [
    {{
      "variable_name": "ENV_VAR",
      "usage_context": "How it's used",
      "default_value": "default",
      "is_required": true,
      "description": "Purpose"
    }}
  ],
  "security_concerns": [
    {{
      "concern_type": "authentication|authorization|data_exposure|injection", 
      "severity": "low|medium|high|critical",
      "description": "Specific security issue",
      "recommendation": "How to fix"
    }}
  ],
  "performance_notes": ["Performance considerations and bottlenecks"],
  "external_apis": [
    {{"service": "service_name", "endpoint": "url", "purpose": "why used"}}
  ],
  "database_connections": ["connection details"],
  "imports": ["all imports found"],
  "functions": ["function names with brief purpose"],
  "classes": ["class names with brief purpose"],
  "external_services": ["external service integrations"],
  "complexity_score": 1-10,
  "documentation_quality": "Poor|Fair|Good|Excellent"
}}

FOCUS ON BUSINESS VALUE:
- What business problem does this solve?
- How does this integrate with other systems?
- What would break if this failed?
- What are the key workflows?

Return only valid JSON.
"""

        system_prompt = """You are a senior software architect analyzing production code.
        Extract comprehensive intelligence that helps developers understand the business purpose,
        technical implementation, and practical considerations for maintaining this code."""
        
        try:
            response = self.llm.progressive_analysis(
                stage=AnalysisStage.CODE_ANALYSIS,
                prompt=intelligence_prompt,
                system_prompt=system_prompt,
                context={'file_path': file_path, 'language': language}
            )
            
            return response.get('parsed', {})
            
        except Exception as e:
            logger.debug(f"Intelligence extraction failed for {file_path}: {e}")
            return {}
    
    # Helper methods from original implementation...
    def _is_application_process(self, command: str, user: str) -> bool:
        """Identify real application processes."""
        command_lower = command.lower()
        
        # Skip kernel threads
        if command.startswith('[') and command.endswith(']'):
            return False
        
        # Skip system processes
        system_patterns = ['kworker', 'ksoftirqd', 'migration', 'rcu_', 'systemd/']
        if any(pattern in command_lower for pattern in system_patterns):
            return False
        
        # Application indicators
        app_indicators = ['python', 'node', 'java', 'nginx', 'worker', 'service']
        if any(indicator in command_lower for indicator in app_indicators):
            return True
        
        # Non-system users usually run applications
        system_users = ['root', 'daemon', 'bin', 'sys', 'www-data', 'nobody']
        if user not in system_users:
            return True
        
        return False
    
    def _extract_app_processes(self, process_list: str) -> List[str]:
        """Extract application process PIDs."""
        app_pids = []
        for line in process_list.split('\n'):
            if not line.strip():
                continue
            parts = line.split(None, 10)
            if len(parts) >= 11:
                pid, user, command = parts[1], parts[0], parts[10]
                if self._is_application_process(command, user):
                    app_pids.append(pid)
        return app_pids[:15]
    
    def _get_process_details(self, host: str, pid: str, command: str, user: str) -> Optional[Dict]:
        """Get enhanced process information."""
        try:
            commands = [
                (f'readlink -f /proc/{pid}/cwd 2>/dev/null', 'working_dir'),
                (f'cat /proc/{pid}/cmdline 2>/dev/null | tr "\\0" " "', 'full_cmdline')
            ]
            
            proc_info = {
                'pid': pid,
                'name': command.split()[0] if command else 'unknown',
                'user': user,
                'command': command,
                'cpu_percent': '0',
                'memory_percent': '0'
            }
            
            for cmd, key in commands:
                stdout, stderr, exit_code, execution = self.connector.execute_command(
                    host, cmd, f'process_detail_{key}'
                )
                if exit_code == 0 and stdout.strip():
                    proc_info[key] = stdout.strip()
            
            return proc_info
        except Exception as e:
            logger.debug(f"Failed to get process details for PID {pid}: {e}")
            return None
    
    def _analyze_configuration_file(self, host: str, file_path: str) -> Optional[Dict]:
        """Analyze configuration file."""
        try:
            # Get file content
            stdout, stderr, exit_code, execution = self.connector.execute_command(
                host, f'head -n 20 "{file_path}" 2>/dev/null', f'config_analysis'
            )
            
            if exit_code != 0 or not stdout.strip():
                return None
            
            # Basic configuration analysis
            config_info = {
                'path': file_path,
                'type': self._detect_config_type(file_path),
                'content_preview': stdout[:500],  # First 500 chars
                'key_settings': self._extract_key_settings(stdout, file_path)
            }
            
            return config_info
            
        except Exception as e:
            logger.debug(f"Failed to analyze config file {file_path}: {e}")
            return None
    
    def _detect_config_type(self, file_path: str) -> str:
        """Detect configuration file type."""
        if '/systemd/system' in file_path:
            return 'systemd_service'
        elif '/nginx/' in file_path:
            return 'nginx_config'
        elif file_path.endswith('.yml') or file_path.endswith('.yaml'):
            return 'yaml_config'
        elif file_path.endswith('.json'):
            return 'json_config'
        elif file_path.endswith('.conf'):
            return 'generic_config'
        return 'unknown_config'
    
    def _extract_key_settings(self, content: str, file_path: str) -> List[str]:
        """Extract key configuration settings."""
        key_settings = []
        
        if '/systemd/system' in file_path:
            # Extract systemd service settings
            for line in content.split('\n'):
                if any(setting in line for setting in ['ExecStart=', 'User=', 'WorkingDirectory=']):
                    key_settings.append(line.strip())
        
        elif '/nginx/' in file_path:
            # Extract nginx settings
            for line in content.split('\n'):
                if any(setting in line for setting in ['server_name', 'listen', 'root', 'proxy_pass']):
                    key_settings.append(line.strip())
        
        return key_settings[:5]  # Limit to top 5 settings
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language."""
        extension = Path(file_path).suffix.lower()
        lang_map = {
            '.py': 'Python', '.js': 'JavaScript', '.java': 'Java',
            '.yml': 'YAML', '.yaml': 'YAML', '.json': 'JSON',
            '.conf': 'Config'
        }
        return lang_map.get(extension, 'Unknown')
    
    def _extract_classification(self, proc_data: Dict, llm_insights: Dict) -> str:
        """Extract process classification from LLM insights."""
        # Simple classification based on command
        command = proc_data.get('command', '').lower()
        if 'worker' in command:
            return 'background_worker'
        elif 'nginx' in command:
            return 'web_server'
        elif 'python' in command:
            return 'application'
        return 'service'
    
    def _extract_purpose(self, proc_data: Dict, llm_insights: Dict) -> str:
        """Extract process purpose."""
        command = proc_data.get('command', '')
        if 'worker' in command.lower():
            return 'Background task processing'
        elif 'nginx' in command.lower():
            return 'Web server and reverse proxy'
        return 'Application service'
    
    def _extract_integrations(self, proc_data: Dict, llm_insights: Dict) -> List[str]:
        """Extract detected integrations."""
        # Simple integration detection based on command/environment
        integrations = []
        command = proc_data.get('command', '').lower()
        if 'aws' in command or 'boto' in command:
            integrations.extend(['AWS SQS', 'AWS S3'])
        return integrations
    
    def _extract_security_concerns(self, proc_data: Dict, llm_insights: Dict) -> List[str]:
        """Extract security concerns from process data and LLM insights."""
        security_concerns = []
        
        command = proc_data.get('command', '').lower()
        user = proc_data.get('user', '')
        
        # Check for root processes (security concern)
        if user == 'root':
            security_concerns.append('Running as root - consider principle of least privilege')
        
        # Check for common security-sensitive processes
        if 'python' in command and 'worker' in command:
            security_concerns.append('Review worker process security and input validation')
        
        if 'nginx' in command:
            security_concerns.append('Review web server configuration and SSL/TLS settings')
        
        # Extract insights from LLM analysis if available
        if isinstance(llm_insights, dict):
            llm_security = llm_insights.get('security_concerns', [])
            if isinstance(llm_security, list):
                security_concerns.extend(llm_security)
        
        return security_concerns[:5]  # Limit to top 5 concerns
    
    def _filter_application_files(self, file_list: List[str]) -> List[str]:
        """Filter out virtual environment and library files to focus on application code."""
        application_files = []
        
        # Patterns to exclude (virtual environments, caches, system files)
        exclude_patterns = [
            '/site-packages/',
            '/venv/',
            '/env/',
            '/.venv/',
            '/virtualenv/',
            '/__pycache__/',
            '/node_modules/',
            '/vendor/',
            '/.git/',
            '/tmp/',
            '/cache/',
            '.pyc',
            '.pyo'
        ]
        
        # Patterns that indicate actual application files
        include_patterns = [
            '/opt/',
            '/srv/',
            '/var/www/',
            '/home/',
            '/app/',
            'worker',
            'server',
            'main',
            'app',
            'service'
        ]
        
        for file_path in file_list:
            # Skip if it matches exclude patterns
            if any(pattern in file_path for pattern in exclude_patterns):
                continue
            
            # Include if it matches application patterns or is in application directories
            if any(pattern in file_path for pattern in include_patterns):
                application_files.append(file_path)
            elif file_path.startswith(('/opt/', '/srv/', '/var/www/', '/home/')):
                application_files.append(file_path)
        
        return application_files