#!/usr/bin/env python3
# enhanced_retriever.py
"""
Universal GraphRAG LLM-to-Cypher Pipeline with FastAPI Integration

An intelligent code retrieval system that adapts to any codebase:
0. Dynamic Schema Analysis → Learns codebase patterns automatically
1. Query Enhancement → LLM generates multiple search strategies  
2. Parallel Vector Search → Retrieves relevant entity context concurrently
3. Cypher Generation → Creates schema-aware queries with codebase context
4. Execution → Runs validated Cypher with fallbacks
5. Response Formatting → Natural language results with detailed info

Author: Praveen
License: MIT
"""

import json
import logging
import os
import re
import asyncio
import concurrent.futures
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
import dotenv
import sys
import uuid
from datetime import datetime

# Progress bar and aesthetic imports
from tqdm import tqdm
import colorama
from colorama import Fore, Back, Style

# Neo4j imports
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, CypherSyntaxError

# OpenAI for LLM pipeline
import openai
from openai import OpenAI

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)

# Configure logging to be less verbose during pretty output
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dotenv.load_dotenv()  # Load environment variables from .env file

# Global event storage for real-time streaming
EVENT_STREAMS = {}

@dataclass
class CodebaseProfile:
    """Dynamic profile of the codebase learned from analysis"""
    common_patterns: Dict[str, List[str]]
    entity_types: List[str]
    file_patterns: List[str]
    common_technologies: List[str]
    architectural_patterns: List[str]
    domain_concepts: Dict[str, List[str]]

@dataclass
class EnhancedQuery:
    """Enhanced query with multiple search strategies"""
    original_query: str
    intent: str
    search_strategies: List[Dict]
    expected_entity_types: List[str]
    query_complexity: str
    confidence: float

@dataclass
class SearchContext:
    """Enhanced context from parallel vector searches"""
    entity_names: List[str]
    entity_types: List[str]
    file_paths: List[str]
    relevant_nodes: List[Dict]
    confidence_scores: List[float]
    search_strategy_results: Dict[str, List[Dict]]
    total_unique_entities: int

@dataclass
class CypherQuery:
    """Generated Cypher query with enhanced metadata"""
    query: str
    reasoning: str
    expected_columns: List[str]
    is_valid: bool
    validation_error: str = None
    fallback_queries: List[str] = None

# Pydantic models for FastAPI
class QueryRequest(BaseModel):
    query: str
    session_id: str

class QueryResponse(BaseModel):
    response: str
    debug_info: Dict
    session_id: str
    timestamp: str

class StreamEvent(BaseModel):
    type: str  # "step", "success", "error", "info", "result"
    message: str
    step_number: Optional[int] = None
    data: Optional[Dict] = None

class PrettyPrinter:
    """Aesthetic console output with colors and formatting + FastAPI streaming"""
    
    current_session_id = None
    
    @staticmethod
    def set_session(session_id: str):
        """Set current session for streaming"""
        PrettyPrinter.current_session_id = session_id
        if session_id not in EVENT_STREAMS:
            EVENT_STREAMS[session_id] = []
    
    @staticmethod
    def _emit_event(event_type: str, message: str, step_number: int = None, data: Dict = None):
        """Emit event for both console and streaming"""
        if PrettyPrinter.current_session_id:
            event = {
                "type": event_type,
                "message": message,
                "step_number": step_number,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            EVENT_STREAMS[PrettyPrinter.current_session_id].append(event)
    
    @staticmethod
    def header(text: str):
        """Print a beautiful header"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}{Style.BRIGHT}🚀 {text}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        PrettyPrinter._emit_event("header", text)
    
    @staticmethod
    def step(step_num: int, title: str, description: str = ""):
        """Print a step with styling"""
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}📍 Step {step_num}: {title}{Style.RESET_ALL}")
        if description:
            print(f"{Fore.WHITE}   {description}{Style.RESET_ALL}")
        PrettyPrinter._emit_event("step", f"{title}: {description}", step_num)
    
    @staticmethod
    def success(message: str):
        """Print success message"""
        print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")
        PrettyPrinter._emit_event("success", message)
    
    @staticmethod
    def info(message: str):
        """Print info message"""
        print(f"{Fore.BLUE}ℹ️  {message}{Style.RESET_ALL}")
        PrettyPrinter._emit_event("info", message)
    
    @staticmethod
    def warning(message: str):
        """Print warning message"""
        print(f"{Fore.YELLOW}⚠️  {message}{Style.RESET_ALL}")
        PrettyPrinter._emit_event("warning", message)
    
    @staticmethod
    def error(message: str):
        """Print error message"""
        print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")
        PrettyPrinter._emit_event("error", message)
    
    @staticmethod
    def result(key: str, value: str):
        """Print key-value result"""
        print(f"{Fore.MAGENTA}📊 {key}:{Style.RESET_ALL} {value}")
        PrettyPrinter._emit_event("result", f"{key}: {value}")
    
    @staticmethod
    def separator():
        """Print a visual separator"""
        print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
        PrettyPrinter._emit_event("separator", "---")

class UniversalNeo4jLLMCypherRetriever:
    """
    Universal LLM-to-Cypher pipeline that adapts to any codebase.
    
    Learns codebase patterns dynamically and generates context-aware queries
    without hardcoded domain knowledge.
    """
    
    def __init__(self, openai_api_key: str = None):
        """Initialize with environment-based configuration"""
        PrettyPrinter.header("Initializing Universal LLM-to-Cypher Pipeline")
        
        # Neo4j configuration
        print(f"{Fore.BLUE}🔧 Setting up database connection...{Style.RESET_ALL}")
        self.neo4j_uri = os.getenv('NEO4J_URI')
        self.neo4j_user = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        self.neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
        if not self.neo4j_uri or not self.neo4j_password:
            raise ValueError("Neo4j credentials required. Set NEO4J_URI and NEO4J_PASSWORD environment variables.")
        
        # Initialize Neo4j connection
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password),
            database=self.neo4j_database
        )
        self._verify_connection()
        
        # OpenAI configuration
        print(f"{Fore.BLUE}🤖 Setting up AI models...{Style.RESET_ALL}")
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.openai_client = OpenAI(api_key=api_key)
        self.embedding_model = "text-embedding-3-small"
        self.llm_model = "gpt-4o-mini"
        
        # Learn codebase patterns dynamically (with error protection)
        print(f"{Fore.BLUE}📚 Learning codebase patterns...{Style.RESET_ALL}")
        try:
            # Schema analysis with progress
            with tqdm(total=4, desc="Analyzing codebase", colour="blue") as pbar:
                pbar.set_description("Extracting schema")
                self.schema = self._extract_schema()
                pbar.update(1)
                
                pbar.set_description("Learning patterns")
                self.codebase_profile = self._learn_codebase_patterns()
                pbar.update(1)
                
                pbar.set_description("Validating setup")
                pbar.update(1)
                
                pbar.set_description("Finalizing")
                pbar.update(1)
                
        except Exception as e:
            PrettyPrinter.error(f"Error during codebase analysis: {e}")
            # Provide minimal fallback
            self.schema = {"nodes": {}, "relationships": {}, "patterns": [], "samples": []}
            self.codebase_profile = CodebaseProfile(
                common_patterns={"cli_indicators": ["main"], "api_indicators": ["endpoint"]},
                entity_types=[],
                file_patterns=[],
                common_technologies=[],
                architectural_patterns=[],
                domain_concepts={}
            )
        
        # Enhanced configuration
        self.max_parallel_searches = 6
        self.vector_search_timeout = 30
        
        try:
            PrettyPrinter.success("Universal LLM-to-Cypher Pipeline initialized successfully!")
            if hasattr(self, 'codebase_profile') and self.codebase_profile:
                PrettyPrinter.info(f"Learned {len(self.codebase_profile.common_technologies)} technologies, {len(self.codebase_profile.domain_concepts)} domain concepts")
            else:
                PrettyPrinter.info("Using fallback codebase profile")
        except Exception as e:
            PrettyPrinter.warning(f"Initialization warning: {e}")
            PrettyPrinter.info("Universal LLM-to-Cypher Pipeline initialized with basic configuration")
    
    def _verify_connection(self):
        """Verify Neo4j connection"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 'Connected' as status")
                PrettyPrinter.success("Neo4j connection verified")
        except Exception as e:
            raise ConnectionError(f"Neo4j connection failed: {e}")
    
    def _extract_schema(self) -> Dict:
        """Extract comprehensive schema information"""
        try:
            with self.driver.session() as session:
                # Get node labels and properties
                node_info = session.run("""
                CALL db.schema.nodeTypeProperties()
                YIELD nodeType, propertyName, propertyTypes, mandatory
                RETURN nodeType, collect({
                    property: propertyName, 
                    types: propertyTypes, 
                    mandatory: mandatory
                }) as properties
                """)
                
                # Get relationship types
                rel_info = session.run("""
                CALL db.schema.relTypeProperties()
                YIELD relType, propertyName, propertyTypes, mandatory
                RETURN relType, collect({
                    property: propertyName,
                    types: propertyTypes,
                    mandatory: mandatory
                }) as properties
                """)
                
                # Get relationship patterns with counts
                patterns = session.run("""
                MATCH (a)-[r]->(b)
                RETURN DISTINCT labels(a)[0] as from_label, 
                       type(r) as relationship, 
                       labels(b)[0] as to_label,
                       count(*) as count
                ORDER BY count DESC
                LIMIT 50
                """)
                
                # Get sample entities
                samples = session.run("""
                MATCH (n)
                WHERE n.name IS NOT NULL
                RETURN labels(n)[0] as label, 
                       n.name as name,
                       n.file_path as file_path,
                       n.github_url as github_url,
                       n.qualified_name as qualified_name,
                       n.content[0..200] as content_preview
                ORDER BY n.name
                LIMIT 50
                """)
                
                schema = {
                    "nodes": {record["nodeType"]: record["properties"] for record in node_info},
                    "relationships": {record["relType"]: record["properties"] for record in rel_info},
                    "patterns": [dict(record) for record in patterns],
                    "samples": [dict(record) for record in samples]
                }
                
                PrettyPrinter.result("Schema", f"{len(schema['nodes'])} node types, {len(schema['relationships'])} relationship types")
                return schema
                
        except Exception as e:
            PrettyPrinter.error(f"Schema extraction failed: {e}")
            return {"nodes": {}, "relationships": {}, "patterns": [], "samples": []}
    
    def _learn_codebase_patterns(self) -> CodebaseProfile:
        """Learn codebase patterns dynamically using LLM analysis"""
        
        # Prepare codebase data for analysis
        sample_entities = []
        file_patterns = set()
        entity_names = []
        
        for sample in self.schema.get("samples", [])[:30]:
            if not sample:  # Skip None samples
                continue
                
            entity_names.append(sample.get("name", ""))
            
            # Safe file path processing
            file_path = sample.get("file_path", "")
            if file_path:
                try:
                    # Safe file extension extraction
                    file_parts = file_path.split("/")
                    if file_parts:
                        filename = file_parts[-1]
                        if "." in filename:
                            file_patterns.add(filename.split(".")[-1])
                    # Safe directory extraction
                    file_patterns.update([part for part in file_parts if part])
                except Exception:
                    pass  # Skip problematic file paths
            
            # Safe content processing
            content = sample.get("content_preview", "") or ""
            content_preview = content[:100] if content else ""
            
            sample_entities.append({
                "name": sample.get("name", ""),
                "type": sample.get("label", ""),
                "file": file_path,
                "content": content_preview
            })
        
        # Use LLM to analyze codebase patterns only if we have meaningful data
        if sample_entities and any(entity.get("name") for entity in sample_entities):
            analysis_prompt = f"""You are a codebase analysis expert. Analyze the provided code entities to identify patterns, technologies, and domain concepts.

CODEBASE SAMPLE DATA:
Entity Types: {list(self.schema['nodes'].keys())}
Sample Entities: {json.dumps(sample_entities[:20], indent=2)}
File Patterns: {list(file_patterns)[:20]}

Your task: Identify the key patterns and concepts in this codebase for better query understanding.

OUTPUT JSON FORMAT:
{{
    "common_patterns": {{
        "cli_indicators": ["main", "argparse", "parser", "argument"],
        "api_indicators": ["endpoint", "route", "handler", "api"],
        "database_indicators": ["query", "insert", "update", "delete", "db"],
        "test_indicators": ["test", "mock", "assert", "fixture"],
        "config_indicators": ["config", "settings", "env", "environment"],
        "auth_indicators": ["auth", "login", "token", "credential", "verify"]
    }},
    "common_technologies": ["technology1", "technology2"],
    "architectural_patterns": ["pattern1", "pattern2"],
    "domain_concepts": {{
        "core_functionality": ["concept1", "concept2"],
        "utilities": ["util1", "util2"],
        "integrations": ["integration1", "integration2"]
    }}
}}

Analyze the patterns in this codebase and respond with JSON only:"""

            try:
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.1,
                    max_tokens=800
                )
                
                result = json.loads(response.choices[0].message.content.strip())
                
                return CodebaseProfile(
                    common_patterns=result.get("common_patterns", {}),
                    entity_types=list(self.schema.get('nodes', {}).keys()),
                    file_patterns=list(file_patterns)[:20],
                    common_technologies=result.get("common_technologies", []),
                    architectural_patterns=result.get("architectural_patterns", []),
                    domain_concepts=result.get("domain_concepts", {})
                )
                
            except Exception as e:
                PrettyPrinter.warning(f"Codebase pattern learning failed: {e}")
        
        # Fallback profile with generic patterns (always returned if LLM fails or no data)
        return CodebaseProfile(
            common_patterns={
                "cli_indicators": ["main", "parse", "argument", "command"],
                "api_indicators": ["endpoint", "route", "handler", "api"],
                "database_indicators": ["query", "insert", "update", "delete"],
                "test_indicators": ["test", "mock", "assert"],
                "config_indicators": ["config", "settings", "env"]
            },
            entity_types=list(self.schema.get('nodes', {}).keys()),
            file_patterns=list(file_patterns)[:20],
            common_technologies=[],
            architectural_patterns=[],
            domain_concepts={}
        )
    
    def step0_enhance_query(self, user_query: str) -> EnhancedQuery:
        """
        Step 0: Enhanced query understanding using learned codebase patterns
        """
        PrettyPrinter.step(0, "Query Enhancement", "Analyzing intent and generating search strategies")
        
        # Prepare codebase context for LLM
        codebase_context = {
            "entity_types": self.codebase_profile.entity_types,
            "common_patterns": self.codebase_profile.common_patterns,
            "technologies": self.codebase_profile.common_technologies,
            "domain_concepts": self.codebase_profile.domain_concepts
        }
        
        enhancement_prompt = f"""You are an expert code intelligence query enhancer. Analyze the user's question and generate multiple search strategies optimized for vector similarity search.

User Query: "{user_query}"

LEARNED CODEBASE CONTEXT:
{json.dumps(codebase_context, indent=2)}

VECTOR SEARCH OPTIMIZATION PRINCIPLES:
1. **Semantic Density**: Use descriptive phrases that embed well
2. **Context Richness**: Include related concepts and synonyms  
3. **Technical Precision**: Include implementation-specific terms
4. **Conceptual Breadth**: Cover different aspects of the query

SEARCH STRATEGY TYPES:
- **exact_match**: Direct entity names and specific terms
- **semantic**: Conceptual descriptions and functional intent  
- **technical**: Implementation patterns, frameworks, libraries
- **contextual**: Related domain concepts and use cases
- **hierarchical**: Parent/child relationships and dependencies
- **cross_reference**: Interacting entities and relationships

VECTOR SEARCH TEXT GUIDELINES:
✅ Good: "authentication login security user verification credentials token"
✅ Good: "command line interface argument parser main function CLI script"
✅ Good: "database insert update delete query CRUD operations data persistence"

❌ Avoid: Single words ("auth", "CLI")  
❌ Avoid: Too technical ("argparse.ArgumentParser.add_argument")
❌ Avoid: Too generic ("function", "class")

OUTPUT JSON FORMAT:
{{
    "intent": "clear description of what user wants to find",
    "query_complexity": "simple|moderate|complex",
    "expected_entity_types": ["Function", "Class"],
    "confidence": 0.85,
    "search_strategies": [
        {{
            "type": "exact_match",
            "search_text": "specific descriptive terms for vector search",
            "node_types": ["Function", "Class"],
            "purpose": "find exact matches",
            "max_results": 10,
            "priority": 1
        }},
        {{
            "type": "semantic",
            "search_text": "semantic conceptual description with context",
            "node_types": ["Function", "Class"],
            "purpose": "find semantically related entities",
            "max_results": 15,
            "priority": 2
        }}
    ]
}}

EXAMPLES:

Query: "CLI commands"  
Intent: "Find command-line interface implementations and argument parsers"
Strategies:
- exact_match: "main function argument parser command line interface"
- semantic: "command line arguments parsing user input script execution"
- technical: "argparse click parser configuration command options"

Query: "authentication functions"
Intent: "Find user authentication and security-related functions"  
Strategies:
- exact_match: "authenticate login verify user credentials"
- semantic: "user authentication security access control validation"
- technical: "token session password hash verification security"

Query: "database operations"
Intent: "Find database interaction and data persistence functions"
Strategies:
- exact_match: "database insert update delete query"
- semantic: "data persistence CRUD operations database interaction"
- technical: "SQL connection query execution database client"

Analyze the query using the learned codebase patterns: "{user_query}"

Respond with JSON only:"""

        try:
            # Show progress for LLM call
            with tqdm(total=1, desc="Generating search strategies", colour="green") as pbar:
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": enhancement_prompt}],
                    temperature=0.1,
                    max_tokens=800
                )
                pbar.update(1)
            
            result = json.loads(response.choices[0].message.content.strip())
            
            enhanced_query = EnhancedQuery(
                original_query=user_query,
                intent=result.get("intent", ""),
                search_strategies=result.get("search_strategies", []),
                expected_entity_types=result.get("expected_entity_types", []),
                query_complexity=result.get("query_complexity", "simple"),
                confidence=result.get("confidence", 0.5)
            )
            
            PrettyPrinter.success(f"Generated {len(enhanced_query.search_strategies)} search strategies")
            PrettyPrinter.result("Intent", enhanced_query.intent)
            PrettyPrinter.result("Complexity", enhanced_query.query_complexity)
            PrettyPrinter.result("Confidence", f"{enhanced_query.confidence:.2f}")
            
            return enhanced_query
            
        except Exception as e:
            PrettyPrinter.warning(f"Query enhancement failed: {e}")
            # Fallback enhancement with generic strategies
            return self._generate_fallback_enhancement(user_query)
    
    def _generate_fallback_enhancement(self, user_query: str) -> EnhancedQuery:
        """Generate fallback enhancement when LLM fails"""
        PrettyPrinter.warning("Using fallback query enhancement")
        
        # Simple keyword-based enhancement
        query_lower = user_query.lower()
        
        strategies = []
        
        # Basic semantic strategy
        strategies.append({
            "type": "semantic",
            "search_text": f"{user_query} implementation functionality code",
            "node_types": ["Function", "Class"],
            "purpose": "Basic semantic search",
            "max_results": 20,
            "priority": 1
        })
        
        # Add specific strategies based on keywords
        if any(term in query_lower for term in ["cli", "command", "argument", "parse"]):
            strategies.append({
                "type": "technical",
                "search_text": "command line argument parser main function script",
                "node_types": ["Function"],
                "purpose": "Find CLI-related code",
                "max_results": 15,
                "priority": 2
            })
        
        if any(term in query_lower for term in ["auth", "login", "security", "credential"]):
            strategies.append({
                "type": "technical", 
                "search_text": "authentication login security user verification",
                "node_types": ["Function", "Class"],
                "purpose": "Find authentication code",
                "max_results": 15,
                "priority": 2
            })
        
        return EnhancedQuery(
            original_query=user_query,
            intent=f"Find entities related to: {user_query}",
            search_strategies=strategies,
            expected_entity_types=["Function", "Class"],
            query_complexity="simple",
            confidence=0.3
        )
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for vector search with error handling"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text[:8000]
            )
            return response.data[0].embedding
        except Exception as e:
            PrettyPrinter.error(f"Embedding generation failed: {e}")
            return []
    
    def _execute_single_vector_search(self, search_strategy: Dict) -> List[Dict]:
        """Execute a single vector search strategy with optimized text"""
        try:
            # The search_text should already be optimized for vector search from the enhancement step
            embedding = self._generate_embedding(search_strategy["search_text"])
            if not embedding:
                return []
            
            # Determine vector index based on node types
            if "Class" in search_strategy["node_types"] and "Function" not in search_strategy["node_types"]:
                index_name = "class_embeddings"
            else:
                index_name = "code_embeddings"
            
            # Lowered similarity threshold to 0.4 for better recall with optimized search texts
            vector_query = """
            CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
            YIELD node, score
            WHERE score >= 0.4
            RETURN 
                node.name as name,
                node.node_type as node_type,
                node.file_path as file_path,
                node.qualified_name as qualified_name,
                node.id as node_id,
                node.content as content,
                score,
                node {.*} as full_node
            ORDER BY score DESC
            """
            
            with self.driver.session() as session:
                result = session.run(
                    vector_query,
                    index_name=index_name,
                    limit=search_strategy["max_results"],
                    embedding=embedding
                )
                
                results = []
                for record in result:
                    results.append({
                        "name": record["name"],
                        "node_type": record["node_type"],
                        "file_path": record["file_path"],
                        "qualified_name": record["qualified_name"],
                        "node_id": record["node_id"],
                        "content": record["content"],
                        "score": record["score"],
                        "full_node": dict(record["full_node"]),
                        "search_strategy": search_strategy["type"]
                    })
                
                return results
                
        except Exception as e:
            return []
    
    def step2_execute_parallel_searches(self, enhanced_query: EnhancedQuery) -> SearchContext:
        """
        Step 2: Execute multiple vector searches in parallel
        """
        PrettyPrinter.step(2, "Parallel Vector Search", f"Executing {len(enhanced_query.search_strategies)} search strategies")
        
        strategy_results = {}
        all_entities = []
        all_types = []
        all_paths = []
        all_nodes = []
        all_scores = []
        
        # Execute searches in parallel with progress bar
        with tqdm(total=len(enhanced_query.search_strategies), desc="Vector searches", colour="blue") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_searches) as executor:
                future_to_strategy = {
                    executor.submit(self._execute_single_vector_search, strategy): strategy
                    for strategy in enhanced_query.search_strategies
                }
                
                for future in concurrent.futures.as_completed(future_to_strategy, timeout=self.vector_search_timeout):
                    strategy = future_to_strategy[future]
                    try:
                        results = future.result()
                        strategy_type = strategy["type"]
                        strategy_results[strategy_type] = results
                        
                        # Update progress bar with strategy info
                        pbar.set_description(f"Completed {strategy_type}")
                        pbar.update(1)
                        
                        # Aggregate results with priority weighting
                        for result in results:
                            all_entities.append(result["name"])
                            all_types.append(result["node_type"])
                            all_paths.append(result["file_path"])
                            all_nodes.append(result["full_node"])
                            # Weight scores by strategy priority
                            weighted_score = result["score"] * (1.0 / strategy.get("priority", 1))
                            all_scores.append(weighted_score)
                            
                    except Exception as e:
                        pbar.set_description(f"Failed {strategy['type']}")
                        pbar.update(1)
                        strategy_results[strategy["type"]] = []
        
        # Deduplicate while preserving highest scores
        unique_entities = {}
        for i, node_id in enumerate([node.get("id") for node in all_nodes]):
            if node_id and (node_id not in unique_entities or all_scores[i] > unique_entities[node_id]["score"]):
                unique_entities[node_id] = {
                    "name": all_entities[i],
                    "type": all_types[i], 
                    "path": all_paths[i],
                    "node": all_nodes[i],
                    "score": all_scores[i],
                    "index": i
                }
        
        # Rebuild deduplicated lists
        dedup_entities = []
        dedup_types = []
        dedup_paths = []
        dedup_nodes = []
        dedup_scores = []
        
        for entity_data in sorted(unique_entities.values(), key=lambda x: x["score"], reverse=True):
            dedup_entities.append(entity_data["name"])
            dedup_types.append(entity_data["type"])
            dedup_paths.append(entity_data["path"])
            dedup_nodes.append(entity_data["node"])
            dedup_scores.append(entity_data["score"])
        
        total_results = sum(len(results) for results in strategy_results.values())
        
        PrettyPrinter.success(f"Parallel search completed!")
        PrettyPrinter.result("Total Results", str(total_results))
        PrettyPrinter.result("Unique Entities", str(len(unique_entities)))
        
        # Show strategy breakdown
        for strategy_type, results in strategy_results.items():
            if results:
                PrettyPrinter.result(f"Strategy '{strategy_type}'", f"{len(results)} matches")
        
        return SearchContext(
            entity_names=dedup_entities,
            entity_types=dedup_types,
            file_paths=dedup_paths,
            relevant_nodes=dedup_nodes,
            confidence_scores=dedup_scores,
            search_strategy_results=strategy_results,
            total_unique_entities=len(unique_entities)
        )
    
    def step3_generate_cypher(self, enhanced_query: EnhancedQuery, search_context: SearchContext) -> CypherQuery:
        """
        Step 3: Generate Cypher using learned codebase patterns
        """
        PrettyPrinter.step(3, "Cypher Generation", "Creating schema-aware database query")
        
        # Prepare context summary
        context_summary = []
        strategy_summaries = []
        
        for strategy_type, results in search_context.search_strategy_results.items():
            if results:
                strategy_summaries.append(f"**{strategy_type}**: {len(results)} results")
                for result in results[:3]:
                    context_summary.append(f"- {result['name']} ({result['node_type']}) in {result['file_path']} [score: {result['score']:.2f}, strategy: {strategy_type}]")
        
        context_text = "\n".join(context_summary) if context_summary else "No relevant entities found"
        strategies_text = " | ".join(strategy_summaries)
        
        # Generate schema context
        schema_text = self._format_schema_for_llm()
        
        # Use learned patterns to generate domain examples
        domain_examples = self._generate_adaptive_examples(enhanced_query.intent)
        
        cypher_prompt = f"""You are an expert Neo4j Cypher query generator. Generate precise queries based on user intent and retrieved context.

User Intent: "{enhanced_query.intent}"
Original Query: "{enhanced_query.original_query}"
Query Complexity: {enhanced_query.query_complexity}

Search Results Summary: {strategies_text}
Retrieved Context (top matches per strategy):
{context_text}

Database Schema:
{schema_text}

Learned Codebase Patterns:
{domain_examples}

CYPHER GENERATION GUIDELINES:

🎯 **CRITICAL RULES:**
1. **Direct Entity Queries**: Query entities directly, avoid unnecessary JOINs
2. **Context-Driven**: Use retrieved entity names to guide query construction
3. **Intent-Focused**: Match Cypher logic to user's actual intent
4. **Detailed URLs**: Get github_url from entities for precise line/column anchors

✅ **PREFERRED PATTERNS:**

For specific entity lookups:
```cypher
MATCH (n {{name: "EntityName"}})
WHERE n:Function OR n:Class
RETURN n.name, n.node_type, n.file_path, n.line_start, n.line_end, n.github_url, n.qualified_name
```

For pattern-based searches:
```cypher
MATCH (n)
WHERE n.content CONTAINS "keyword" OR n.name CONTAINS "pattern"
RETURN n.name, n.node_type, n.file_path, n.github_url, n.content[0..300]
```

For relationship queries:
```cypher
MATCH (f:Function)-[:HAS_PARAMETER]->(p:Parameter)
WHERE f.name = "target_function"
RETURN f.name, f.file_path, f.github_url, collect(p.name) as parameters
```

**CONTEXT UTILIZATION:**
Retrieved entity names: {[name for name in search_context.entity_names[:10]]}
Use these actual entities when building your query.

**RETURN REQUIREMENTS:**
Always include: name, node_type, file_path, line_start, line_end, github_url
For functions: also include parameters, content preview
For classes: also include methods, base_classes

OUTPUT JSON FORMAT:
{{
    "cypher": "optimized Cypher query here",
    "reasoning": "detailed explanation of query logic",
    "expected_columns": ["name", "file_path", "line_start", "github_url"],
    "explanation": "what this query accomplishes", 
    "fallback_queries": [
        "alternative query for broader search",
        "backup query if main fails"
    ]
}}

Generate the optimal Cypher query for: "{enhanced_query.original_query}"

Respond with JSON only:"""

        try:
            # Show progress for Cypher generation
            with tqdm(total=1, desc="Generating Cypher query", colour="yellow") as pbar:
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": cypher_prompt}],
                    temperature=0.1,
                    max_tokens=1000
                )
                pbar.update(1)
            
            result = json.loads(response.choices[0].message.content.strip())
            
            cypher_query = CypherQuery(
                query=result["cypher"],
                reasoning=result.get("reasoning", ""),
                expected_columns=result.get("expected_columns", []),
                is_valid=True,
                fallback_queries=result.get("fallback_queries", [])
            )
            
            # Validate syntax
            cypher_query = self._validate_cypher(cypher_query)
            
            if cypher_query.is_valid:
                PrettyPrinter.success("Cypher query generated and validated")
            else:
                PrettyPrinter.warning("Cypher validation failed, using fallback")
                
            return cypher_query
            
        except Exception as e:
            PrettyPrinter.error(f"Cypher generation failed: {e}")
            return CypherQuery(
                query="",
                reasoning="Failed to generate query",
                expected_columns=[],
                is_valid=False,
                validation_error=str(e),
                fallback_queries=[]
            )
    
    def _format_schema_for_llm(self) -> str:
        """Format schema information for LLM context"""
        schema_parts = []
        
        # Node types and properties
        schema_parts.append("🏗️ NODE TYPES AND PROPERTIES:")
        for node_type, properties in self.schema["nodes"].items():
            priority_props = []
            other_props = []
            
            for prop in properties:
                prop_name = prop["property"]
                if prop_name in ["name", "file_path", "line_start", "line_end", "github_url", "qualified_name", "content", "node_type"]:
                    priority_props.append(prop_name)
                else:
                    other_props.append(prop_name)
            
            all_props = priority_props + other_props[:4]
            schema_parts.append(f"- **{node_type}**: {', '.join(all_props)}")
        
        # Entity URL precision
        schema_parts.append("\n🎯 GITHUB URL PRECISION:")
        schema_parts.append("✅ **Entity URLs**: Detailed with line/column anchors (#L37C1-L68C27)")
        schema_parts.append("❌ **File URLs**: Basic file-level only")
        
        # Relationship patterns
        schema_parts.append("\n🔗 RELATIONSHIP PATTERNS:")
        for pattern in self.schema["patterns"][:10]:
            count = pattern.get("count", 0)
            schema_parts.append(f"- ({pattern['from_label']})-[:{pattern['relationship']}]->({pattern['to_label']}) [{count}]")
        
        # Sample entities
        if self.schema["samples"]:
            schema_parts.append("\n📋 SAMPLE ENTITIES:")
            for sample in self.schema["samples"][:5]:
                name = sample.get('name', 'N/A')
                entity_type = sample.get('label', 'N/A')
                file_path = sample.get('file_path', 'N/A')
                schema_parts.append(f"- **{name}** ({entity_type}) in {file_path}")
        
        return "\n".join(schema_parts)
    
    def _generate_adaptive_examples(self, intent: str) -> str:
        """Generate examples based on learned codebase patterns"""
        examples = []
        
        # Use learned patterns to generate relevant examples
        patterns = self.codebase_profile.common_patterns
        
        # CLI-related examples
        if any(indicator in intent.lower() for indicator in patterns.get("cli_indicators", [])):
            examples.append(f"""
**CLI Pattern Examples (learned from codebase):**
Common CLI indicators: {', '.join(patterns.get('cli_indicators', []))}
```cypher
MATCH (f:Function)
WHERE f.content CONTAINS "main" OR f.content CONTAINS "argument"
RETURN f.name, f.file_path, f.github_url, f.content[0..500]
```""")
        
        # API-related examples  
        if any(indicator in intent.lower() for indicator in patterns.get("api_indicators", [])):
            examples.append(f"""
**API Pattern Examples (learned from codebase):**
Common API indicators: {', '.join(patterns.get('api_indicators', []))}
```cypher
MATCH (f:Function)
WHERE f.content CONTAINS "endpoint" OR f.content CONTAINS "route"
RETURN f.name, f.file_path, f.github_url
```""")
        
        # Database-related examples
        if any(indicator in intent.lower() for indicator in patterns.get("database_indicators", [])):
            examples.append(f"""
**Database Pattern Examples (learned from codebase):**
Common DB indicators: {', '.join(patterns.get('database_indicators', []))}
```cypher
MATCH (f:Function)
WHERE f.content CONTAINS "query" OR f.content CONTAINS "insert"
RETURN f.name, f.file_path, f.github_url
```""")
        
        return "\n".join(examples) if examples else "// No specific patterns learned for this query type"
    
    def _validate_cypher(self, cypher_query: CypherQuery) -> CypherQuery:
        """Validate Cypher syntax with fallback testing"""
        try:
            with self.driver.session() as session:
                session.run(f"EXPLAIN {cypher_query.query}")
                cypher_query.is_valid = True
                return cypher_query
                
        except CypherSyntaxError as e:
            PrettyPrinter.warning(f"Primary Cypher syntax error, trying fallbacks")
            cypher_query.is_valid = False
            cypher_query.validation_error = str(e)
            
            # Try fallback queries
            if cypher_query.fallback_queries:
                for i, fallback in enumerate(cypher_query.fallback_queries):
                    try:
                        with self.driver.session() as session:
                            session.run(f"EXPLAIN {fallback}")
                        PrettyPrinter.success(f"Fallback query {i+1} validated successfully")
                        cypher_query.query = fallback
                        cypher_query.is_valid = True
                        cypher_query.validation_error = None
                        cypher_query.reasoning += f" (using fallback query {i+1})"
                        return cypher_query
                    except Exception:
                        continue
            
            return cypher_query
            
        except Exception as e:
            PrettyPrinter.warning(f"Cypher validation failed: {e}")
            cypher_query.is_valid = False
            cypher_query.validation_error = str(e)
            return cypher_query
    
    def step4_execute_cypher(self, cypher_query: CypherQuery) -> List[Dict]:
        """Execute validated Cypher query with fallback support"""
        PrettyPrinter.step(4, "Query Execution", "Running Cypher query against database")
        
        if not cypher_query.is_valid:
            PrettyPrinter.error(f"Cannot execute invalid Cypher: {cypher_query.validation_error}")
            return []
        
        try:
            with tqdm(total=1, desc="Executing query", colour="green") as pbar:
                with self.driver.session() as session:
                    result = session.run(cypher_query.query)
                    records = [dict(record) for record in result]
                    pbar.update(1)
                    
            PrettyPrinter.success(f"Query executed successfully")
            PrettyPrinter.result("Records Found", str(len(records)))
            return records
                
        except Exception as e:
            PrettyPrinter.error(f"Cypher execution failed: {e}")
            
            # Try fallback queries
            if cypher_query.fallback_queries:
                for i, fallback in enumerate(cypher_query.fallback_queries):
                    try:
                        PrettyPrinter.info(f"Trying fallback query {i+1}...")
                        with self.driver.session() as session:
                            result = session.run(fallback)
                            records = [dict(record) for record in result]
                            PrettyPrinter.success(f"Fallback query {i+1} returned {len(records)} records")
                            return records
                    except Exception as fallback_error:
                        PrettyPrinter.warning(f"Fallback query {i+1} failed")
                        continue
            
            return []
    
    def step5_format_response(self, enhanced_query: EnhancedQuery, cypher_query: CypherQuery, 
                             query_results: List[Dict], search_context: SearchContext) -> str:
        """Format comprehensive response with learned context"""
        PrettyPrinter.step(5, "Response Formatting", "Creating comprehensive answer")
        
        if not query_results:
            return self._format_no_results_response(enhanced_query, cypher_query, search_context)
        
        # Prepare results summary
        results_text = []
        for i, record in enumerate(query_results[:10], 1):
            result_parts = []
            for key, value in record.items():
                if value is not None:
                    if key == 'github_url':
                        result_parts.append(f"{key}: {value}")
                    elif key == 'content' and len(str(value)) > 200:
                        result_parts.append(f"{key}: {str(value)[:200]}...")
                    elif isinstance(value, list):
                        result_parts.append(f"{key}: {', '.join(map(str, value))}")
                    else:
                        result_parts.append(f"{key}: {value}")
            results_text.append(f"{i}. {' | '.join(result_parts)}")
        
        results_summary = "\n".join(results_text)
        
        # Strategy info
        strategy_info = []
        for strategy_type, results in search_context.search_strategy_results.items():
            if results:
                strategy_info.append(f"{strategy_type}: {len(results)} matches")
        
        strategy_summary = " | ".join(strategy_info) if strategy_info else "No strategies executed"
        
        response_prompt = f"""You are a helpful code intelligence assistant. Format a comprehensive response based on the analysis.

User Intent: "{enhanced_query.intent}"
Original Query: "{enhanced_query.original_query}"

Search Strategy Results: {strategy_summary}
Total Unique Entities: {search_context.total_unique_entities}

Generated Cypher:
```cypher
{cypher_query.query}
```

Query Results ({len(query_results)} total):
{results_summary}

FORMATTING GUIDELINES:

📋 **STRUCTURE:**
1. **Summary**: Direct answer to user's query
2. **Details**: Specific findings with locations
3. **Context**: Additional insights

📍 **FOR EACH RESULT:**
- **Name & Type**: Entity name and classification
- **Location**: `file_path:L{{line_start}}-L{{line_end}}`
- **GitHub**: [View Code](complete_github_url)
- **Context**: Relevant details (parameters, content preview, etc.)

🔗 **LINK REQUIREMENTS:**
- Preserve complete GitHub URLs with line anchors
- Format as clickable markdown links
- Never truncate github_url values

💡 **CONTENT FOCUS:**
- Highlight most relevant results first
- Include code snippets when helpful
- Note search strategy contributions
- Suggest related queries if relevant

Generate a comprehensive response for: "{enhanced_query.original_query}"

Format as markdown:"""

        try:
            with tqdm(total=1, desc="Formatting response", colour="magenta") as pbar:
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": response_prompt}],
                    temperature=0.3,
                    max_tokens=1200
                )
                pbar.update(1)
            
            PrettyPrinter.success("Response formatted successfully")
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            PrettyPrinter.error(f"Response formatting failed: {e}")
            return self._format_fallback_response(enhanced_query, query_results, search_context)
    
    def _format_no_results_response(self, enhanced_query: EnhancedQuery, cypher_query: CypherQuery, 
                                   search_context: SearchContext) -> str:
        """Format response when no results found"""
        if not cypher_query.is_valid:
            return f"""## Query Processing Failed

**Query**: "{enhanced_query.original_query}"
**Intent**: {enhanced_query.intent}

**Error**: {cypher_query.validation_error}

**Suggestions**:
- Try rephrasing with different terms
- Check entity name spelling
- Use broader search concepts

**Context**: Found {search_context.total_unique_entities} related entities during search"""
        
        strategy_info = []
        for strategy_type, results in search_context.search_strategy_results.items():
            if results:
                strategy_info.append(f"- **{strategy_type}**: {len(results)} related entities")
        
        related_entities = "\n".join(strategy_info) if strategy_info else "- No related entities found"
        
        return f"""## No Direct Matches Found

**Query**: "{enhanced_query.original_query}"
**Intent**: {enhanced_query.intent}

**Search Results**:
{related_entities}

**Generated Query**:
```cypher
{cypher_query.query}
```

**Suggestions**:
- Try broader search terms
- Look for partial matches in related entities above
- Consider alternative query phrasings"""
    
    def _format_fallback_response(self, enhanced_query: EnhancedQuery, query_results: List[Dict], 
                                 search_context: SearchContext) -> str:
        """Fallback response when LLM formatting fails"""
        response = f"""## Results for: "{enhanced_query.original_query}"

**Intent**: {enhanced_query.intent}
**Found**: {len(query_results)} results

"""
        
        for i, record in enumerate(query_results[:5], 1):
            response += f"### {i}. Result\n"
            for key, value in record.items():
                if value is not None:
                    if key == 'github_url':
                        response += f"- **GitHub**: [View Code]({value})\n"
                    elif key == 'content' and len(str(value)) > 200:
                        response += f"- **{key}**: {str(value)[:200]}...\n"
                    else:
                        response += f"- **{key}**: {value}\n"
            response += "\n"
        
        return response
    
    def search(self, user_query: str, session_id: str = None) -> Tuple[str, Dict]:
        """
        Universal search interface that adapts to any codebase
        
        Returns:
            Tuple[str, Dict]: (response, debug_info)
        """
        start_time = time.time()
        debug_info = {}
        
        # Set session for streaming
        if session_id:
            PrettyPrinter.set_session(session_id)
        
        PrettyPrinter.header(f"Processing Query: \"{user_query}\"")
        
        try:
            # Step 0: Enhanced query understanding with learned patterns
            enhanced_query = self.step0_enhance_query(user_query)
            debug_info["enhanced_query"] = {
                "intent": enhanced_query.intent,
                "complexity": enhanced_query.query_complexity,
                "strategies_count": len(enhanced_query.search_strategies),
                "confidence": enhanced_query.confidence
            }
            
            # Step 2: Parallel vector searches with optimized texts
            search_context = self.step2_execute_parallel_searches(enhanced_query)
            debug_info["search_context"] = {
                "total_results": sum(len(results) for results in search_context.search_strategy_results.values()),
                "unique_entities": search_context.total_unique_entities,
                "strategy_breakdown": {k: len(v) for k, v in search_context.search_strategy_results.items()}
            }
            
            # Step 3: Adaptive Cypher generation
            cypher_query = self.step3_generate_cypher(enhanced_query, search_context)
            debug_info["cypher_query"] = {
                "query": cypher_query.query,
                "is_valid": cypher_query.is_valid,
                "reasoning": cypher_query.reasoning,
                "has_fallbacks": len(cypher_query.fallback_queries or []) > 0
            }
            
            # Step 4: Execute with fallbacks
            query_results = self.step4_execute_cypher(cypher_query)
            debug_info["results_count"] = len(query_results)
            
            # Step 5: Adaptive formatting
            response = self.step5_format_response(enhanced_query, cypher_query, query_results, search_context)
            
            elapsed_time = time.time() - start_time
            debug_info["execution_time"] = f"{elapsed_time:.2f}s"
            debug_info["pipeline_version"] = "universal_v1"
            debug_info["learned_technologies"] = self.codebase_profile.common_technologies
            
            PrettyPrinter.separator()
            PrettyPrinter.success(f"Pipeline completed in {elapsed_time:.2f}s")
            PrettyPrinter.separator()
            
            return response, debug_info
            
        except Exception as e:
            PrettyPrinter.error(f"Universal pipeline failed: {e}")
            return f"An error occurred: {str(e)}", {"error": str(e)}
    
    def close(self):
        """Close database connections"""
        if self.driver:
            self.driver.close()
            PrettyPrinter.success("Neo4j connection closed")


# Initialize FastAPI app
app = FastAPI(title="Universal LLM-to-Cypher API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retriever instance
retriever = None

@app.on_event("startup")
async def startup_event():
    """Initialize retriever on startup"""
    global retriever
    try:
        retriever = UniversalNeo4jLLMCypherRetriever()
        print("✅ Retriever initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize retriever: {e}")
        retriever = None

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global retriever
    if retriever:
        retriever.close()

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process a query request"""
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    
    try:
        # Process query in background
        response, debug_info = retriever.search(request.query, request.session_id)
        
        return QueryResponse(
            response=response,
            debug_info=debug_info,
            session_id=request.session_id,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream/{session_id}")
async def stream_events(session_id: str):
    """Stream real-time events for a session"""
    def event_generator():
        if session_id not in EVENT_STREAMS:
            EVENT_STREAMS[session_id] = []
        
        last_index = 0
        while True:
            events = EVENT_STREAMS[session_id]
            if len(events) > last_index:
                for event in events[last_index:]:
                    yield f"data: {json.dumps(event)}\n\n"
                last_index = len(events)
            
            time.sleep(0.5)  # Poll every 500ms
    
    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/events/{session_id}")
async def get_events(session_id: str):
    """Get all events for a session"""
    if session_id not in EVENT_STREAMS:
        return {"events": []}
    
    return {"events": EVENT_STREAMS[session_id]}

@app.post("/new_session")
async def new_session():
    """Create a new session"""
    session_id = str(uuid.uuid4())
    EVENT_STREAMS[session_id] = []
    return {"session_id": session_id}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "retriever_initialized": retriever is not None,
        "timestamp": datetime.now().isoformat()
    }

# Usage example (for standalone running)
def main():
    """Example usage of the Universal LLM-to-Cypher Pipeline"""
    try:
        retriever = UniversalNeo4jLLMCypherRetriever()
        
        test_queries = [
            "How to create a simple chain"
        ]
        
        for query in test_queries:
            response, debug = retriever.search(query)
            
            # Print the beautiful response
            print(f"\n{Fore.GREEN}{Style.BRIGHT}🎯 QUERY RESULTS{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            print(f"\n{response}")
            
            # Print debug information with colors
            print(f"\n{Fore.YELLOW}{Style.BRIGHT}🔍 DEBUG INFORMATION{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
            PrettyPrinter.result("Technologies Learned", str(debug.get('learned_technologies', [])))
            PrettyPrinter.result("Intent", debug.get('enhanced_query', {}).get('intent', 'N/A'))
            PrettyPrinter.result("Strategies", str(debug.get('enhanced_query', {}).get('strategies_count', 0)))
            PrettyPrinter.result("Unique Entities", str(debug.get('search_context', {}).get('unique_entities', 0)))
            PrettyPrinter.result("Final Results", str(debug.get('results_count', 0)))
            PrettyPrinter.result("Execution Time", debug.get('execution_time', 'N/A'))
        
        retriever.close()
        
    except Exception as e:
        PrettyPrinter.error(f"Error: {e}")
        print(f"{Fore.YELLOW}Ensure environment variables are set: NEO4J_URI, NEO4J_PASSWORD, OPENAI_API_KEY{Style.RESET_ALL}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true", help="Run as FastAPI server")
    parser.add_argument("--port", type=int, default=8001, help="Port for FastAPI server")
    args = parser.parse_args()
    
    if args.api:
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        main()