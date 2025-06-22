#!/usr/bin/env python3
# knowledge_graph_generator.py
"""
Neo4j AuraDB Code Intelligence Importer

A comprehensive tool for importing AST analysis results into Neo4j AuraDB
with vector embeddings for advanced code intelligence and similarity search.

Features:
- Neo4j AuraDB cloud connectivity
- Hierarchical relationship preservation
- Vector embeddings for semantic search
- Batch processing for performance
- Comprehensive relationship mapping
- Enhanced query examples and utilities
- Semantic-based function discovery
- Accurate GitHub URL preservation

Author: Praveen
License: MIT
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import asyncio
from dataclasses import dataclass
import dotenv

dotenv.load_dotenv()
# Core imports
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, Neo4jError

# Embedding imports with fallbacks
EMBEDDINGS_AVAILABLE = False
EMBEDDING_ERROR = None

try:
    # Try OpenAI first (most robust) - new v1.0+ syntax
    from openai import OpenAI
    import openai
    OPENAI_AVAILABLE = True
    logger_temp = logging.getLogger(__name__)
    logger_temp.info(f"OpenAI library version: {openai.__version__}")
except ImportError as e:
    OPENAI_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(f"OpenAI import failed: {e}")

try:
    # Fallback to Sentence Transformers
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('neo4j_importer.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Neo4jConfig:
    """
    Neo4j connection configuration for AuraDB instances.
    
    Attributes:
        uri (str): Neo4j connection URI
        username (str): Database username
        password (str): Database password
        database (str): Database name
        instance_id (str): AuraDB instance ID
        instance_name (str): Human-readable instance name
    """
    uri: str
    username: str
    password: str
    database: str
    instance_id: str
    instance_name: str

@dataclass
class ImportStats:
    """
    Statistics tracking for the import process.
    
    Attributes:
        nodes_created (int): Number of nodes created
        relationships_created (int): Number of relationships created
        embeddings_created (int): Number of embeddings generated
        errors (int): Number of errors encountered
        start_time (float): Import start timestamp
        end_time (float): Import end timestamp
    """
    nodes_created: int = 0
    relationships_created: int = 0
    embeddings_created: int = 0
    errors: int = 0
    start_time: float = 0
    end_time: float = 0


class EmbeddingGenerator:
    """
    Optimized embedding generator with batching and rate limiting.
    """
    
    def __init__(self, model_type: str = "auto"):
        self.model_type = model_type
        self.model = None
        self.embedding_dim = 384
        self.openai_client = None
        
        # Batching configuration
        self.EMBEDDING_BATCH_SIZE = 100  # OpenAI allows up to 2048 inputs per request
        self.REQUEST_DELAY = 0.1  # 100ms between requests to avoid rate limits
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model with fallback support."""
        global EMBEDDINGS_AVAILABLE, EMBEDDING_ERROR
        
        if self.model_type == "openai" or (self.model_type == "auto" and OPENAI_AVAILABLE):
            try:
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                
                # Initialize OpenAI client with new v1.0+ syntax
                self.openai_client = OpenAI(api_key=api_key)
                
                # Test with a small request
                test_response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input="test"
                )
                
                self.model_type = "openai"
                self.embedding_dim = 1536
                EMBEDDINGS_AVAILABLE = True
                logger.info("[OK] Initialized OpenAI embeddings (text-embedding-3-small)")
                logger.info(f"[BATCH] Batch size: {self.EMBEDDING_BATCH_SIZE} texts per request")
                return
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {e}")
                self.openai_client = None
        
        if self.model_type == "sentence_transformers" or (self.model_type == "auto" and SENTENCE_TRANSFORMERS_AVAILABLE):
            try:
                model_name = "all-MiniLM-L6-v2"
                self.model = SentenceTransformer(model_name)
                self.model_type = "sentence_transformers"
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                EMBEDDINGS_AVAILABLE = True
                logger.info("[OK] Initialized Sentence Transformers: {model_name}")
                return
            except Exception as e:
                logger.warning(f"Sentence Transformers initialization failed: {e}")
        
        EMBEDDINGS_AVAILABLE = False
        EMBEDDING_ERROR = "No embedding models available"
        logger.warning("[WARN] No embeddings available")
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batches for efficiency.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[Optional[List[float]]]: List of embeddings (same order as input)
        """
        if not EMBEDDINGS_AVAILABLE or not texts:
            return [None] * len(texts)
        
        all_embeddings = []
        total_batches = (len(texts) + self.EMBEDDING_BATCH_SIZE - 1) // self.EMBEDDING_BATCH_SIZE
        
        logger.info(f"[BATCH] Generating embeddings for {len(texts)} texts in {total_batches} batches...")
        
        for i in range(0, len(texts), self.EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + self.EMBEDDING_BATCH_SIZE]
            batch_num = (i // self.EMBEDDING_BATCH_SIZE) + 1
            
            try:
                if self.model_type == "openai" and self.openai_client:
                    # Rate limiting
                    if i > 0:
                        time.sleep(self.REQUEST_DELAY)
                    
                    logger.info(f"[API] Batch {batch_num}/{total_batches}: Requesting {len(batch)} embeddings...")
                    
                    response = self.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=[text[:8000] for text in batch]  # Truncate long texts
                    )
                    
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    logger.info(f"[OK] Batch {batch_num}/{total_batches} completed ({len(batch_embeddings)} embeddings)")
                
                elif self.model_type == "sentence_transformers" and self.model:
                    logger.info(f"[PROC] Batch {batch_num}/{total_batches}: Processing {len(batch)} texts...")
                    
                    batch_embeddings = self.model.encode(batch)
                    all_embeddings.extend([emb.tolist() for emb in batch_embeddings])
                    
                    logger.info(f"[OK] Batch {batch_num}/{total_batches} completed")
                
                else:
                    # Fallback: return None for this batch
                    all_embeddings.extend([None] * len(batch))
                    
            except Exception as e:
                logger.error(f"[ERROR] Batch {batch_num}/{total_batches} failed: {e}")
                # Add None embeddings for failed batch
                all_embeddings.extend([None] * len(batch))
                
                # Exponential backoff for rate limit errors
                if "rate_limit" in str(e).lower():
                    wait_time = min(2 ** (batch_num % 5), 30)  # Max 30 seconds
                    logger.warning(f"[WAIT] Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
        
        logger.info(f"[DONE] Embedding generation complete: {len([e for e in all_embeddings if e is not None])}/{len(texts)} successful")
        return all_embeddings
    
    def prepare_code_texts(self, nodes: List[Dict]) -> List[str]:
        """
        Prepare rich text representations for code nodes.
        
        Args:
            nodes (List[Dict]): List of code nodes
            
        Returns:
            List[str]: List of prepared texts for embedding
        """
        texts = []
        for node in nodes:
            text_parts = []
            
            # Add node name and type
            text_parts.append(f"{node.get('node_type', '')} {node.get('name', '')}")
            
            # Add semantic information
            if node.get('purpose'):
                text_parts.append(f"purpose: {node['purpose']}")
            
            if node.get('domain'):
                text_parts.append(f"domain: {node['domain']}")
            
            if node.get('technologies'):
                text_parts.append(f"technologies: {', '.join(node['technologies'])}")
            
            if node.get('operations'):
                text_parts.append(f"operations: {', '.join(node['operations'])}")
            
            # Add docstring if available
            if node.get('docstring'):
                text_parts.append(node['docstring'])
            
            # Add parameters for functions
            if node.get('parameters'):
                text_parts.append(f"parameters: {', '.join(node['parameters'])}")
            
            # Add limited content
            if node.get('content'):
                content = node['content'][:500]  # Limit content length
                text_parts.append(content)
            
            # Add file path context
            if node.get('location', {}).get('file_path'):
                file_path = node['location']['file_path']
                path_parts = file_path.replace('\\', '/').split('/')
                meaningful_parts = [p for p in path_parts if p and not p.startswith('.')]
                text_parts.append(f"file: {'/'.join(meaningful_parts[-2:])}")
            
            combined_text = " | ".join(text_parts)
            texts.append(combined_text)
        
        return texts
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate single embedding (uses batch method for consistency).
        """
        embeddings = self.generate_embeddings_batch([text])
        return embeddings[0] if embeddings else None
    
    def generate_code_embedding(self, node: Dict) -> Optional[List[float]]:
        """
        Generate single code embedding (uses batch method for consistency).
        """
        texts = self.prepare_code_texts([node])
        embeddings = self.generate_embeddings_batch(texts)
        return embeddings[0] if embeddings else None


class Neo4jCodeImporter:
    """
    Main importer for Neo4j AuraDB with enhanced code intelligence features.
    
    This class handles the complete process of importing AST analysis results
    into Neo4j, creating optimized schemas, and establishing comprehensive
    relationships for code navigation and discovery.
    """
    
    def __init__(self, config: Neo4jConfig, embedding_model: str = "auto"):
        """
        Initialize the Neo4j code importer.
        
        Args:
            config (Neo4jConfig): Neo4j connection configuration
            embedding_model (str): Embedding model type to use
        """
        self.config = config
        self.driver: Optional[Driver] = None
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.stats = ImportStats()
        
        # Batch sizes for performance optimization
        self.BATCH_SIZE = 1000
        self.RELATIONSHIP_BATCH_SIZE = 5000
        
        self._connect()
    
    def _connect(self):
        """
        Establish connection to Neo4j AuraDB with proper error handling.
        
        Raises:
            Exception: If connection cannot be established
        """
        try:
            logger.info(f"Connecting to Neo4j AuraDB: {self.config.instance_name}")
            
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                database=self.config.database
            )
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info("Successfully connected to Neo4j AuraDB")
                else:
                    raise Exception("Connection test failed")
                    
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection and cleanup resources."""
        if self.driver:
            self.driver.close()
    
    def clear_database(self, confirm: bool = False):
        """
        Clear all data from database - USE WITH EXTREME CAUTION.
        
        Args:
            confirm (bool): Must be True to actually clear the database
        """
        if not confirm:
            logger.warning("Database clear not confirmed - skipping")
            return
        
        logger.warning("CLEARING ALL DATA FROM DATABASE")
        with self.driver.session() as session:
            # Remove all constraints and indexes first
            session.run("CALL apoc.schema.assert({}, {})")
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")
    
    def create_schema(self):
        """
        Create optimized schema with indexes and constraints for code intelligence.
        
        This method sets up the database schema optimized for code navigation,
        search, and relationship discovery.
        """
        logger.info("Creating Neo4j schema...")
        
        with self.driver.session() as session:
            # Create constraints for unique nodes
            constraints = [
                "CREATE CONSTRAINT function_id_unique IF NOT EXISTS FOR (f:Function) REQUIRE f.id IS UNIQUE",
                "CREATE CONSTRAINT class_id_unique IF NOT EXISTS FOR (c:Class) REQUIRE c.id IS UNIQUE", 
                "CREATE CONSTRAINT import_id_unique IF NOT EXISTS FOR (i:Import) REQUIRE i.id IS UNIQUE",
                "CREATE CONSTRAINT file_path_unique IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE",
                "CREATE CONSTRAINT parameter_id_unique IF NOT EXISTS FOR (p:Parameter) REQUIRE p.id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.debug(f"Created constraint: {constraint.split()[2]}")
                except Exception as e:
                    logger.debug(f"Constraint already exists or failed: {e}")
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX function_name_idx IF NOT EXISTS FOR (f:Function) ON (f.name)",
                "CREATE INDEX class_name_idx IF NOT EXISTS FOR (c:Class) ON (c.name)",
                "CREATE INDEX file_name_idx IF NOT EXISTS FOR (f:File) ON (f.name)",
                "CREATE INDEX node_type_idx IF NOT EXISTS FOR (n) ON (n.node_type)",
                "CREATE INDEX complexity_idx IF NOT EXISTS FOR (f:Function) ON (f.complexity)",
                "CREATE INDEX qualified_name_idx IF NOT EXISTS FOR (f:Function) ON (f.qualified_name)",
                
                # Enhanced indexes for semantic search
                "CREATE INDEX purpose_idx IF NOT EXISTS FOR (f:Function) ON (f.purpose)",
                "CREATE INDEX domain_idx IF NOT EXISTS FOR (f:Function) ON (f.domain)",
                "CREATE INDEX operations_idx IF NOT EXISTS FOR (f:Function) ON (f.operations)",
                "CREATE INDEX technologies_idx IF NOT EXISTS FOR (f:Function) ON (f.technologies)",
                "CREATE INDEX class_purpose_idx IF NOT EXISTS FOR (c:Class) ON (c.purpose)",
                "CREATE INDEX class_domain_idx IF NOT EXISTS FOR (c:Class) ON (c.domain)"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                    logger.debug(f"Created index: {index.split()[2]}")
                except Exception as e:
                    logger.debug(f"Index already exists or failed: {e}")
            
            # Create vector index for embeddings if available
            if EMBEDDINGS_AVAILABLE:
                try:
                    vector_index = f"""
                    CREATE VECTOR INDEX code_embeddings IF NOT EXISTS
                    FOR (n:Function) ON (n.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {self.embedding_generator.embedding_dim},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                    """
                    session.run(vector_index)
                    logger.info(f"Created vector index with {self.embedding_generator.embedding_dim} dimensions")
                    
                    # Also create for classes
                    class_vector_index = f"""
                    CREATE VECTOR INDEX class_embeddings IF NOT EXISTS
                    FOR (n:Class) ON (n.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {self.embedding_generator.embedding_dim},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                    """
                    session.run(class_vector_index)
                    
                except Exception as e:
                    logger.warning(f"Vector index creation failed: {e}")
        
        logger.info("Enhanced schema creation completed")
    
    def import_analysis_file(self, json_file_path: str):
        """
        Import AST analysis JSON file into Neo4j with comprehensive processing.
        
        This method orchestrates the complete import process including metadata,
        files, nodes, relationships, and enhanced code intelligence features.
        
        Args:
            json_file_path (str): Path to the AST analysis JSON file
        """
        logger.info(f"Starting import from: {json_file_path}")
        self.stats.start_time = time.time()
        
        # Load JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get('analysis_metadata', {})
        nodes = data.get('nodes', [])
        
        logger.info(f"Loaded {len(nodes)} nodes from analysis")
        logger.info(f"Repository: {metadata.get('repo_name', 'Unknown')}")
        logger.info(f"Branch: {metadata.get('repo_branch', 'main')}")
        
        # Import in stages for optimal performance
        self._import_repository_metadata(metadata)
        self._import_files(nodes)
        self._import_nodes_batch(nodes)
        self._create_relationships(nodes)
        self._create_enhanced_relationships()
        
        self.stats.end_time = time.time()
        self._print_import_summary()
    
    def _import_repository_metadata(self, metadata: Dict):
        """
        Import repository-level metadata with enhanced information.
        
        Args:
            metadata (Dict): Repository metadata from analysis
        """
        logger.info("Importing repository metadata...")
        
        with self.driver.session() as session:
            query = """
            MERGE (repo:Repository {url: $repo_url})
            SET repo.name = $repo_name,
                repo.branch = $repo_branch,
                repo.analysis_timestamp = $timestamp,
                repo.total_nodes = $total_nodes,
                repo.analyzer_version = $analyzer_version,
                repo.tree_sitter_available = $tree_sitter,
                repo.semantic_analysis_enabled = $semantic_analysis
            RETURN repo
            """
            
            session.run(query, {
                'repo_url': metadata.get('repo_url', ''),
                'repo_name': metadata.get('repo_name', 'Unknown'),
                'repo_branch': metadata.get('repo_branch', 'main'),
                'timestamp': metadata.get('analysis_timestamp', 0),
                'total_nodes': metadata.get('total_nodes', 0),
                'analyzer_version': metadata.get('analyzer_version', ''),
                'tree_sitter': metadata.get('tree_sitter_available', False),
                'semantic_analysis': metadata.get('semantic_analysis_enabled', False)
            })
        
        logger.info("Repository metadata imported")
    
    def _import_files(self, nodes: List[Dict]):
        """
        Import file nodes and create hierarchical file structure.
        
        Args:
            nodes (List[Dict]): List of all nodes from analysis
        """
        logger.info("Importing file structure...")
        
        # Extract unique files with enhanced metadata
        files = {}
        for node in nodes:
            file_path = node.get('location', {}).get('file_path')
            if file_path and file_path not in files:
                # Extract branch from first node's repo_url if available
                repo_branch = "main"  # default
                if hasattr(nodes[0], 'get') and nodes[0].get('repo_url'):
                    # Try to extract branch from github_url pattern
                    github_url = node.get('github_url', '')
                    if '/blob/' in github_url:
                        try:
                            branch_part = github_url.split('/blob/')[1].split('/')[0]
                            if branch_part:
                                repo_branch = branch_part
                        except:
                            pass
                
                files[file_path] = {
                    'path': file_path,
                    'name': file_path.split('/')[-1],
                    'extension': file_path.split('.')[-1] if '.' in file_path else '',
                    'directory': '/'.join(file_path.split('/')[:-1]) if '/' in file_path else '',
                    'repo_url': node.get('repo_url', ''),
                    'github_url': f"{node.get('repo_url', '').rstrip('.git')}/blob/{repo_branch}/{file_path}" if node.get('repo_url') else ''
                }
        
        # Batch insert files
        with self.driver.session() as session:
            for i in range(0, len(files), self.BATCH_SIZE):
                batch = list(files.values())[i:i + self.BATCH_SIZE]
                
                query = """
                UNWIND $files as file
                MERGE (f:File {path: file.path})
                SET f.name = file.name,
                    f.extension = file.extension,
                    f.directory = file.directory,
                    f.repo_url = file.repo_url,
                    f.github_url = file.github_url,
                    f.created_at = timestamp()
                """
                
                session.run(query, {'files': batch})
        
        logger.info(f"Imported {len(files)} files")
    
    def _import_nodes_batch(self, nodes: List[Dict]):
        """
        Import code nodes in batches with optimized embedding generation.
        
        Args:
            nodes (List[Dict]): List of all nodes to import
        """
        logger.info("[START] Starting optimized node import with batch embeddings...")
        
        # Group nodes by type for efficient processing
        functions = [n for n in nodes if n.get('node_type') == 'function']
        classes = [n for n in nodes if n.get('node_type') == 'class']
        imports = [n for n in nodes if n.get('node_type') == 'import']
        
        logger.info(f"[INFO] Node breakdown: {len(functions)} functions, {len(classes)} classes, {len(imports)} imports")
        
        # Import each type with enhanced processing
        if functions:
            self._import_functions(functions)
        
        if classes:
            self._import_classes(classes)
        
        if imports:
            self._import_imports(imports)
    
    def _import_functions(self, functions: List[Dict]):
        """
        Import function nodes with batched embeddings for optimal performance.
        
        Args:
            functions (List[Dict]): List of function nodes to import
        """
        logger.info(f"[START] Importing {len(functions)} functions with batch embeddings...")
        
        # Generate ALL embeddings in batches FIRST
        if EMBEDDINGS_AVAILABLE:
            logger.info("[PREP] Preparing texts for embedding...")
            texts = self.embedding_generator.prepare_code_texts(functions)
            
            logger.info("[BATCH] Generating embeddings in batches...")
            embeddings = self.embedding_generator.generate_embeddings_batch(texts)
            
            # Assign embeddings back to functions
            successful_embeddings = 0
            for func, embedding in zip(functions, embeddings):
                if embedding:
                    func['embedding'] = embedding
                    successful_embeddings += 1
            
            self.stats.embeddings_created += successful_embeddings
            logger.info(f"[OK] Generated {successful_embeddings}/{len(functions)} function embeddings")
        
        # Now import functions to Neo4j in batches
        with self.driver.session() as session:
            for i in range(0, len(functions), self.BATCH_SIZE):
                batch = functions[i:i + self.BATCH_SIZE]
                
                query = """
                UNWIND $functions as func
                MERGE (f:Function {id: func.id})
                SET f.name = func.name,
                    f.qualified_name = COALESCE(func.qualified_name, func.name),
                    f.file_path = func.location.file_path,
                    f.line_start = func.location.line_start,
                    f.line_end = func.location.line_end,
                    f.column_start = COALESCE(func.location.column_start, 0),
                    f.column_end = COALESCE(func.location.column_end, 0),
                    f.content = func.content,
                    f.complexity = func.complexity,
                    f.is_async = func.is_async,
                    f.is_method = func.is_method,
                    f.docstring = func.docstring,
                    f.github_url = func.github_url,
                    f.repo_url = func.repo_url,
                    f.created_at = func.created_at,
                    f.node_type = 'function'
                
                // Set embedding if available
                FOREACH (embedding IN CASE WHEN func.embedding IS NOT NULL THEN [func.embedding] ELSE [] END |
                    SET f.embedding = embedding
                )
                
                // Connect to file
                WITH f, func
                MATCH (file:File {path: func.location.file_path})
                MERGE (f)-[:DEFINED_IN]->(file)
                """
                
                session.run(query, {'functions': batch})
                self.stats.nodes_created += len(batch)
                
                # Progress logging
                imported_count = min(i + self.BATCH_SIZE, len(functions))
                if imported_count % (self.BATCH_SIZE * 2) == 0 or imported_count == len(functions):
                    logger.info(f"[SAVE] Imported {imported_count}/{len(functions)} functions to Neo4j...")
        
        # Create parameters as separate nodes
        self._import_function_parameters(functions)
    
    def _import_function_parameters(self, functions: List[Dict]):
        """
        Import function parameters as separate nodes with relationships.
        
        Args:
            functions (List[Dict]): List of function nodes
        """
        logger.info("Creating parameter nodes and relationships...")
        
        parameters = []
        for func in functions:
            func_id = func.get('id')
            for i, param in enumerate(func.get('parameters', [])):
                param_id = f"{func_id}_param_{i}"
                parameters.append({
                    'id': param_id,
                    'name': param,
                    'function_id': func_id,
                    'position': i
                })
        
        if not parameters:
            return
        
        with self.driver.session() as session:
            for i in range(0, len(parameters), self.BATCH_SIZE):
                batch = parameters[i:i + self.BATCH_SIZE]
                
                query = """
                UNWIND $parameters as param
                MERGE (p:Parameter {id: param.id})
                SET p.name = param.name,
                    p.position = param.position,
                    p.created_at = timestamp()
                
                WITH p, param
                MATCH (f:Function {id: param.function_id})
                MERGE (f)-[:HAS_PARAMETER]->(p)
                """
                
                session.run(query, {'parameters': batch})
                self.stats.nodes_created += len(batch)
                self.stats.relationships_created += len(batch)
    
    def _import_classes(self, classes: List[Dict]):
        """
        Import class nodes with batched embeddings for optimal performance.
        
        Args:
            classes (List[Dict]): List of class nodes to import
        """
        logger.info(f"[START] Importing {len(classes)} classes with batch embeddings...")
        
        # Generate ALL embeddings in batches FIRST
        if EMBEDDINGS_AVAILABLE:
            logger.info("[PREP] Preparing texts for embedding...")
            texts = self.embedding_generator.prepare_code_texts(classes)
            
            logger.info("[BATCH] Generating embeddings in batches...")
            embeddings = self.embedding_generator.generate_embeddings_batch(texts)
            
            # Assign embeddings back to classes
            successful_embeddings = 0
            for cls, embedding in zip(classes, embeddings):
                if embedding:
                    cls['embedding'] = embedding
                    successful_embeddings += 1
            
            self.stats.embeddings_created += successful_embeddings
            logger.info(f"[OK] Generated {successful_embeddings}/{len(classes)} class embeddings")
        
        # Now import classes to Neo4j in batches
        with self.driver.session() as session:
            for i in range(0, len(classes), self.BATCH_SIZE):
                batch = classes[i:i + self.BATCH_SIZE]
                
                query = """
                UNWIND $classes as cls
                MERGE (c:Class {id: cls.id})
                SET c.name = cls.name,
                    c.qualified_name = COALESCE(cls.qualified_name, cls.name),
                    c.file_path = cls.location.file_path,
                    c.line_start = cls.location.line_start,
                    c.line_end = cls.location.line_end,
                    c.column_start = COALESCE(cls.location.column_start, 0),
                    c.column_end = COALESCE(cls.location.column_end, 0),
                    c.content = cls.content,
                    c.docstring = cls.docstring,
                    c.github_url = cls.github_url,
                    c.repo_url = cls.repo_url,
                    c.created_at = cls.created_at,
                    c.node_type = 'class'
                
                // Set embedding if available
                FOREACH (embedding IN CASE WHEN cls.embedding IS NOT NULL THEN [cls.embedding] ELSE [] END |
                    SET c.embedding = embedding
                )
                
                // Connect to file
                WITH c, cls
                MATCH (file:File {path: cls.location.file_path})
                MERGE (c)-[:DEFINED_IN]->(file)
                """
                
                session.run(query, {'classes': batch})
                self.stats.nodes_created += len(batch)
                
                # Progress logging
                imported_count = min(i + self.BATCH_SIZE, len(classes))
                if imported_count % (self.BATCH_SIZE * 2) == 0 or imported_count == len(classes):
                    logger.info(f"[SAVE] Imported {imported_count}/{len(classes)} classes to Neo4j...")
    
    def _import_imports(self, imports: List[Dict]):
        """
        Import import nodes and create module relationships with dependency categorization.
        
        Args:
            imports (List[Dict]): List of import nodes to import
        """
        logger.info(f"Importing {len(imports)} imports with dependency information...")
        
        with self.driver.session() as session:
            for i in range(0, len(imports), self.BATCH_SIZE):
                batch = imports[i:i + self.BATCH_SIZE]
                
                query = """
                UNWIND $imports as imp
                MERGE (i:Import {id: imp.id})
                SET i.name = imp.name,
                    i.module = imp.module,
                    i.import_type = imp.import_type,
                    i.file_path = imp.location.file_path,
                    i.line_start = imp.location.line_start,
                    i.line_end = COALESCE(imp.location.line_end, imp.location.line_start),
                    i.column_start = COALESCE(imp.location.column_start, 0),
                    i.column_end = COALESCE(imp.location.column_end, 0),
                    i.content = imp.content,
                    i.github_url = imp.github_url,
                    i.created_at = imp.created_at,
                    i.node_type = 'import'
                
                // Connect to file
                WITH i, imp
                MATCH (file:File {path: imp.location.file_path})
                MERGE (file)-[:IMPORTS]->(i)
                
                // Create module node if it doesn't exist
                WITH i, imp
                MERGE (m:Module {name: imp.module})
                SET m.created_at = timestamp()
                MERGE (i)-[:FROM_MODULE]->(m)
                """
                
                session.run(query, {'imports': batch})
                self.stats.nodes_created += len(batch) * 2  # imports + modules
                self.stats.relationships_created += len(batch) * 2
    
    def _create_relationships(self, nodes: List[Dict]):
        """
        Create relationships between code elements with enhanced accuracy.
        
        Args:
            nodes (List[Dict]): List of all nodes for relationship creation
        """
        logger.info("Creating code relationships...")
        
        # Create function call relationships
        self._create_function_calls(nodes)
        
        # Create class inheritance relationships
        self._create_class_inheritance(nodes)
        
        # Create class-method relationships
        self._create_class_methods(nodes)
    
    def _create_function_calls(self, nodes: List[Dict]):
        """
        Create CALLS relationships between functions with improved accuracy.
        
        Args:
            nodes (List[Dict]): List of all nodes
        """
        logger.info("Creating function call relationships...")
        
        function_calls = []
        functions = [n for n in nodes if n.get('node_type') == 'function']
        
        # Build function name to ID mapping with qualified names for better accuracy
        func_name_to_id = {}
        qualified_name_to_id = {}
        
        for func in functions:
            name = func.get('name')
            qualified_name = func.get('qualified_name')
            func_id = func.get('id')
            
            if name:
                if name not in func_name_to_id:
                    func_name_to_id[name] = []
                func_name_to_id[name].append(func_id)
            
            if qualified_name:
                qualified_name_to_id[qualified_name] = func_id
        
        # Create call relationships with priority for qualified names
        for func in functions:
            caller_id = func.get('id')
            caller_file = func.get('location', {}).get('file_path', '')
            
            for called_func in func.get('calls', []):
                # First try qualified name match within same file
                caller_module = caller_file.replace('/', '.').replace('\\', '.').replace('.py', '')
                qualified_called = f"{caller_module}.{called_func}"
                
                if qualified_called in qualified_name_to_id:
                    callee_id = qualified_name_to_id[qualified_called]
                    if caller_id != callee_id:
                        function_calls.append({
                            'caller_id': caller_id,
                            'callee_id': callee_id,
                            'called_name': called_func,
                            'resolution_type': 'qualified'
                        })
                        continue
                
                # Fallback to name-based matching
                if called_func in func_name_to_id:
                    for callee_id in func_name_to_id[called_func]:
                        if caller_id != callee_id:
                            function_calls.append({
                                'caller_id': caller_id,
                                'callee_id': callee_id,
                                'called_name': called_func,
                                'resolution_type': 'name_based'
                            })
                            break  # Only take first match to avoid duplicates
        
        if not function_calls:
            logger.info("No function calls to create")
            return
        
        logger.info(f"Creating {len(function_calls)} function call relationships...")
        
        with self.driver.session() as session:
            for i in range(0, len(function_calls), self.RELATIONSHIP_BATCH_SIZE):
                batch = function_calls[i:i + self.RELATIONSHIP_BATCH_SIZE]
                
                query = """
                UNWIND $calls as call
                MATCH (caller:Function {id: call.caller_id})
                MATCH (callee:Function {id: call.callee_id})
                MERGE (caller)-[r:CALLS]->(callee)
                SET r.called_name = call.called_name,
                    r.resolution_type = call.resolution_type,
                    r.created_at = timestamp()
                """
                
                session.run(query, {'calls': batch})
                self.stats.relationships_created += len(batch)
    
    def _create_class_inheritance(self, nodes: List[Dict]):
        """
        Create INHERITS relationships between classes with enhanced matching.
        
        Args:
            nodes (List[Dict]): List of all nodes
        """
        logger.info("Creating class inheritance relationships...")
        
        classes = [n for n in nodes if n.get('node_type') == 'class']
        
        # Build class name to ID mapping with qualified names
        class_name_to_id = {}
        qualified_name_to_id = {}
        
        for cls in classes:
            name = cls.get('name')
            qualified_name = cls.get('qualified_name')
            cls_id = cls.get('id')
            
            if name:
                if name not in class_name_to_id:
                    class_name_to_id[name] = []
                class_name_to_id[name].append(cls_id)
            
            if qualified_name:
                qualified_name_to_id[qualified_name] = cls_id
        
        inheritance = []
        for cls in classes:
            child_id = cls.get('id')
            child_file = cls.get('location', {}).get('file_path', '')
            
            for base_class in cls.get('base_classes', []):
                # Try qualified name first
                child_module = child_file.replace('/', '.').replace('\\', '.').replace('.py', '')
                qualified_base = f"{child_module}.{base_class}"
                
                if qualified_base in qualified_name_to_id:
                    parent_id = qualified_name_to_id[qualified_base]
                    if child_id != parent_id:
                        inheritance.append({
                            'child_id': child_id,
                            'parent_id': parent_id,
                            'base_name': base_class,
                            'resolution_type': 'qualified'
                        })
                        continue
                
                # Fallback to name-based matching
                if base_class in class_name_to_id:
                    for parent_id in class_name_to_id[base_class]:
                        if child_id != parent_id:
                            inheritance.append({
                                'child_id': child_id,
                                'parent_id': parent_id,
                                'base_name': base_class,
                                'resolution_type': 'name_based'
                            })
                            break
        
        if not inheritance:
            logger.info("No class inheritance to create")
            return
        
        logger.info(f"Creating {len(inheritance)} inheritance relationships...")
        
        with self.driver.session() as session:
            for i in range(0, len(inheritance), self.RELATIONSHIP_BATCH_SIZE):
                batch = inheritance[i:i + self.RELATIONSHIP_BATCH_SIZE]
                
                query = """
                UNWIND $inheritance as inh
                MATCH (child:Class {id: inh.child_id})
                MATCH (parent:Class {id: inh.parent_id})
                MERGE (child)-[r:INHERITS]->(parent)
                SET r.base_name = inh.base_name,
                    r.resolution_type = inh.resolution_type,
                    r.created_at = timestamp()
                """
                
                session.run(query, {'inheritance': batch})
                self.stats.relationships_created += len(batch)
    
    def _create_class_methods(self, nodes: List[Dict]):
        """
        Create BELONGS_TO relationships between methods and classes with improved detection.
        
        Args:
            nodes (List[Dict]): List of all nodes
        """
        logger.info("Creating class-method relationships...")
        
        methods = [n for n in nodes if n.get('node_type') == 'function' and n.get('is_method')]
        classes = [n for n in nodes if n.get('node_type') == 'class']
        
        # Group by file for efficient matching
        classes_by_file = {}
        for cls in classes:
            file_path = cls.get('location', {}).get('file_path')
            if file_path:
                if file_path not in classes_by_file:
                    classes_by_file[file_path] = []
                classes_by_file[file_path].append(cls)
        
        method_class_relations = []
        for method in methods:
            method_file = method.get('location', {}).get('file_path')
            method_line = method.get('location', {}).get('line_start', 0)
            
            if method_file in classes_by_file:
                # Find the class that contains this method based on line numbers
                best_match = None
                smallest_range = float('inf')
                
                for cls in classes_by_file[method_file]:
                    cls_start = cls.get('location', {}).get('line_start', 0)
                    cls_end = cls.get('location', {}).get('line_end', 0)
                    
                    if cls_start <= method_line <= cls_end:
                        range_size = cls_end - cls_start
                        if range_size < smallest_range:
                            smallest_range = range_size
                            best_match = cls
                
                if best_match:
                    method_class_relations.append({
                        'method_id': method.get('id'),
                        'class_id': best_match.get('id'),
                        'method_name': method.get('name'),
                        'class_name': best_match.get('name')
                    })
        
        if not method_class_relations:
            logger.info("No class-method relationships to create")
            return
        
        logger.info(f"Creating {len(method_class_relations)} class-method relationships...")
        
        with self.driver.session() as session:
            for i in range(0, len(method_class_relations), self.RELATIONSHIP_BATCH_SIZE):
                batch = method_class_relations[i:i + self.RELATIONSHIP_BATCH_SIZE]
                
                query = """
                UNWIND $relations as rel
                MATCH (method:Function {id: rel.method_id})
                MATCH (class:Class {id: rel.class_id})
                MERGE (method)-[r:BELONGS_TO]->(class)
                SET r.method_name = rel.method_name,
                    r.class_name = rel.class_name,
                    r.created_at = timestamp()
                """
                
                session.run(query, {'relations': batch})
                self.stats.relationships_created += len(batch)
    
    def _create_enhanced_relationships(self):
        """
        Create enhanced relationships for go-to-definition functionality.
        
        This method creates sophisticated relationships that enable accurate
        code navigation and definition resolution.
        """
        logger.info("Creating enhanced relationships for go-to-definition...")
        
        # 1. Create import resolution relationships
        self._create_import_resolution()
        
        # 2. Create namespace-aware function calls  
        self._create_namespace_aware_calls()
        
        logger.info("Enhanced relationships creation completed")
    
    def _create_semantic_relationships(self):
        """
        Create semantic relationships based on purpose, domain, and technology.
        
        This method creates relationships that enable semantic search and
        discovery of related functionality across the codebase.
        """
        logger.info("Creating semantic relationships...")
        
        with self.driver.session() as session:
            # Create SIMILAR_PURPOSE relationships
            query = """
            MATCH (f1:Function), (f2:Function)
            WHERE f1 <> f2 
              AND f1.purpose IS NOT NULL 
              AND f2.purpose IS NOT NULL
              AND f1.purpose = f2.purpose
              AND f1.purpose <> ''
            MERGE (f1)-[r:SIMILAR_PURPOSE]->(f2)
            SET r.purpose = f1.purpose,
                r.created_at = timestamp()
            """
            session.run(query)
            
            # Create SAME_DOMAIN relationships
            query = """
            MATCH (n1), (n2)
            WHERE n1 <> n2 
              AND n1.domain IS NOT NULL 
              AND n2.domain IS NOT NULL
              AND n1.domain = n2.domain
              AND n1.domain <> ''
              AND (n1:Function OR n1:Class)
              AND (n2:Function OR n2:Class)
            MERGE (n1)-[r:SAME_DOMAIN]->(n2)
            SET r.domain = n1.domain,
                r.created_at = timestamp()
            """
            session.run(query)
            
            # Create USES_TECHNOLOGY relationships
            query = """
            MATCH (n), (m:Module)
            WHERE n.technologies IS NOT NULL
              AND size(n.technologies) > 0
              AND ANY(tech IN n.technologies WHERE tech = m.name)
            MERGE (n)-[r:USES_TECHNOLOGY]->(m)
            SET r.created_at = timestamp()
            """
            session.run(query)
        
        logger.info("Semantic relationships creation completed")
    
    def _create_import_resolution(self):
        """
        Link imports to their actual definitions for accurate go-to-definition.
        
        This method creates RESOLVES_TO relationships between import statements
        and the actual function/class definitions they reference.
        """
        logger.info("Creating import resolution relationships...")
        
        with self.driver.session() as session:
            # Resolve 'from module import item' statements
            query = """
            MATCH (imp:Import)
            WHERE imp.import_type = 'from_import' AND imp.module IS NOT NULL
            
            // Extract imported items from the imports array
            UNWIND imp.imports as imported_item
            
            // Method 1: Find exact file path match (local modules)
            OPTIONAL MATCH (def_func:Function)
            WHERE def_func.file_path CONTAINS replace(imp.module, '.', '/')
              AND def_func.name = imported_item
            
            OPTIONAL MATCH (def_class:Class)
            WHERE def_class.file_path CONTAINS replace(imp.module, '.', '/')
              AND def_class.name = imported_item
            
            // Method 2: Find by module name pattern (for packages)
            OPTIONAL MATCH (def_func2:Function)
            WHERE def_func2.file_path ENDS WITH (imp.module + '.py')
              AND def_func2.name = imported_item
            
            OPTIONAL MATCH (def_class2:Class)
            WHERE def_class2.file_path ENDS WITH (imp.module + '.py')
              AND def_class2.name = imported_item
            
            // Create resolution relationships
            FOREACH (f IN CASE WHEN def_func IS NOT NULL THEN [def_func] ELSE [] END |
                MERGE (imp)-[:RESOLVES_TO]->(f)
            )
            FOREACH (c IN CASE WHEN def_class IS NOT NULL THEN [def_class] ELSE [] END |
                MERGE (imp)-[:RESOLVES_TO]->(c)
            )
            FOREACH (f2 IN CASE WHEN def_func2 IS NOT NULL THEN [def_func2] ELSE [] END |
                MERGE (imp)-[:RESOLVES_TO]->(f2)
            )
            FOREACH (c2 IN CASE WHEN def_class2 IS NOT NULL THEN [def_class2] ELSE [] END |
                MERGE (imp)-[:RESOLVES_TO]->(c2)
            )
            """
            
            result = session.run(query)
            logger.info("Import resolution relationships created")
            
            # Also handle direct imports (import module)
            direct_import_query = """
            MATCH (imp:Import)
            WHERE imp.import_type = 'import'
            
            UNWIND imp.imports as module_name
            
            // Find files that match the module
            OPTIONAL MATCH (file:File)
            WHERE file.path CONTAINS replace(module_name, '.', '/')
              AND file.path ENDS WITH '.py'
            
            // Link import to the file's main classes/functions
            OPTIONAL MATCH (file)<-[:DEFINED_IN]-(def:Function)
            WHERE def.name = split(file.name, '.')[0]  // Main function matching filename
            
            OPTIONAL MATCH (file)<-[:DEFINED_IN]-(def_class:Class)
            WHERE def_class.name = split(file.name, '.')[0]  // Main class matching filename
            
            FOREACH (d IN CASE WHEN def IS NOT NULL THEN [def] ELSE [] END |
                MERGE (imp)-[:RESOLVES_TO]->(d)
            )
            FOREACH (dc IN CASE WHEN def_class IS NOT NULL THEN [def_class] ELSE [] END |
                MERGE (imp)-[:RESOLVES_TO]->(dc)
            )
            """
            
            session.run(direct_import_query)
    def _create_qualified_name_relationships(self):
        """
        Create relationships based on qualified names for better go-to-definition accuracy.
        
        This method creates SAME_MODULE relationships between functions and classes
        in the same module, which helps with accurate definition resolution.
        """
        logger.info("Creating qualified name relationships...")
        
        with self.driver.session() as session:
            # Create SAME_MODULE relationships for functions and classes in the same file
            query = """
            MATCH (n1), (n2)
            WHERE n1 <> n2 
              AND n1.file_path = n2.file_path
              AND (n1:Function OR n1:Class)
              AND (n2:Function OR n2:Class)
            MERGE (n1)-[r:SAME_MODULE]->(n2)
            SET r.module_path = n1.file_path,
                r.created_at = timestamp()
            """
            session.run(query)
            
            # Create DEFINES relationships from files to their contained functions/classes
            query = """
            MATCH (file:File), (entity)
            WHERE entity.file_path = file.path
              AND (entity:Function OR entity:Class)
            MERGE (file)-[r:DEFINES]->(entity)
            SET r.created_at = timestamp()
            """
            session.run(query)
        
        logger.info("Qualified name relationships created")
    
    
    def _create_namespace_aware_calls(self):
        """
        Create enhanced function call relationships with proper resolution priority.
        
        This method creates sophisticated call relationships that respect
        Python scoping rules and import resolution.
        """
        logger.info("Creating namespace-aware function calls...")
        
        with self.driver.session() as session:
            query = """
            MATCH (caller:Function)
            WHERE size(caller.calls) > 0
            
            UNWIND caller.calls as called_name
            WITH caller, called_name
            WHERE called_name IS NOT NULL AND called_name <> ''
            
            // Get caller's file for context
            MATCH (caller)-[:DEFINED_IN]->(caller_file:File)
            
            // Priority 1: Exact match in same file
            OPTIONAL MATCH (caller_file)<-[:DEFINED_IN]-(local_func:Function)
            WHERE local_func.name = called_name AND local_func <> caller
            
            // Priority 2: Imported function resolution
            OPTIONAL MATCH (caller_file)-[:IMPORTS]->(imp:Import)-[:RESOLVES_TO]->(imported_func:Function)
            WHERE imported_func.name = called_name
            
            // Priority 3: Method call within same class
            OPTIONAL MATCH (caller)-[:BELONGS_TO]->(caller_class:Class)
            OPTIONAL MATCH (method:Function)-[:BELONGS_TO]->(caller_class)
            WHERE method.name = called_name AND method <> caller
            
            // Priority 4: Global function with same name (fallback)
            OPTIONAL MATCH (global_func:Function)
            WHERE global_func.name = called_name 
              AND global_func <> caller
              AND local_func IS NULL 
              AND imported_func IS NULL 
              AND method IS NULL
            
            // Create relationships with priority metadata
            FOREACH (target IN CASE WHEN local_func IS NOT NULL THEN [local_func] ELSE [] END |
                MERGE (caller)-[r:CALLS_LOCAL]->(target)
                SET r.called_name = called_name,
                    r.resolution_type = 'local',
                    r.priority = 1,
                    r.created_at = timestamp()
            )
            
            FOREACH (target IN CASE WHEN imported_func IS NOT NULL THEN [imported_func] ELSE [] END |
                MERGE (caller)-[r:CALLS_IMPORTED]->(target)
                SET r.called_name = called_name,
                    r.resolution_type = 'imported',
                    r.priority = 2,
                    r.created_at = timestamp()
            )
            
            FOREACH (target IN CASE WHEN method IS NOT NULL THEN [method] ELSE [] END |
                MERGE (caller)-[r:CALLS_METHOD]->(target)
                SET r.called_name = called_name,
                    r.resolution_type = 'method',
                    r.priority = 3,
                    r.created_at = timestamp()
            )
            
            FOREACH (target IN CASE WHEN global_func IS NOT NULL THEN [global_func] ELSE [] END |
                MERGE (caller)-[r:CALLS_GLOBAL]->(target)
                SET r.called_name = called_name,
                    r.resolution_type = 'global',
                    r.priority = 4,
                    r.created_at = timestamp()
            )
            
            RETURN count(*) as enhanced_calls_created
            """
            
            result = session.run(query)
            record = result.single()
            if record:
                calls_created = record['enhanced_calls_created']
                logger.info(f"Created {calls_created} enhanced function call relationships")
                self.stats.relationships_created += calls_created
    
    def _generate_file_github_url(self, repo_url: str, file_path: str, repo_branch: str) -> str:
        """
        Generate GitHub URL for a file.
        
        Args:
            repo_url (str): Repository URL
            file_path (str): Path to the file
            repo_branch (str): Repository branch name
            
        Returns:
            str: GitHub URL for the file
        """
        try:
            if not repo_url or not repo_url.startswith('http'):
                return ""
            
            # Extract owner/repo from URL
            if repo_url.endswith('.git'):
                repo_url = repo_url[:-4]
            
            # Handle both https://github.com/owner/repo and git@github.com:owner/repo formats
            if 'github.com' in repo_url:
                if repo_url.startswith('git@'):
                    # Convert git@github.com:owner/repo to https://github.com/owner/repo
                    repo_url = repo_url.replace('git@github.com:', 'https://github.com/')
                
                return f"{repo_url}/blob/{repo_branch}/{file_path}"
            
            return ""
        except Exception as e:
            logger.debug(f"Error generating file GitHub URL: {e}")
            return ""
    
    def create_sample_queries(self):
        """
        Create example queries for enhanced code intelligence with semantic search.
        
        This method generates comprehensive query examples that demonstrate
        the full capabilities of the code intelligence system.
        """
        logger.info("Creating enhanced sample query functions...")
        
        queries = {
            'find_complex_functions': """
            // Find the most complex functions
            MATCH (f:Function)
            WHERE f.complexity IS NOT NULL
            RETURN f.name, f.complexity, f.file_path, f.github_url
            ORDER BY f.complexity DESC
            LIMIT 10
            """,
            
            'find_function_calls': """
            // Find functions that call a specific function
            MATCH (caller:Function)-[:CALLS_LOCAL|CALLS_IMPORTED|CALLS_METHOD|CALLS_GLOBAL]->(callee:Function {name: $function_name})
            RETURN caller.name, caller.file_path, caller.github_url
            """,
            
            'find_neo4j_insertion_functions': """
            // Find functions that handle Neo4j graph insertion
            MATCH (f:Function)
            WHERE (toLower(f.name) CONTAINS 'insert' OR f.name =~ '.*[Ii]nsert.*')
              AND (f.file_path CONTAINS 'neo4j' 
                   OR ANY(call IN f.calls WHERE call CONTAINS 'session') 
                   OR f.content CONTAINS 'neo4j'
                   OR f.content CONTAINS 'graph')
            RETURN f.name, f.qualified_name, f.file_path, f.github_url, f.docstring, f.complexity
            ORDER BY f.name
            """,
            
            'go_to_function_definition': """
            // Go to definition: Find where a called function is defined
            MATCH (caller_file:File {path: $caller_file_path})
            
            // Priority 1: Local function in same file
            OPTIONAL MATCH (caller_file)<-[:DEFINED_IN]-(local_func:Function {name: $function_name})
            
            // Priority 2: Imported function
            OPTIONAL MATCH (caller_file)-[:IMPORTS]->(imp:Import)-[:RESOLVES_TO]->(imported_func:Function)
            WHERE imported_func.name = $function_name
            
            // Priority 3: Method in class context
            OPTIONAL MATCH (caller_file)<-[:DEFINED_IN]-(caller:Function {name: $caller_function_name})
                          -[:BELONGS_TO]->(class:Class)
                          <-[:BELONGS_TO]-(method:Function {name: $function_name})
            
            // Priority 4: Global search
            OPTIONAL MATCH (global_func:Function {name: $function_name})
            
            WITH 
                local_func, imported_func, method, global_func,
                CASE 
                    WHEN local_func IS NOT NULL THEN 1
                    WHEN imported_func IS NOT NULL THEN 2  
                    WHEN method IS NOT NULL THEN 3
                    ELSE 4
                END as priority
            
            WITH 
                CASE 
                    WHEN local_func IS NOT NULL THEN local_func
                    WHEN imported_func IS NOT NULL THEN imported_func
                    WHEN method IS NOT NULL THEN method 
                    ELSE global_func
                END as definition, priority
            
            WHERE definition IS NOT NULL
            
            RETURN 
                definition.name as function_name,
                definition.qualified_name as qualified_name,
                definition.file_path as file_path,
                definition.line_start as line_number,
                definition.github_url as github_link,
                definition.docstring as documentation,
                definition.purpose as purpose,
                definition.domain as domain,
                CASE priority
                    WHEN 1 THEN 'local'
                    WHEN 2 THEN 'imported'  
                    WHEN 3 THEN 'method'
                    ELSE 'global'
                END as resolution_type
            ORDER BY priority LIMIT 1
            """,
            
            'find_all_usages': """
            // Find all usages: Where is this function/class used?
            MATCH (definition)
            WHERE definition.name = $definition_name
              AND definition.node_type IN ['function', 'class']
            
            // Find enhanced function calls
            OPTIONAL MATCH (caller:Function)-[:CALLS_LOCAL|CALLS_IMPORTED|CALLS_METHOD|CALLS_GLOBAL]->(definition)
            WHERE definition.node_type = 'function'
            
            // Find class inheritance
            OPTIONAL MATCH (child:Class)-[:INHERITS]->(definition)  
            WHERE definition.node_type = 'class'
            
            // Find imports
            OPTIONAL MATCH (imp:Import)-[:RESOLVES_TO]->(definition)
            
            RETURN DISTINCT
                CASE 
                    WHEN caller IS NOT NULL THEN caller.file_path
                    WHEN child IS NOT NULL THEN child.file_path
                    WHEN imp IS NOT NULL THEN imp.file_path
                END as usage_file,
                CASE 
                    WHEN caller IS NOT NULL THEN caller.line_start
                    WHEN child IS NOT NULL THEN child.line_start  
                    WHEN imp IS NOT NULL THEN imp.line_start
                END as usage_line,
                CASE 
                    WHEN caller IS NOT NULL THEN 'function_call'
                    WHEN child IS NOT NULL THEN 'inheritance'
                    WHEN imp IS NOT NULL THEN 'import'
                END as usage_type,
                CASE 
                    WHEN caller IS NOT NULL THEN caller.github_url
                    WHEN child IS NOT NULL THEN child.github_url
                    WHEN imp IS NOT NULL THEN imp.github_url  
                END as github_link
            """,
            
            'find_class_hierarchy': """
            // Find class inheritance hierarchy
            MATCH path = (child:Class)-[:INHERITS*]->(parent:Class)
            WHERE parent.name = $class_name
            RETURN path
            """,
            
            'find_similar_functions': """
            // Find functions similar to a given function (requires embeddings)
            MATCH (target:Function {name: $function_name})
            WHERE target.embedding IS NOT NULL
            
            CALL db.index.vector.queryNodes('code_embeddings', 5, target.embedding)
            YIELD node, score
            WHERE node <> target
            RETURN node.name, node.file_path, node.github_url, node.purpose, score
            """,
            
            'find_functions_by_domain_and_purpose': """
            // Find functions by combining domain and purpose criteria
            MATCH (f:Function)
            WHERE ($domain IS NULL OR f.domain = $domain)
              AND ($purpose IS NULL OR f.purpose = $purpose)
              AND ($technology IS NULL OR $technology IN f.technologies)
            RETURN f.name, f.qualified_name, f.purpose, f.domain, f.technologies, f.file_path, f.github_url
            ORDER BY f.domain, f.purpose, f.name
            """,
            
            'semantic_code_search': """
            // Semantic search for functions based on multiple criteria
            MATCH (f:Function)
            WHERE (f.purpose CONTAINS $search_term 
                   OR f.domain CONTAINS $search_term 
                   OR f.name CONTAINS $search_term
                   OR ANY(tech IN f.technologies WHERE tech CONTAINS $search_term)
                   OR f.docstring CONTAINS $search_term)
            RETURN f.name, f.qualified_name, f.purpose, f.domain, f.technologies, 
                   f.file_path, f.github_url, f.docstring
            ORDER BY f.name
            LIMIT 20
            """,
            
            'code_metrics': """
            // Get repository code metrics with semantic breakdown
            MATCH (f:Function)
            WITH count(f) as total_functions,
                 avg(f.complexity) as avg_complexity,
                 max(f.complexity) as max_complexity,
                 collect(DISTINCT f.purpose) as purposes,
                 collect(DISTINCT f.domain) as domains
            
            MATCH (c:Class)
            WITH total_functions, avg_complexity, max_complexity, purposes, domains, count(c) as total_classes
            
            MATCH (file:File)
            RETURN {
                total_functions: total_functions,
                total_classes: total_classes,
                total_files: count(file),
                avg_complexity: round(avg_complexity, 2),
                max_complexity: max_complexity,
                purposes: purposes,
                domains: domains
            } as metrics
            """
        }
        
        # Save queries to file for reference
        queries_file = "enhanced_sample_queries.cypher"
        with open(queries_file, 'w') as f:
            for name, query in queries.items():
                f.write(f"// {name.replace('_', ' ').title()}\n")
                f.write(query.strip() + "\n\n")
        
        logger.info(f"Enhanced sample queries saved to: {queries_file}")
    
    def go_to_definition(self, function_name: str, caller_file: str, caller_function: str = None):
        """
        Enhanced go-to-definition query helper with accurate resolution.
        
        Args:
            function_name (str): Name of the function to find
            caller_file (str): File where the function is called from
            caller_function (str): Optional name of the calling function
            
        Returns:
            Optional[Dict]: Function definition information or None if not found
        """
        with self.driver.session() as session:
            result = session.run("""
            MATCH (caller_file:File {path: $caller_file_path})
            
            // Priority 1: Local function in same file
            OPTIONAL MATCH (caller_file)<-[:DEFINED_IN]-(local_func:Function {name: $function_name})
            
            // Priority 2: Imported function
            OPTIONAL MATCH (caller_file)-[:IMPORTS]->(imp:Import)-[:RESOLVES_TO]->(imported_func:Function)
            WHERE imported_func.name = $function_name
            
            // Priority 3: Method in class context
            OPTIONAL MATCH (caller_file)<-[:DEFINED_IN]-(caller:Function {name: $caller_function_name})
                          -[:BELONGS_TO]->(class:Class)
                          <-[:BELONGS_TO]-(method:Function {name: $function_name})
            
            // Priority 4: Global search
            OPTIONAL MATCH (global_func:Function {name: $function_name})
            
            WITH 
                local_func, imported_func, method, global_func,
                CASE 
                    WHEN local_func IS NOT NULL THEN 1
                    WHEN imported_func IS NOT NULL THEN 2  
                    WHEN method IS NOT NULL THEN 3
                    ELSE 4
                END as priority
            
            WITH 
                CASE 
                    WHEN local_func IS NOT NULL THEN local_func
                    WHEN imported_func IS NOT NULL THEN imported_func
                    WHEN method IS NOT NULL THEN method 
                    ELSE global_func
                END as definition, priority
            
            WHERE definition IS NOT NULL
            
            RETURN 
                definition.name as function_name,
                definition.qualified_name as qualified_name,
                definition.file_path as file_path,
                definition.line_start as line_number,
                definition.github_url as github_link,
                definition.docstring as documentation,
                definition.purpose as purpose,
                definition.domain as domain,
                definition.technologies as technologies,
                CASE priority
                    WHEN 1 THEN 'local'
                    WHEN 2 THEN 'imported'  
                    WHEN 3 THEN 'method'
                    ELSE 'global'
                END as resolution_type
            ORDER BY priority LIMIT 1
            """, {
                'function_name': function_name,
                'caller_file_path': caller_file,
                'caller_function_name': caller_function
            })
            
            record = result.single()
            if record:
                return {
                    'name': record['function_name'],
                    'qualified_name': record['qualified_name'],
                    'file': record['file_path'],
                    'line': record['line_number'],
                    'github_url': record['github_link'],
                    'docs': record['documentation'],
                    'purpose': record['purpose'],
                    'domain': record['domain'],
                    'technologies': record['technologies'],
                    'type': record['resolution_type']
                }
            return None
    
    def run_sample_query(self, query_name: str, **params):
        """
        Run a sample query with parameters.
        
        Args:
            query_name (str): Name of the query to run
            **params: Parameters for the query
            
        Returns:
            List[Dict]: Query results
        """
        queries = {
            'find_complex_functions': """
            MATCH (f:Function)
            WHERE f.complexity IS NOT NULL
            RETURN f.name, f.complexity, f.file_path, f.github_url
            ORDER BY f.complexity DESC
            LIMIT 10
            """,
            
            'code_metrics': """
            MATCH (f:Function)
            WITH count(f) as total_functions,
                 avg(f.complexity) as avg_complexity,
                 max(f.complexity) as max_complexity,
                 collect(DISTINCT f.purpose) as purposes,
                 collect(DISTINCT f.domain) as domains
            
            MATCH (c:Class)
            WITH total_functions, avg_complexity, max_complexity, purposes, domains, count(c) as total_classes
            
            MATCH (file:File)
            RETURN {
                total_functions: total_functions,
                total_classes: total_classes,
                total_files: count(file),
                avg_complexity: round(avg_complexity, 2),
                max_complexity: max_complexity,
                purposes: purposes,
                domains: domains
            } as metrics
            """,
            
            'enhanced_call_stats': """
            // Check enhanced relationship creation
            MATCH ()-[r:CALLS_LOCAL|CALLS_IMPORTED|CALLS_METHOD|CALLS_GLOBAL]->()
            RETURN 
                type(r) as relationship_type,
                count(r) as count
            ORDER BY count DESC
            """
        }
        
        if query_name not in queries:
            logger.error(f"Query '{query_name}' not found")
            return None
        
        with self.driver.session() as session:
            result = session.run(queries[query_name], params)
            return [record.data() for record in result]
    
    def _print_import_summary(self):
        """Print comprehensive import summary with enhanced statistics."""
        duration = self.stats.end_time - self.stats.start_time
        
        print("\n" + "="*60)
        print("Neo4j Import Complete!")
        print("="*60)
        print(f"Duration: {duration:.2f} seconds")
        print(f"Nodes created: {self.stats.nodes_created:,}")
        print(f"Relationships created: {self.stats.relationships_created:,}")
        if EMBEDDINGS_AVAILABLE:
            print(f"Embeddings created: {self.stats.embeddings_created:,}")
        if self.stats.errors > 0:
            print(f"Errors: {self.stats.errors}")
        print(f"Neo4j Instance: {self.config.instance_name}")
        print(f"Database: {self.config.database}")
        print("="*60)
        
        # Run quick metrics query
        try:
            metrics = self.run_sample_query('code_metrics')
            if metrics:
                print("\nRepository Metrics:")
                print("-" * 30)
                for key, value in metrics[0]['metrics'].items():
                    if isinstance(value, list):
                        print(f"{key.replace('_', ' ').title()}: {len(value)} unique values")
                    else:
                        print(f"{key.replace('_', ' ').title()}: {value:,}")
                        
        except Exception as e:
            logger.debug(f"Could not fetch metrics: {e}")


def main():
    """
    Main entry point with enhanced command-line interface.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Import AST analysis results into Neo4j AuraDB with enhanced code intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import analysis with default settings
  python knowledge_graph_generator.py --input analysis.json
  
  # Import with custom Neo4j config
  python knowledge_graph_generator.py --input analysis.json --uri "neo4j+s://xxx.databases.neo4j.io"
  
  # Clear database before import
  python knowledge_graph_generator.py --input analysis.json --clear-db
  
  # Use OpenAI embeddings (requires OPENAI_API_KEY)
  python knowledge_graph_generator.py --input analysis.json --embeddings openai
  
  # Run with verbose logging
  python knowledge_graph_generator.py --input analysis.json --verbose
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to AST analysis JSON file'
    )
    
    parser.add_argument(
        '--uri',
        type=str,
        default=os.getenv('NEO4J_URI', 'neo4j+s://a3d73c70.databases.neo4j.io'),
        help='Neo4j URI (default: from NEO4J_URI env var)'
    )
    
    parser.add_argument(
        '--username',
        type=str,
        default=os.getenv('NEO4J_USERNAME', 'neo4j'),
        help='Neo4j username (default: from NEO4J_USERNAME env var)'
    )
    
    parser.add_argument(
        '--password',
        type=str,
        default=os.getenv('NEO4J_PASSWORD'),
        help='Neo4j password (default: from NEO4J_PASSWORD env var)'
    )
    
    parser.add_argument(
        '--database',
        type=str,
        default=os.getenv('NEO4J_DATABASE', 'neo4j'),
        help='Neo4j database name (default: from NEO4J_DATABASE env var)'
    )
    
    parser.add_argument(
        '--embeddings',
        type=str,
        choices=['auto', 'openai', 'sentence_transformers', 'none'],
        default='auto',
        help='Embedding model to use (default: auto)'
    )
    
    parser.add_argument(
        '--clear-db',
        action='store_true',
        help='Clear database before import (USE WITH CAUTION)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return 1
    
    # Check for required password
    if not args.password:
        print("Error: Neo4j password is required. Set NEO4J_PASSWORD environment variable or use --password")
        return 1
    
    try:
        # Create configuration
        config = Neo4jConfig(
            uri=args.uri,
            username=args.username,
            password=args.password,
            database=args.database,
            instance_id=os.getenv('AURA_INSTANCEID', ''),
            instance_name=os.getenv('AURA_INSTANCENAME', 'Neo4j')
        )
        
        # Initialize importer
        embedding_model = args.embeddings if args.embeddings != 'none' else None
        importer = Neo4jCodeImporter(config, embedding_model)
        
        try:
            # Clear database if requested
            if args.clear_db:
                confirm = input("Are you sure you want to clear the database? (yes/no): ")
                if confirm.lower() == 'yes':
                    importer.clear_database(confirm=True)
                else:
                    print("Database clear cancelled")
                    return 0
            
            # Create schema
            importer.create_schema()
            
            # Import data
            importer.import_analysis_file(args.input)
            
            # Create sample queries
            importer.create_sample_queries()
            
            print("\nImport completed successfully!")
            print("Enhanced code intelligence features available:")
            print("- Accurate go-to-definition with context awareness")
            print("- Enhanced find-all-references")
            print("- Better GitHub URL generation with line/column info")
            print("- Improved function discovery by name patterns")
            
            if EMBEDDINGS_AVAILABLE:
                print("- Vector similarity search for finding similar code")
            
            # Test enhanced relationships
            try:
                enhanced_stats = importer.run_sample_query('enhanced_call_stats')
                if enhanced_stats:
                    print("\nEnhanced Relationships Created:")
                    for stat in enhanced_stats:
                        rel_type = stat['relationship_type'].replace('CALLS_', '').lower()
                        print(f"  {rel_type} calls: {stat['count']:,}")
                        
                # Test Neo4j function discovery  
                print("\nTesting Neo4j function discovery...")
                with importer.driver.session() as session:
                    result = session.run("""
                    MATCH (f:Function)
                    WHERE toLower(f.name) CONTAINS 'insert' 
                      AND (f.file_path CONTAINS 'neo4j' OR f.content CONTAINS 'neo4j')
                    RETURN count(f) as neo4j_functions
                    """)
                    count = result.single()
                    if count and count['neo4j_functions'] > 0:
                        print(f"  Found {count['neo4j_functions']} potential Neo4j insertion functions")
                        
            except Exception as e:
                logger.debug(f"Could not fetch enhanced stats: {e}")
            
            return 0
            
        finally:
            importer.close()
            
    except KeyboardInterrupt:
        print("\nImport interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Import failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())