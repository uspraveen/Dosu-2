#!/usr/bin/env python3
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
- Query examples and utilities

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
    # Try OpenAI first (most robust)
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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
    """Neo4j connection configuration"""
    uri: str
    username: str
    password: str
    database: str
    instance_id: str
    instance_name: str

@dataclass
class ImportStats:
    """Statistics tracking for import process"""
    nodes_created: int = 0
    relationships_created: int = 0
    embeddings_created: int = 0
    errors: int = 0
    start_time: float = 0
    end_time: float = 0

class EmbeddingGenerator:
    """Handle text embeddings for vector similarity search"""
    
    def __init__(self, model_type: str = "auto"):
        self.model_type = model_type
        self.model = None
        self.embedding_dim = 384  # Default dimension
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model with fallbacks"""
        global EMBEDDINGS_AVAILABLE, EMBEDDING_ERROR
        
        if self.model_type == "openai" or (self.model_type == "auto" and OPENAI_AVAILABLE):
            try:
                # Check for OpenAI API key
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                
                openai.api_key = api_key
                self.model_type = "openai"
                self.embedding_dim = 1536  # OpenAI ada-002 dimension
                EMBEDDINGS_AVAILABLE = True
                logger.info("Initialized OpenAI embeddings (ada-002)")
                return
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {e}")
        
        if self.model_type == "sentence_transformers" or (self.model_type == "auto" and SENTENCE_TRANSFORMERS_AVAILABLE):
            try:
                # Use a good general-purpose model
                model_name = "all-MiniLM-L6-v2"  # Fast and good quality
                self.model = SentenceTransformer(model_name)
                self.model_type = "sentence_transformers"
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                EMBEDDINGS_AVAILABLE = True
                logger.info(f"Initialized Sentence Transformers: {model_name}")
                return
            except Exception as e:
                logger.warning(f"Sentence Transformers initialization failed: {e}")
        
        # No embeddings available
        EMBEDDINGS_AVAILABLE = False
        EMBEDDING_ERROR = "No embedding models available. Install: pip install openai sentence-transformers"
        logger.warning(f"WARNING: {EMBEDDING_ERROR}")
        logger.warning("Vector similarity search will not be available")
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text"""
        if not EMBEDDINGS_AVAILABLE:
            return None
        
        try:
            if self.model_type == "openai":
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=text[:8000]  # OpenAI limit
                )
                return response['data'][0]['embedding']
            
            elif self.model_type == "sentence_transformers":
                embedding = self.model.encode(text)
                return embedding.tolist()
            
        except Exception as e:
            logger.debug(f"Embedding generation failed: {e}")
            return None
    
    def generate_code_embedding(self, node: Dict) -> Optional[List[float]]:
        """Generate specialized embedding for code nodes"""
        if not EMBEDDINGS_AVAILABLE:
            return None
        
        # Create rich text for embedding
        text_parts = []
        
        # Add node name and type
        text_parts.append(f"{node.get('node_type', '')} {node.get('name', '')}")
        
        # Add docstring if available
        if node.get('docstring'):
            text_parts.append(node['docstring'])
        
        # Add parameters for functions
        if node.get('parameters'):
            text_parts.append(f"parameters: {', '.join(node['parameters'])}")
        
        # Add content (limited)
        if node.get('content'):
            content = node['content'][:500]  # Limit content length
            text_parts.append(content)
        
        # Add file path context
        if node.get('location', {}).get('file_path'):
            file_path = node['location']['file_path']
            # Extract meaningful path components
            path_parts = file_path.replace('\\', '/').split('/')
            meaningful_parts = [p for p in path_parts if p and not p.startswith('.')]
            text_parts.append(f"file: {'/'.join(meaningful_parts[-2:])}")
        
        combined_text = " | ".join(text_parts)
        return self.generate_embedding(combined_text)


class Neo4jCodeImporter:
    """Main importer for Neo4j AuraDB with code intelligence features"""
    
    def __init__(self, config: Neo4jConfig, embedding_model: str = "auto"):
        self.config = config
        self.driver: Optional[Driver] = None
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.stats = ImportStats()
        
        # Batch sizes for performance
        self.BATCH_SIZE = 1000
        self.RELATIONSHIP_BATCH_SIZE = 5000
        
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j AuraDB"""
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
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def clear_database(self, confirm: bool = False):
        """Clear all data from database - USE WITH CAUTION"""
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
        """Create optimized schema with indexes and constraints"""
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
                "CREATE INDEX complexity_idx IF NOT EXISTS FOR (f:Function) ON (f.complexity)"
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
        
        logger.info("Schema creation completed")
    
    def import_analysis_file(self, json_file_path: str):
        """Import AST analysis JSON file into Neo4j"""
        logger.info(f"Starting import from: {json_file_path}")
        self.stats.start_time = time.time()
        
        # Load JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get('analysis_metadata', {})
        nodes = data.get('nodes', [])
        
        logger.info(f"Loaded {len(nodes)} nodes from analysis")
        logger.info(f"Repository: {metadata.get('repo_name', 'Unknown')}")
        
        # Import in stages
        self._import_repository_metadata(metadata)
        self._import_files(nodes)
        self._import_nodes_batch(nodes)
        self._create_relationships(nodes)
        
        self.stats.end_time = time.time()
        self._print_import_summary()
    
    def _import_repository_metadata(self, metadata: Dict):
        """Import repository-level metadata"""
        logger.info("Importing repository metadata...")
        
        with self.driver.session() as session:
            query = """
            MERGE (repo:Repository {url: $repo_url})
            SET repo.name = $repo_name,
                repo.analysis_timestamp = $timestamp,
                repo.total_nodes = $total_nodes,
                repo.analyzer_version = $analyzer_version,
                repo.tree_sitter_available = $tree_sitter
            RETURN repo
            """
            
            session.run(query, {
                'repo_url': metadata.get('repo_url', ''),
                'repo_name': metadata.get('repo_name', 'Unknown'),
                'timestamp': metadata.get('analysis_timestamp', 0),
                'total_nodes': metadata.get('total_nodes', 0),
                'analyzer_version': metadata.get('analyzer_version', ''),
                'tree_sitter': metadata.get('tree_sitter_available', False)
            })
        
        logger.info("Repository metadata imported")
    
    def _import_files(self, nodes: List[Dict]):
        """Import file nodes and create file hierarchy"""
        logger.info("Importing file structure...")
        
        # Extract unique files
        files = {}
        for node in nodes:
            file_path = node.get('location', {}).get('file_path')
            if file_path and file_path not in files:
                files[file_path] = {
                    'path': file_path,
                    'name': file_path.split('/')[-1],
                    'extension': file_path.split('.')[-1] if '.' in file_path else '',
                    'directory': '/'.join(file_path.split('/')[:-1]) if '/' in file_path else ''
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
                    f.created_at = timestamp()
                """
                
                session.run(query, {'files': batch})
        
        logger.info(f"Imported {len(files)} files")
    
    def _import_nodes_batch(self, nodes: List[Dict]):
        """Import code nodes in batches with embeddings"""
        logger.info("Importing code nodes...")
        
        # Group nodes by type for efficient processing
        functions = [n for n in nodes if n.get('node_type') == 'function']
        classes = [n for n in nodes if n.get('node_type') == 'class']
        imports = [n for n in nodes if n.get('node_type') == 'import']
        
        # Import each type
        self._import_functions(functions)
        self._import_classes(classes)
        self._import_imports(imports)
    
    def _import_functions(self, functions: List[Dict]):
        """Import function nodes with embeddings"""
        logger.info(f"Importing {len(functions)} functions...")
        
        with self.driver.session() as session:
            for i in range(0, len(functions), self.BATCH_SIZE):
                batch = functions[i:i + self.BATCH_SIZE]
                
                # Generate embeddings for batch
                for func in batch:
                    if EMBEDDINGS_AVAILABLE:
                        embedding = self.embedding_generator.generate_code_embedding(func)
                        if embedding:
                            func['embedding'] = embedding
                            self.stats.embeddings_created += 1
                
                query = """
                UNWIND $functions as func
                MERGE (f:Function {id: func.id})
                SET f.name = func.name,
                    f.file_path = func.location.file_path,
                    f.line_start = func.location.line_start,
                    f.line_end = func.location.line_end,
                    f.column_start = func.location.column_start,
                    f.column_end = func.location.column_end,
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
                
                if (i + self.BATCH_SIZE) % (self.BATCH_SIZE * 5) == 0:
                    logger.info(f"Imported {min(i + self.BATCH_SIZE, len(functions))} functions...")
        
        # Create parameters as separate nodes
        self._import_function_parameters(functions)
    
    def _import_function_parameters(self, functions: List[Dict]):
        """Import function parameters as separate nodes"""
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
        """Import class nodes with embeddings"""
        logger.info(f"Importing {len(classes)} classes...")
        
        with self.driver.session() as session:
            for i in range(0, len(classes), self.BATCH_SIZE):
                batch = classes[i:i + self.BATCH_SIZE]
                
                # Generate embeddings for batch
                for cls in batch:
                    if EMBEDDINGS_AVAILABLE:
                        embedding = self.embedding_generator.generate_code_embedding(cls)
                        if embedding:
                            cls['embedding'] = embedding
                            self.stats.embeddings_created += 1
                
                query = """
                UNWIND $classes as cls
                MERGE (c:Class {id: cls.id})
                SET c.name = cls.name,
                    c.file_path = cls.location.file_path,
                    c.line_start = cls.location.line_start,
                    c.line_end = cls.location.line_end,
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
    
    def _import_imports(self, imports: List[Dict]):
        """Import import nodes and create module relationships"""
        logger.info(f"Importing {len(imports)} imports...")
        
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
        """Create relationships between code elements"""
        logger.info("Creating code relationships...")
        
        # Create function call relationships
        self._create_function_calls(nodes)
        
        # Create class inheritance relationships
        self._create_class_inheritance(nodes)
        
        # Create class-method relationships
        self._create_class_methods(nodes)
    
    def _create_function_calls(self, nodes: List[Dict]):
        """Create CALLS relationships between functions"""
        logger.info("Creating function call relationships...")
        
        function_calls = []
        functions = [n for n in nodes if n.get('node_type') == 'function']
        
        # Build function name to ID mapping
        func_name_to_id = {}
        for func in functions:
            name = func.get('name')
            if name:
                if name not in func_name_to_id:
                    func_name_to_id[name] = []
                func_name_to_id[name].append(func.get('id'))
        
        # Create call relationships
        for func in functions:
            caller_id = func.get('id')
            for called_func in func.get('calls', []):
                if called_func in func_name_to_id:
                    for callee_id in func_name_to_id[called_func]:
                        if caller_id != callee_id:  # No self-calls
                            function_calls.append({
                                'caller_id': caller_id,
                                'callee_id': callee_id,
                                'called_name': called_func
                            })
        
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
                    r.created_at = timestamp()
                """
                
                session.run(query, {'calls': batch})
                self.stats.relationships_created += len(batch)
    
    def _create_class_inheritance(self, nodes: List[Dict]):
        """Create INHERITS relationships between classes"""
        logger.info("Creating class inheritance relationships...")
        
        classes = [n for n in nodes if n.get('node_type') == 'class']
        
        # Build class name to ID mapping
        class_name_to_id = {}
        for cls in classes:
            name = cls.get('name')
            if name:
                if name not in class_name_to_id:
                    class_name_to_id[name] = []
                class_name_to_id[name].append(cls.get('id'))
        
        inheritance = []
        for cls in classes:
            child_id = cls.get('id')
            for base_class in cls.get('base_classes', []):
                if base_class in class_name_to_id:
                    for parent_id in class_name_to_id[base_class]:
                        if child_id != parent_id:
                            inheritance.append({
                                'child_id': child_id,
                                'parent_id': parent_id,
                                'base_name': base_class
                            })
        
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
                    r.created_at = timestamp()
                """
                
                session.run(query, {'inheritance': batch})
                self.stats.relationships_created += len(batch)
    
    def _create_class_methods(self, nodes: List[Dict]):
        """Create BELONGS_TO relationships between methods and classes"""
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
                # Find the class that contains this method
                for cls in classes_by_file[method_file]:
                    cls_start = cls.get('location', {}).get('line_start', 0)
                    cls_end = cls.get('location', {}).get('line_end', 0)
                    
                    if cls_start <= method_line <= cls_end:
                        method_class_relations.append({
                            'method_id': method.get('id'),
                            'class_id': cls.get('id')
                        })
                        break
        
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
                SET r.created_at = timestamp()
                """
                
                session.run(query, {'relations': batch})
                self.stats.relationships_created += len(batch)
    
    def create_sample_queries(self):
        """Create example queries for code intelligence"""
        logger.info("Creating sample query functions...")
        
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
            MATCH (caller:Function)-[:CALLS]->(callee:Function {name: $function_name})
            RETURN caller.name, caller.file_path, caller.github_url
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
            RETURN node.name, node.file_path, node.github_url, score
            """,
            
            'find_file_dependencies': """
            // Find files that import from a specific module
            MATCH (file:File)-[:IMPORTS]->(imp:Import)-[:FROM_MODULE]->(mod:Module {name: $module_name})
            RETURN file.path, imp.content, imp.github_url
            """,
            
            'code_metrics': """
            // Get repository code metrics
            MATCH (f:Function)
            WITH count(f) as total_functions,
                 avg(f.complexity) as avg_complexity,
                 max(f.complexity) as max_complexity
            
            MATCH (c:Class)
            WITH total_functions, avg_complexity, max_complexity, count(c) as total_classes
            
            MATCH (file:File)
            RETURN {
                total_functions: total_functions,
                total_classes: total_classes,
                total_files: count(file),
                avg_complexity: round(avg_complexity, 2),
                max_complexity: max_complexity
            } as metrics
            """
        }
        
        # Save queries to file for reference
        queries_file = "sample_queries.cypher"
        with open(queries_file, 'w') as f:
            for name, query in queries.items():
                f.write(f"// {name.replace('_', ' ').title()}\n")
                f.write(query.strip() + "\n\n")
        
        logger.info(f"Sample queries saved to: {queries_file}")
    
    def run_sample_query(self, query_name: str, **params):
        """Run a sample query with parameters"""
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
                 max(f.complexity) as max_complexity
            
            MATCH (c:Class)
            WITH total_functions, avg_complexity, max_complexity, count(c) as total_classes
            
            MATCH (file:File)
            RETURN {
                total_functions: total_functions,
                total_classes: total_classes,
                total_files: count(file),
                avg_complexity: round(avg_complexity, 2),
                max_complexity: max_complexity
            } as metrics
            """
        }
        
        if query_name not in queries:
            logger.error(f"Query '{query_name}' not found")
            return None
        
        with self.driver.session() as session:
            result = session.run(queries[query_name], params)
            return [record.data() for record in result]
    
    def _print_import_summary(self):
        """Print comprehensive import summary"""
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
                    print(f"{key.replace('_', ' ').title()}: {value:,}")
        except Exception as e:
            logger.debug(f"Could not fetch metrics: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Import AST analysis results into Neo4j AuraDB with code intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import analysis with default settings
  python neo4j_importer.py --input analysis.json
  
  # Import with custom Neo4j config
  python neo4j_importer.py --input analysis.json --uri "neo4j+s://xxx.databases.neo4j.io"
  
  # Clear database before import
  python neo4j_importer.py --input analysis.json --clear-db
  
  # Use OpenAI embeddings (requires OPENAI_API_KEY)
  python neo4j_importer.py --input analysis.json --embeddings openai
  
  # Run with verbose logging
  python neo4j_importer.py --input analysis.json --verbose
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
            print("You can now run queries against your code intelligence graph.")
            
            if EMBEDDINGS_AVAILABLE:
                print("Vector similarity search is available for finding similar code.")
            
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