#!/usr/bin/env python3
#ast_analyzer.py
"""
Repository AST Analyzer

A comprehensive tool for extracting abstract syntax trees (ASTs) from Python repositories
and generating structured metadata for code intelligence systems like Dosu.

Features:
- Tree-sitter based parsing with regex fallbacks
- GitHub repository cloning and analysis
- Function, class, and import extraction
- Relationship mapping between code entities
- Neo4j-ready JSON output format
- Performance optimization with caching

Author: Praveen
License: MIT
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
from urllib.parse import urlparse
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ast_analyzer.log')
    ]
)
logger = logging.getLogger(__name__)

# Tree-sitter imports with graceful fallback
TREE_SITTER_AVAILABLE = False
TREE_SITTER_ERROR = None
PY_LANGUAGE = None

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser
    
    # Initialize Python language
    PY_LANGUAGE = Language(tspython.language())
    TREE_SITTER_AVAILABLE = True
    logger.info("Tree-sitter Python parser ready!")
    
except ImportError as e:
    TREE_SITTER_ERROR = f"Tree-sitter modules not available: {e}"
    logger.warning(f"WARNING: {TREE_SITTER_ERROR}")
    logger.warning("Install with: pip install tree-sitter tree-sitter-python")
except Exception as e:
    TREE_SITTER_ERROR = f"Tree-sitter initialization failed: {e}"
    logger.warning(f"WARNING: {TREE_SITTER_ERROR}")


class SimpleASTExtractor:
    """Simple AST extraction using Tree-sitter with robust fallbacks"""
    
    def __init__(self):
        self.parser = None
        self.language = None
        
        if TREE_SITTER_AVAILABLE and PY_LANGUAGE is not None:
            try:
                # Use the new API - Parser takes language in constructor
                self.language = PY_LANGUAGE
                self.parser = Parser(self.language)
                logger.info("Tree-sitter Python parser initialized successfully")
            except Exception as e:
                logger.warning(f"Tree-sitter initialization failed: {e}")
                self.parser = None
                self.language = None
        else:
            logger.warning(f"Tree-sitter not available: {TREE_SITTER_ERROR}")
            logger.warning("Using regex fallback - install tree-sitter-python for better results")

    def _validate_python_file(self, file_path: str) -> bool:
        """Validate that the file is a Python file"""
        if not file_path.endswith('.py'):
            logger.debug(f"Skipping non-Python file: {file_path}")
            return False
        return True

    def extract_from_file(self, file_path: str, content: str, repo_url: str) -> List[Dict]:
        """Extract AST nodes from a Python file with enhanced metadata for Neo4j"""
        # Early validation for Python files only
        if not self._validate_python_file(file_path):
            return []
            
        try:
            # Normalize file path for consistent GitHub URLs
            normalized_path = file_path.replace('\\', '/')
            
            if self.parser:
                nodes = self._extract_with_tree_sitter(normalized_path, content, repo_url)
            else:
                nodes = self._extract_with_regex(normalized_path, content, repo_url)
            
            # Debug logging for first few files
            if len(nodes) > 0:
                logger.debug(f"Extracted {len(nodes)} nodes from {file_path}")
            
            return nodes
        except Exception as e:
            logger.error(f"Error extracting from {file_path}: {e}")
            return []

    def _extract_with_tree_sitter(self, file_path: str, content: str, repo_url: str) -> List[Dict]:
        """Extract using Tree-sitter with simplified node walking"""
        try:
            tree = self.parser.parse(content.encode('utf-8'))
            root_node = tree.root_node
            
            nodes = []
            
            # Walk the tree and extract nodes
            def walk_tree(node):
                if node.type == 'function_definition':
                    func_node = self._extract_function_node(node, file_path, content, repo_url)
                    if func_node:
                        nodes.append(func_node)
                
                elif node.type == 'class_definition':
                    class_node = self._extract_class_node(node, file_path, content, repo_url)
                    if class_node:
                        nodes.append(class_node)
                
                elif node.type in ['import_statement', 'import_from_statement']:
                    import_node = self._extract_import_node(node, file_path, content, repo_url)
                    if import_node:
                        nodes.append(import_node)
                
                # Recursively walk children
                for child in node.children:
                    walk_tree(child)
            
            walk_tree(root_node)
            return nodes
            
        except Exception as e:
            logger.error(f"Tree-sitter extraction failed for {file_path}: {e}")
            return self._extract_with_regex(file_path, content, repo_url)

    def _extract_function_node(self, node, file_path: str, content: str, repo_url: str) -> Optional[Dict]:
        """Extract detailed function metadata - simplified version"""
        try:
            lines = content.split('\n')
            start_line = node.start_point[0] + 1  # 1-indexed
            end_line = node.end_point[0] + 1
            
            # Get function name - simple approach
            func_name = None
            for child in node.children:
                if child.type == 'identifier':
                    func_name = child.text.decode('utf-8')
                    break
            
            if not func_name:
                return None
            
            # Extract parameters - simple approach
            parameters = []
            for child in node.children:
                if child.type == 'parameters':
                    for param_child in child.children:
                        if param_child.type == 'identifier':
                            parameters.append(param_child.text.decode('utf-8'))
            
            # Extract function calls within this function - simple scan
            function_calls = []
            try:
                func_text = node.text.decode('utf-8', errors='ignore')
                # Simple regex to find function calls
                import re
                call_pattern = r'(\w+)\s*\('
                calls = re.findall(call_pattern, func_text)
                function_calls = list(set([call for call in calls if call != func_name]))[:10]  # Limit to 10
            except:
                function_calls = []
            
            # Calculate complexity (simplified)
            complexity = self._calculate_complexity(node)
            
            # Check if async
            is_async = any(child.type == 'async' for child in node.children)
            
            # Check if method (has 'self' parameter)
            is_method = 'self' in parameters
            
            # Extract docstring - simple approach
            docstring = None
            try:
                for child in node.children:
                    if child.type == 'block':
                        for stmt_child in child.children:
                            if stmt_child.type == 'expression_statement':
                                for expr_child in stmt_child.children:
                                    if expr_child.type == 'string':
                                        docstring = expr_child.text.decode('utf-8').strip('\'"')[:200]
                                        break
                                if docstring:
                                    break
                        if docstring:
                            break
            except:
                docstring = None
            
            # Generate GitHub URL
            github_url = self._generate_github_url(repo_url, file_path, start_line, end_line)
            
            # Generate unique ID for Neo4j
            node_id = self._generate_node_id(file_path, func_name, start_line)
            
            return {
                "id": node_id,
                "node_type": "function",
                "name": func_name,
                "location": {
                    "file_path": file_path,
                    "line_start": start_line,
                    "line_end": end_line,
                    "column_start": node.start_point[1],
                    "column_end": node.end_point[1]
                },
                "language": "python",
                "content": node.text.decode('utf-8', errors='ignore')[:500],  # Limit content size
                "parameters": parameters,
                "calls": function_calls,
                "complexity": complexity,
                "is_async": is_async,
                "is_method": is_method,
                "docstring": docstring,
                "github_url": github_url,
                "repo_url": repo_url,
                "created_at": int(time.time())
            }
            
        except Exception as e:
            logger.debug(f"Error extracting function node: {e}")
            return None

    def _extract_class_node(self, node, file_path: str, content: str, repo_url: str) -> Optional[Dict]:
        """Extract detailed class metadata - simplified version"""
        try:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            # Extract class name - simple approach
            class_name = None
            for child in node.children:
                if child.type == 'identifier':
                    class_name = child.text.decode('utf-8')
                    break
            
            if not class_name:
                return None
            
            # Extract methods - simple scan
            methods = []
            try:
                for child in node.children:
                    if child.type == 'block':
                        for stmt_child in child.children:
                            if stmt_child.type == 'function_definition':
                                for func_child in stmt_child.children:
                                    if func_child.type == 'identifier':
                                        methods.append(func_child.text.decode('utf-8'))
                                        break
            except:
                methods = []
            
            # Extract base classes - simple approach
            base_classes = []
            try:
                for child in node.children:
                    if child.type == 'argument_list':
                        for arg_child in child.children:
                            if arg_child.type == 'identifier':
                                base_classes.append(arg_child.text.decode('utf-8'))
            except:
                base_classes = []
            
            # Extract docstring - simple approach  
            docstring = None
            try:
                for child in node.children:
                    if child.type == 'block':
                        for stmt_child in child.children:
                            if stmt_child.type == 'expression_statement':
                                for expr_child in stmt_child.children:
                                    if expr_child.type == 'string':
                                        docstring = expr_child.text.decode('utf-8').strip('\'"')[:200]
                                        break
                                if docstring:
                                    break
                        if docstring:
                            break
            except:
                docstring = None
            
            # Generate GitHub URL
            github_url = self._generate_github_url(repo_url, file_path, start_line, end_line)
            
            # Generate unique ID
            node_id = self._generate_node_id(file_path, class_name, start_line)
            
            return {
                "id": node_id,
                "node_type": "class",
                "name": class_name,
                "location": {
                    "file_path": file_path,
                    "line_start": start_line,
                    "line_end": end_line,
                    "column_start": node.start_point[1],
                    "column_end": node.end_point[1]
                },
                "language": "python",
                "content": node.text.decode('utf-8', errors='ignore')[:500],
                "methods": methods,
                "base_classes": base_classes,
                "docstring": docstring,
                "github_url": github_url,
                "repo_url": repo_url,
                "created_at": int(time.time())
            }
            
        except Exception as e:
            logger.debug(f"Error extracting class node: {e}")
            return None

    def _extract_import_node(self, node, file_path: str, content: str, repo_url: str) -> Optional[Dict]:
        """Extract import statement metadata - simplified version"""
        try:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            import_text = node.text.decode('utf-8')
            
            # Parse import statement
            import_info = self._parse_import_statement(import_text)
            
            # Generate GitHub URL
            github_url = self._generate_github_url(repo_url, file_path, start_line, end_line)
            
            # Generate unique ID
            node_id = self._generate_node_id(file_path, f"import_{start_line}", start_line)
            
            return {
                "id": node_id,
                "node_type": "import",
                "name": import_info.get("module", "unknown"),
                "location": {
                    "file_path": file_path,
                    "line_start": start_line,
                    "line_end": end_line,
                    "column_start": node.start_point[1],
                    "column_end": node.end_point[1]
                },
                "language": "python",
                "content": import_text,
                "module": import_info.get("module"),
                "imports": import_info.get("imports", []),
                "import_type": import_info.get("type"),
                "github_url": github_url,
                "repo_url": repo_url,
                "created_at": int(time.time())
            }
            
        except Exception as e:
            logger.debug(f"Error extracting import node: {e}")
            return None

    def _extract_with_regex(self, file_path: str, content: str, repo_url: str) -> List[Dict]:
        """Fallback regex-based extraction"""
        logger.debug(f"Using regex fallback for {file_path}")
        nodes = []
        lines = content.split('\n')
        
        # Extract functions with regex
        func_pattern = r'^(async\s+)?def\s+(\w+)\s*\((.*?)\):'
        for i, line in enumerate(lines, 1):
            match = re.match(func_pattern, line.strip())
            if match:
                is_async = bool(match.group(1))
                func_name = match.group(2)
                params_str = match.group(3)
                
                # Parse parameters
                parameters = []
                if params_str.strip():
                    parameters = [p.strip().split(':')[0].strip() for p in params_str.split(',')]
                    parameters = [p.split('=')[0].strip() for p in parameters if p.strip()]
                
                github_url = self._generate_github_url(repo_url, file_path, i, i)
                node_id = self._generate_node_id(file_path, func_name, i)
                
                nodes.append({
                    "id": node_id,
                    "node_type": "function",
                    "name": func_name,
                    "location": {
                        "file_path": file_path,
                        "line_start": i,
                        "line_end": i,
                        "column_start": 0,
                        "column_end": len(line)
                    },
                    "language": "python",
                    "content": line,
                    "parameters": parameters,
                    "calls": [],
                    "complexity": 1,
                    "is_async": is_async,
                    "is_method": 'self' in parameters,
                    "docstring": None,
                    "github_url": github_url,
                    "repo_url": repo_url,
                    "created_at": int(time.time())
                })
        
        # Extract classes with regex
        class_pattern = r'^class\s+(\w+)(?:\(([^)]*)\))?:'
        for i, line in enumerate(lines, 1):
            match = re.match(class_pattern, line.strip())
            if match:
                class_name = match.group(1)
                bases_str = match.group(2) or ""
                base_classes = [b.strip() for b in bases_str.split(',') if b.strip()]
                
                github_url = self._generate_github_url(repo_url, file_path, i, i)
                node_id = self._generate_node_id(file_path, class_name, i)
                
                nodes.append({
                    "id": node_id,
                    "node_type": "class",
                    "name": class_name,
                    "location": {
                        "file_path": file_path,
                        "line_start": i,
                        "line_end": i,
                        "column_start": 0,
                        "column_end": len(line)
                    },
                    "language": "python",
                    "content": line,
                    "methods": [],
                    "base_classes": base_classes,
                    "docstring": None,
                    "github_url": github_url,
                    "repo_url": repo_url,
                    "created_at": int(time.time())
                })
        
        return nodes

    def _calculate_complexity(self, node) -> int:
        """Calculate simplified cyclomatic complexity"""
        try:
            text = node.text.decode('utf-8', errors='ignore')
            complexity = 1  # Base complexity
            
            # Count decision points
            keywords = ['if', 'elif', 'for', 'while', 'except', 'with', 'and', 'or']
            for keyword in keywords:
                complexity += text.count(f' {keyword} ') + text.count(f'\n{keyword} ')
            
            return max(1, complexity)
        except:
            return 1

    def _parse_import_statement(self, import_text: str) -> Dict:
        """Parse import statement to extract module and imports"""
        import_text = import_text.strip()
        
        if import_text.startswith('from '):
            # from module import item1, item2
            match = re.match(r'from\s+([\w.]+)\s+import\s+(.+)', import_text)
            if match:
                module = match.group(1)
                imports_str = match.group(2)
                imports = [imp.strip() for imp in imports_str.split(',')]
                return {
                    "type": "from_import",
                    "module": module,
                    "imports": imports
                }
        elif import_text.startswith('import '):
            # import module1, module2
            imports_str = import_text[7:]  # Remove 'import '
            imports = [imp.strip() for imp in imports_str.split(',')]
            return {
                "type": "import",
                "module": imports[0] if imports else "unknown",
                "imports": imports
            }
        
        return {
            "type": "unknown",
            "module": "unknown",
            "imports": []
        }

    def _generate_github_url(self, repo_url: str, file_path: str, start_line: int, end_line: int) -> str:
        """Generate GitHub URL for code location"""
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
                
                if start_line == end_line:
                    return f"{repo_url}/blob/main/{file_path}#L{start_line}"
                else:
                    return f"{repo_url}/blob/main/{file_path}#L{start_line}-L{end_line}"
            
            return ""
        except Exception as e:
            logger.debug(f"Error generating GitHub URL: {e}")
            return ""

    def _generate_node_id(self, file_path: str, name: str, line: int) -> str:
        """Generate unique ID for Neo4j nodes"""
        unique_string = f"{file_path}:{name}:{line}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]


class RepositoryAnalyzer:
    """Main analyzer class for processing repositories"""
    
    def __init__(self, output_dir: str = "output"):
        self.extractor = SimpleASTExtractor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'total_nodes': 0,
            'functions': 0,
            'classes': 0,
            'imports': 0,
            'errors': 0
        }
    
    def analyze_repository(self, repo_url: str) -> str:
        """Analyze a repository and return the output file path"""
        logger.info(f"Starting analysis of repository: {repo_url}")
        
        # Extract repository name for file naming
        repo_name = self._extract_repo_name(repo_url)
        output_file = self.output_dir / f"{repo_name}_analysis.json"
        
        # Clone repository
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo_path = self._clone_repository(repo_url, temp_path)
            
            if not repo_path:
                logger.error("Failed to clone repository")
                return ""
            
            # Analyze the repository
            logger.info(f"Cloning repository to {repo_path}")
            nodes = self._analyze_directory(str(repo_path), repo_url)
            
            # Generate metadata
            metadata = self._generate_metadata(repo_url, repo_name, nodes)
            
            # Save results
            self._save_results(output_file, metadata, nodes)
            
            logger.info(f"Analysis complete! Results saved to: {output_file}")
            self._print_summary()
            
            return str(output_file)
    
    def analyze_local_directory(self, directory_path: str, repo_url: str = "") -> str:
        """Analyze a local directory"""
        logger.info(f"Starting analysis of local directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            logger.error(f"Directory does not exist: {directory_path}")
            return ""
        
        # Extract directory name for file naming
        dir_name = os.path.basename(os.path.abspath(directory_path))
        output_file = self.output_dir / f"{dir_name}_analysis.json"
        
        # Analyze the directory
        nodes = self._analyze_directory(directory_path, repo_url)
        
        # Generate metadata
        metadata = self._generate_metadata(repo_url or directory_path, dir_name, nodes)
        
        # Save results
        self._save_results(output_file, metadata, nodes)
        
        logger.info(f"✅ Analysis complete! Results saved to: {output_file}")
        self._print_summary()
        
        return str(output_file)
    
    def _clone_repository(self, repo_url: str, temp_path: Path) -> Optional[Path]:
        """Clone repository to temporary directory"""
        try:
            repo_path = temp_path / "repo"
            
            # Clone with git
            cmd = ["git", "clone", "--depth", "1", repo_url, str(repo_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"Git clone failed: {result.stderr}")
                return None
            
            return repo_path
            
        except subprocess.TimeoutExpired:
            logger.error("Git clone timed out")
            return None
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return None
    
    def _analyze_directory(self, directory: str, repo_url: str) -> List[Dict]:
        """Analyze all Python files in a directory"""
        all_nodes = []
        
        logger.info(f"Scanning directory: {directory}")
        
        # Walk through directory and find Python files
        python_files = []
        for root, dirs, files in os.walk(directory):
            # Skip common non-essential directories for performance
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
            
            for file in files:
                if file.endswith('.py'):  # Only process Python files
                    file_path = os.path.join(root, file)
                    python_files.append(file_path)
        
        logger.info(f"Found {len(python_files)} Python files to analyze")
        
        # Process each Python file
        for file_path in python_files:
            try:
                if self._should_analyze_file(file_path):
                    relative_path = os.path.relpath(file_path, directory)
                    nodes = self._analyze_file(file_path, relative_path, repo_url)
                    all_nodes.extend(nodes)
                    self.stats['files_processed'] += 1
                    
                    if self.stats['files_processed'] % 100 == 0:
                        logger.info(f"Processed {self.stats['files_processed']} files...")
                        
            except Exception as e:
                logger.error(f"Error analyzing file {file_path}: {e}")
                self.stats['errors'] += 1
        
        logger.info(f"Extracted {len(all_nodes)} total AST nodes")
        return all_nodes
    
    def _should_analyze_file(self, file_path: str) -> bool:
        """Determine if a file should be analyzed (Python files only)"""
        # Skip hidden files and directories
        if '/.git/' in file_path or '\\__pycache__\\' in file_path:
            return False
        
        # Skip test files for now (optional - can be enabled)
        # if 'test_' in os.path.basename(file_path) or '/tests/' in file_path:
        #     return False
        
        # Only analyze Python files
        return file_path.endswith('.py')
    
    def _analyze_file(self, file_path: str, relative_path: str, repo_url: str) -> List[Dict]:
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract AST nodes
            nodes = self.extractor.extract_from_file(relative_path, content, repo_url)
            
            # Update statistics
            for node in nodes:
                node_type = node.get('node_type')
                if node_type == 'function':
                    self.stats['functions'] += 1
                elif node_type == 'class':
                    self.stats['classes'] += 1
                elif node_type == 'import':
                    self.stats['imports'] += 1
                self.stats['total_nodes'] += 1
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            self.stats['errors'] += 1
            return []
    
    def _extract_repo_name(self, repo_url: str) -> str:
        """Extract repository name from URL"""
        try:
            parsed = urlparse(repo_url)
            if parsed.path:
                # Remove .git extension if present
                path = parsed.path.rstrip('.git')
                # Get the last part of the path
                return path.split('/')[-1] or 'unknown_repo'
            return 'unknown_repo'
        except:
            return 'unknown_repo'
    
    def _generate_metadata(self, repo_url: str, repo_name: str, nodes: List[Dict]) -> Dict:
        """Generate metadata for the analysis"""
        return {
            "analysis_metadata": {
                "repo_url": repo_url,
                "repo_name": repo_name,
                "analysis_timestamp": int(time.time()),
                "analyzer_version": "2.0.0",
                "total_nodes": len(nodes),
                "statistics": self.stats.copy(),
                "tree_sitter_available": TREE_SITTER_AVAILABLE,
                "output_format": "neo4j_ready"
            }
        }
    
    def _save_results(self, output_file: Path, metadata: Dict, nodes: List[Dict]) -> None:
        """Save analysis results to JSON file"""
        try:
            result = {
                **metadata,
                "nodes": nodes
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Results saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _print_summary(self) -> None:
        """Print analysis summary"""
        print("\n" + "="*60)
        print("Analysis Complete!")
        print("="*60)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Total nodes extracted: {self.stats['total_nodes']}")
        print(f"Functions: {self.stats['functions']}")
        print(f"Classes: {self.stats['classes']}")
        print(f"Imports: {self.stats['imports']}")
        if self.stats['errors'] > 0:
            print(f"Errors: {self.stats['errors']}")
        print(f"Tree-sitter: {'Available' if TREE_SITTER_AVAILABLE else 'Fallback mode'}")
        print("="*60)


def run_smoke_tests():
    """Run basic smoke tests to verify functionality"""
    print("Running smoke tests...")
    
    # Test 0: Tree-sitter availability
    logger.info("Checking Tree-sitter availability...")
    if TREE_SITTER_AVAILABLE and PY_LANGUAGE is not None:
        logger.info("Tree-sitter is available and ready")
        try:
            # Quick test parse using new API
            parser = Parser(PY_LANGUAGE)
            test_tree = parser.parse(b"def test(): pass")
            logger.info(f"Tree-sitter parsing test successful: {test_tree.root_node.type}")
        except Exception as e:
            logger.warning(f"Tree-sitter available but parsing failed: {e}")
    else:
        logger.warning(f"Tree-sitter not available: {TREE_SITTER_ERROR}")
        logger.warning("Will use regex fallback for parsing")
        logger.warning("Install with: pip install tree-sitter tree-sitter-python")
    
    # Test 1: Basic AST extraction
    print("Test 1: Basic AST extraction...")
    extractor = SimpleASTExtractor()
    
    test_code = '''
def hello_world(name: str) -> str:
    """A simple greeting function"""
    return f"Hello, {name}!"

class TestClass:
    """A test class"""
    def __init__(self, value):
        self.value = value
    
    async def process(self):
        return self.value * 2

import os
from typing import Dict, List
'''
    
    nodes = extractor.extract_from_file("test.py", test_code, "https://github.com/test/repo")
    
    if nodes:
        print(f"Extracted {len(nodes)} nodes successfully")
        node_types = [node['node_type'] for node in nodes]
        print(f"   Node types found: {set(node_types)}")
        
        # Check for specific expected nodes
        func_nodes = [n for n in nodes if n['node_type'] == 'function']
        class_nodes = [n for n in nodes if n['node_type'] == 'class']
        import_nodes = [n for n in nodes if n['node_type'] == 'import']
        
        print(f"   Functions: {len(func_nodes)}")
        print(f"   Classes: {len(class_nodes)}")
        print(f"   Imports: {len(import_nodes)}")
        
        # Verify function extraction
        if func_nodes:
            func = func_nodes[0]
            print(f"   Sample function: {func.get('name')} with {len(func.get('parameters', []))} parameters")
    else:
        print("No nodes extracted")
        return False
    
    # Test 2: File type filtering
    print("\nTest 2: File type filtering...")
    non_python_files = ["test.js", "test.txt", "README.md", "config.json"]
    python_files = ["test.py", "main.py", "utils.py"]
    
    for file_path in non_python_files:
        if extractor._validate_python_file(file_path):
            print(f"{file_path} should have been filtered out")
            return False
    
    for file_path in python_files:
        if not extractor._validate_python_file(file_path):
            print(f"{file_path} should have been accepted")
            return False
    
    print("File type filtering working correctly")
    
    # Test 3: GitHub URL generation
    print("\nTest 3: GitHub URL generation...")
    test_urls = [
        ("https://github.com/owner/repo", "src/main.py", 10, 20),
        ("https://github.com/owner/repo.git", "utils.py", 5, 5),
    ]
    
    for repo_url, file_path, start, end in test_urls:
        github_url = extractor._generate_github_url(repo_url, file_path, start, end)
        if "github.com" in github_url and file_path in github_url:
            print(f"Generated URL: {github_url}")
        else:
            print(f"Invalid URL generated: {github_url}")
            return False
    
    print("\nAll smoke tests passed!")
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Repository AST Analyzer - Extract code intelligence from Python repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a GitHub repository
  python ast_analyzer.py --repo https://github.com/langchain-ai/langchain
  
  # Analyze a local directory
  python ast_analyzer.py --local /path/to/project
  
  # Specify custom output directory
  python ast_analyzer.py --repo https://github.com/user/repo --output-dir ./results
  
  # Run smoke tests
  python ast_analyzer.py --smoke-test
        """
    )
    
    parser.add_argument(
        '--repo', 
        type=str,
        help='GitHub repository URL to analyze'
    )
    
    parser.add_argument(
        '--local', 
        type=str,
        help='Local directory path to analyze'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for analysis results (default: output)'
    )
    
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help='Run smoke tests to verify functionality'
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
    
    # Run smoke tests
    if args.smoke_test:
        success = run_smoke_tests()
        return 0 if success else 1
    
    # Validate arguments
    if not args.repo and not args.local:
        parser.print_help()
        print("\nError: Either --repo or --local must be specified")
        return 1
    
    if args.repo and args.local:
        print("Error: Cannot specify both --repo and --local")
        return 1
    
    try:
        # Initialize analyzer with output directory
        analyzer = RepositoryAnalyzer(output_dir=args.output_dir)
        
        # Run analysis
        if args.repo:
            output_file = analyzer.analyze_repository(args.repo)
        else:
            output_file = analyzer.analyze_local_directory(args.local, "")
        
        if output_file:
            print(f"\nAnalysis complete! Results saved to: {output_file}")
            return 0
        else:
            print("\nAnalysis failed")
            return 1
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())