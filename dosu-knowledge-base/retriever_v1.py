#!/usr/bin/env python3
#retriever.py
"""
GraphRAG LLM-to-Cypher Pipeline

An intelligent code retrieval system using a multi-stage LLM pipeline:
1. Query Understanding → Plans vector searches  
2. Vector Search → Retrieves relevant entity context
3. Cypher Generation → Creates schema-aware queries
4. Execution → Runs validated Cypher
5. Response Formatting → Natural language results

Author: Praveen
License: MIT
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
import dotenv

# Neo4j imports
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, CypherSyntaxError

# OpenAI for LLM pipeline
import openai
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dotenv.load_dotenv()  # Load environment variables from .env file

@dataclass
class SearchContext:
    """Context from vector search to inform Cypher generation"""
    entity_names: List[str]
    entity_types: List[str]
    file_paths: List[str]
    relevant_nodes: List[Dict]
    confidence_scores: List[float]


@dataclass
class CypherQuery:
    """Generated Cypher query with metadata"""
    query: str
    reasoning: str
    expected_columns: List[str]
    is_valid: bool
    validation_error: str = None


class Neo4jLLMCypherRetriever:
    """
    Production-ready LLM-to-Cypher pipeline for code intelligence.
    
    Uses multi-stage LLM processing for flexible, schema-aware code discovery.
    """
    
    def __init__(self, openai_api_key: str = None):
        """Initialize with environment-based configuration"""
        # Neo4j configuration
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
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.openai_client = OpenAI(api_key=api_key)
        self.embedding_model = "text-embedding-3-small"
        self.llm_model = "gpt-4o-mini"
        
        # Get schema once at initialization
        self.schema = self._extract_enhanced_schema()
        
        logger.info("LLM-to-Cypher Pipeline initialized successfully")
    
    def _verify_connection(self):
        """Verify Neo4j connection"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 'Connected' as status")
                logger.info("Neo4j connection verified")
        except Exception as e:
            raise ConnectionError(f"Neo4j connection failed: {e}")
    
    def _extract_enhanced_schema(self) -> Dict:
        """Extract comprehensive schema information for LLM context"""
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
                
                # Get relationship patterns
                patterns = session.run("""
                MATCH (a)-[r]->(b)
                RETURN DISTINCT labels(a)[0] as from_label, 
                       type(r) as relationship, 
                       labels(b)[0] as to_label
                LIMIT 100
                """)
                
                # Get sample data for important nodes
                samples = session.run("""
                MATCH (n)
                WHERE n.node_type IN ['function', 'class']
                RETURN labels(n)[0] as label, 
                       n.name as name,
                       n.file_path as file_path,
                       n.github_url as github_url
                LIMIT 20
                """)
                
                schema = {
                    "nodes": {record["nodeType"]: record["properties"] for record in node_info},
                    "relationships": {record["relType"]: record["properties"] for record in rel_info},
                    "patterns": [dict(record) for record in patterns],
                    "samples": [dict(record) for record in samples]
                }
                
                logger.info(f"Schema extracted: {len(schema['nodes'])} node types, {len(schema['relationships'])} relationship types")
                return schema
                
        except Exception as e:
            logger.error(f"Schema extraction failed: {e}")
            return {"nodes": {}, "relationships": {}, "patterns": [], "samples": []}
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for vector search"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text[:8000]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    def step1_plan_vector_search(self, user_query: str) -> Dict:
        """
        Step 1: LLM analyzes query and plans vector search strategy
        """
        planning_prompt = f"""You are a code intelligence query planner. Analyze the user's question to plan effective vector searches.

User Query: "{user_query}"

Your task: Plan what vector searches to perform to gather relevant context for answering this query.

Consider:
- What types of code entities might be relevant? (functions, classes, etc.)
- What search terms would find relevant code?
- How many searches should be performed?
- What context is needed for accurate answers?

Available Schema:
Node Types: {list(self.schema['nodes'].keys())}
Relationship Types: {list(self.schema['relationships'].keys())}

OUTPUT JSON FORMAT:
{{
    "search_strategy": "brief description of approach",
    "vector_searches": [
        {{
            "search_text": "text to search for",
            "node_types": ["Function", "Class"], 
            "purpose": "why this search is needed",
            "max_results": 5
        }}
    ],
    "expected_info": "what information you expect to find",
    "reasoning": "why this approach will work"
}}

EXAMPLES:
- "where is class ExampleLinksDirective defined?" → search for "ExampleLinksDirective class definition"
- "neo4j insertion functions" → search for "neo4j insert database graph"
- "authentication logic" → search for "auth authentication login security"

Respond with JSON only:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": planning_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            return json.loads(response.choices[0].message.content.strip())
            
        except Exception as e:
            logger.warning(f"Vector search planning failed: {e}")
            # Fallback plan
            return {
                "search_strategy": "Simple keyword search",
                "vector_searches": [{
                    "search_text": user_query,
                    "node_types": ["Function", "Class"],
                    "purpose": "Find relevant code entities",
                    "max_results": 10
                }],
                "expected_info": "Relevant code entities",
                "reasoning": "Fallback to basic search"
            }
    
    def step2_execute_vector_searches(self, search_plan: Dict) -> SearchContext:
        """
        Step 2: Execute planned vector searches to gather context
        """
        all_entities = []
        all_types = []
        all_paths = []
        all_nodes = []
        all_scores = []
        
        for search in search_plan["vector_searches"]:
            try:
                embedding = self._generate_embedding(search["search_text"])
                if not embedding:
                    continue
                
                # Determine vector index based on node types
                if "Class" in search["node_types"] and "Function" not in search["node_types"]:
                    index_name = "class_embeddings"
                else:
                    index_name = "code_embeddings"
                
                vector_query = """
                CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
                YIELD node, score
                WHERE score >= 0.6
                RETURN 
                    node.name as name,
                    node.node_type as node_type,
                    node.file_path as file_path,
                    node.qualified_name as qualified_name,
                    node.id as node_id,
                    score,
                    node {.*} as full_node
                ORDER BY score DESC
                """
                
                with self.driver.session() as session:
                    result = session.run(
                        vector_query,
                        index_name=index_name,
                        limit=search["max_results"],
                        embedding=embedding
                    )
                    
                    for record in result:
                        all_entities.append(record["name"])
                        all_types.append(record["node_type"])
                        all_paths.append(record["file_path"])
                        all_nodes.append(dict(record["full_node"]))
                        all_scores.append(record["score"])
                        
            except Exception as e:
                logger.error(f"Vector search failed for '{search['search_text']}': {e}")
                continue
        
        logger.info(f"Vector search found {len(all_entities)} relevant entities")
        
        return SearchContext(
            entity_names=all_entities,
            entity_types=all_types,
            file_paths=all_paths,
            relevant_nodes=all_nodes,
            confidence_scores=all_scores
        )
    
    def step3_generate_cypher(self, user_query: str, search_context: SearchContext) -> CypherQuery:
        """
        Step 3: LLM generates schema-aware Cypher query based on context
        """
        # Prepare context summary
        context_summary = []
        for i, (name, node_type, path, score) in enumerate(zip(
            search_context.entity_names[:10], 
            search_context.entity_types[:10],
            search_context.file_paths[:10],
            search_context.confidence_scores[:10]
        )):
            context_summary.append(f"- {name} ({node_type}) in {path} (score: {score:.2f})")
        
        context_text = "\n".join(context_summary) if context_summary else "No relevant entities found"
        
        # Prepare schema information
        schema_text = self._format_schema_for_llm()
        
        cypher_prompt = f"""You are an expert Neo4j Cypher query generator. Generate a precise Cypher query to answer the user's question.

User Question: "{user_query}"

Retrieved Context (relevant entities found):
{context_text}

Database Schema:
{schema_text}

CYPHER GENERATION GUIDELINES:
1. Use ONLY the schema elements provided above
2. Focus on the retrieved entities to guide your query
3. Use proper Cypher syntax with correct relationship directions
4. **ALWAYS query entities directly - DO NOT join with File nodes unless specifically needed**
5. **CRITICAL: Get github_url from the actual entity (Class/Function), not from File nodes**

CORRECT PATTERNS (use these):
✅ MATCH (c:Class {{name: "ClassName"}}) RETURN c.name, c.file_path, c.line_start, c.line_end, c.github_url, c.qualified_name
✅ MATCH (f:Function {{name: "FunctionName"}}) RETURN f.name, f.file_path, f.line_start, f.line_end, f.github_url, f.qualified_name
✅ MATCH (n) WHERE n.name CONTAINS "pattern" RETURN n.name, n.file_path, n.line_start, n.line_end, n.github_url

INCORRECT PATTERNS (avoid these):
❌ MATCH (c:Class)-[:DEFINED_IN]->(f:File) RETURN f.github_url  (File URLs are less detailed)
❌ Joining with File nodes when you can get everything from the entity itself

CRITICAL: Entity nodes (Class, Function) have detailed github_url with line/column anchors like:
"https://github.com/repo/blob/branch/file.py#L37C1-L68C27"

File nodes only have basic file-level URLs without line numbers.

REQUIRED RETURN FIELDS for location queries:
- n.name (entity name)
- n.file_path (file path) 
- n.line_start (starting line number)
- n.line_end (ending line number)
- n.github_url (DETAILED URL with line/column anchors from entity, not file)
- n.qualified_name (full qualified name)

EXAMPLES:
Query: "where is class ExampleLinksDirective defined?"
Cypher: MATCH (c:Class {{name: "ExampleLinksDirective"}}) RETURN c.name, c.file_path, c.line_start, c.line_end, c.github_url, c.qualified_name

Query: "find function insertData"
Cypher: MATCH (f:Function {{name: "insertData"}}) RETURN f.name, f.file_path, f.line_start, f.line_end, f.github_url, f.qualified_name

OUTPUT JSON FORMAT:
{{
    "cypher": "MATCH (entity) ... RETURN entity.name, entity.file_path, entity.line_start, entity.line_end, entity.github_url, ...",
    "reasoning": "why this query answers the question",
    "expected_columns": ["name", "file_path", "line_start", "line_end", "github_url"],
    "explanation": "what this query does"
}}

Generate a Cypher query that directly answers: "{user_query}"

Respond with JSON only:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": cypher_prompt}],
                temperature=0.1,
                max_tokens=600
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            
            cypher_query = CypherQuery(
                query=result["cypher"],
                reasoning=result.get("reasoning", ""),
                expected_columns=result.get("expected_columns", []),
                is_valid=True
            )
            
            # Validate syntax
            cypher_query = self._validate_cypher(cypher_query)
            
            return cypher_query
            
        except Exception as e:
            logger.error(f"Cypher generation failed: {e}")
            return CypherQuery(
                query="",
                reasoning="Failed to generate query",
                expected_columns=[],
                is_valid=False,
                validation_error=str(e)
            )
    
    def _format_schema_for_llm(self) -> str:
        """Format schema information for LLM context with emphasis on entity-level URLs"""
        schema_parts = []
        
        # Node types and properties with emphasis on entity-level GitHub URLs
        schema_parts.append("NODE TYPES AND KEY PROPERTIES:")
        for node_type, properties in self.schema["nodes"].items():
            # Prioritize location-related properties
            priority_props = []
            other_props = []
            
            for prop in properties:
                prop_name = prop["property"]
                if prop_name in ["name", "file_path", "line_start", "line_end", "github_url", "qualified_name"]:
                    priority_props.append(prop_name)
                else:
                    other_props.append(prop_name)
            
            # Show priority props first, then others (limited)
            all_props = priority_props + other_props[:3]
            schema_parts.append(f"- {node_type}: {', '.join(all_props)}")
        
        # Emphasize critical differences between entity and file URLs
        schema_parts.append("\n🎯 CRITICAL URL DIFFERENCES:")
        schema_parts.append("✅ ENTITY URLs (Class/Function nodes): https://github.com/repo/blob/branch/file.py#L37C1-L68C27")
        schema_parts.append("   ↳ Include precise line/column anchors (L37C1-L68C27)")
        schema_parts.append("❌ FILE URLs (File nodes): https://github.com/repo/blob/branch/file.py")
        schema_parts.append("   ↳ Only point to file, no line numbers")
        schema_parts.append("")
        schema_parts.append("💡 ALWAYS USE ENTITY URLs for precise code locations!")
        
        # Critical properties with emphasis on entity-level data
        schema_parts.append("\nCRITICAL LOCATION PROPERTIES (from entity nodes):")
        schema_parts.append("- name: Entity name")
        schema_parts.append("- file_path: Full file path") 
        schema_parts.append("- line_start: Starting line number")
        schema_parts.append("- line_end: Ending line number")
        schema_parts.append("- github_url: 🎯 DETAILED GitHub URL with line/column anchors (L#C#-L#C#)")
        schema_parts.append("- qualified_name: Full qualified name")
        
        # Relationship types
        schema_parts.append("\nRELATIONSHIP TYPES:")
        for rel_type in self.schema["relationships"].keys():
            schema_parts.append(f"- {rel_type}")
        
        # Common patterns with emphasis on direct entity queries
        schema_parts.append("\n🎯 PREFERRED QUERY PATTERNS:")
        schema_parts.append("✅ MATCH (c:Class {name: 'ClassName'}) RETURN c.name, c.github_url")
        schema_parts.append("✅ MATCH (f:Function {name: 'FuncName'}) RETURN f.name, f.github_url") 
        schema_parts.append("❌ MATCH (c:Class)-[:DEFINED_IN]->(file:File) RETURN file.github_url")
        
        schema_parts.append("\nRELATIONSHIP PATTERNS:")
        for pattern in self.schema["patterns"][:8]:
            schema_parts.append(f"- ({pattern['from_label']})-[:{pattern['relationship']}]->({pattern['to_label']})")
        
        # Sample entities with their actual GitHub URLs
        if self.schema["samples"]:
            schema_parts.append("\nSAMPLE ENTITIES WITH URLS:")
            for sample in self.schema["samples"][:3]:
                github_url = sample.get('github_url', 'N/A')
                # Truncate URL for display but show the pattern
                if github_url and len(github_url) > 80:
                    display_url = github_url[:80] + "..."
                else:
                    display_url = github_url
                schema_parts.append(f"- {sample['name']} ({sample['label']}) → {display_url}")
        
        return "\n".join(schema_parts)
    
    def _validate_cypher(self, cypher_query: CypherQuery) -> CypherQuery:
        """Validate Cypher syntax without executing"""
        try:
            with self.driver.session() as session:
                # Use EXPLAIN to validate syntax without execution
                session.run(f"EXPLAIN {cypher_query.query}")
                cypher_query.is_valid = True
                return cypher_query
                
        except CypherSyntaxError as e:
            logger.warning(f"Cypher syntax error: {e}")
            cypher_query.is_valid = False
            cypher_query.validation_error = str(e)
            return cypher_query
        except Exception as e:
            logger.warning(f"Cypher validation failed: {e}")
            cypher_query.is_valid = False
            cypher_query.validation_error = str(e)
            return cypher_query
    
    def step4_execute_cypher(self, cypher_query: CypherQuery) -> List[Dict]:
        """
        Step 4: Execute validated Cypher query
        """
        if not cypher_query.is_valid:
            logger.error(f"Cannot execute invalid Cypher: {cypher_query.validation_error}")
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query.query)
                records = [dict(record) for record in result]
                logger.info(f"Cypher execution returned {len(records)} records")
                return records
                
        except Exception as e:
            logger.error(f"Cypher execution failed: {e}")
            return []
    
    def step5_format_response(self, user_query: str, cypher_query: CypherQuery, 
                             query_results: List[Dict], search_context: SearchContext) -> str:
        """
        Step 5: LLM formats final response with detailed GitHub URLs
        """
        if not query_results:
            return self._format_no_results_response(user_query, cypher_query, search_context)
        
        # Prepare results summary with full URL preservation
        results_text = []
        for i, record in enumerate(query_results[:5], 1):
            result_parts = []
            for key, value in record.items():
                if value is not None:
                    # Preserve full github_url without truncation
                    if key == 'github_url':
                        result_parts.append(f"{key}: {value}")
                    else:
                        result_parts.append(f"{key}: {value}")
            results_text.append(f"{i}. {' | '.join(result_parts)}")
        
        results_summary = "\n".join(results_text)
        
        response_prompt = f"""You are a helpful code assistant. Format a clear, helpful response with EXACT line numbers and detailed GitHub links.

User Question: "{user_query}"

Generated Cypher Query:
```cypher
{cypher_query.query}
```

Query Results:
{results_summary}

CRITICAL REQUIREMENTS:
1. **Always include exact line numbers** when available (line_start, line_end)
2. **Always include FULL GitHub URLs** with line/column anchors when available  
3. **Format GitHub links** as clickable markdown links: [View on GitHub](full_url)
4. **Preserve complete URLs** - do not truncate or modify github_url values
5. **Show precise location format**: file.py:L37-L68

RESPONSE FORMAT:
- **Entity Name**: [exact name from results]
- **Location**: `file_path:L{{line_start}}-L{{line_end}}`
- **GitHub**: [View on GitHub](complete_github_url_with_anchors)
- **Qualified Name**: [qualified_name if available]

EXAMPLE RESPONSE FORMAT:
The class `ExampleLinksDirective` is defined at:
- **Location**: `docs/api_reference/conf.py:L37-L68`  
- **GitHub**: [View on GitHub](https://github.com/langchain-ai/langchain/blob/master/docs/api_reference/conf.py#L37C1-L68C27)
- **Qualified Name**: `docs.api_reference.conf.ExampleLinksDirective`

CRITICAL: 
- Use the COMPLETE github_url from results without any truncation
- If github_url contains line anchors like #L37C1-L68C27, preserve them exactly
- Format as clickable markdown links
- Be precise with line number citations
- Extract line_start and line_end values from the query results and use them in the location format

Format the response with complete GitHub URLs:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": response_prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            return self._format_fallback_response(user_query, query_results)
    
    def _format_no_results_response(self, user_query: str, cypher_query: CypherQuery, 
                                   search_context: SearchContext) -> str:
        """Format response when no results found"""
        if not cypher_query.is_valid:
            return f"""I couldn't generate a valid query for: "{user_query}"

**Error**: {cypher_query.validation_error}

**Suggestion**: Try rephrasing your question or ask about specific function/class names that exist in your codebase."""
        
        context_info = ""
        if search_context.entity_names:
            context_info = f"\n**Related entities found**: {', '.join(search_context.entity_names[:5])}"
        
        return f"""I couldn't find specific results for: "{user_query}"

**Generated Query**:
```cypher
{cypher_query.query}
```
{context_info}

**Suggestions**:
- Check if the entity names exist in your codebase
- Try using different search terms
- Ask about broader concepts if your query was very specific"""
    
    def _format_fallback_response(self, user_query: str, query_results: List[Dict]) -> str:
        """Fallback response formatting when LLM fails"""
        response = f"Found {len(query_results)} results for: '{user_query}'\n\n"
        
        for i, record in enumerate(query_results[:3], 1):
            response += f"**{i}.** "
            parts = []
            for key, value in record.items():
                if value is not None:
                    # Preserve full URLs in fallback too
                    parts.append(f"{key}: {value}")
            response += " | ".join(parts) + "\n"
        
        return response
    
    def search(self, user_query: str) -> Tuple[str, Dict]:
        """
        Main search interface using LLM-to-Cypher pipeline
        
        Returns:
            Tuple[str, Dict]: (response, debug_info)
        """
        start_time = time.time()
        debug_info = {}
        
        try:
            # Step 1: Plan vector searches
            logger.info("Step 1: Planning vector searches...")
            search_plan = self.step1_plan_vector_search(user_query)
            debug_info["search_plan"] = search_plan
            
            # Step 2: Execute vector searches
            logger.info("Step 2: Executing vector searches...")
            search_context = self.step2_execute_vector_searches(search_plan)
            debug_info["search_context"] = {
                "entities_found": len(search_context.entity_names),
                "entity_names": search_context.entity_names[:5]
            }
            
            # Step 3: Generate Cypher
            logger.info("Step 3: Generating Cypher query...")
            cypher_query = self.step3_generate_cypher(user_query, search_context)
            debug_info["cypher_query"] = {
                "query": cypher_query.query,
                "is_valid": cypher_query.is_valid,
                "reasoning": cypher_query.reasoning
            }
            
            # Step 4: Execute Cypher
            logger.info("Step 4: Executing Cypher query...")
            query_results = self.step4_execute_cypher(cypher_query)
            debug_info["results_count"] = len(query_results)
            
            # Step 5: Format response
            logger.info("Step 5: Formatting response...")
            response = self.step5_format_response(user_query, cypher_query, query_results, search_context)
            
            elapsed_time = time.time() - start_time
            debug_info["execution_time"] = f"{elapsed_time:.2f}s"
            
            logger.info(f"Pipeline completed in {elapsed_time:.2f}s")
            return response, debug_info
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return f"An error occurred: {str(e)}", {"error": str(e)}
    
    def close(self):
        """Close database connections"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")


# Usage example
def main():
    """Example usage of the LLM-to-Cypher Pipeline"""
    try:
        retriever = Neo4jLLMCypherRetriever()
        
        test_queries = [
            "where is the class ExampleLinksDirective defined?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)
            
            response, debug = retriever.search(query)
            
            print(f"\n**Response:**\n{response}")
            
            if debug.get("cypher_query"):
                print(f"\n**Generated Cypher:**\n```cypher\n{debug['cypher_query']['query']}\n```")
                print(f"\n**Execution Time:** {debug.get('execution_time', 'N/A')}")
                print(f"**Results Found:** {debug.get('results_count', 0)}")
        
        retriever.close()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure environment variables are set: NEO4J_URI, NEO4J_PASSWORD, OPENAI_API_KEY")


if __name__ == "__main__":
    main()