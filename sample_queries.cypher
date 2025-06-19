// Find Complex Functions
// Find the most complex functions
            MATCH (f:Function)
            WHERE f.complexity IS NOT NULL
            RETURN f.name, f.complexity, f.file_path, f.github_url
            ORDER BY f.complexity DESC
            LIMIT 10

// Find Function Calls
// Find functions that call a specific function
            MATCH (caller:Function)-[:CALLS]->(callee:Function {name: $function_name})
            RETURN caller.name, caller.file_path, caller.github_url

// Find Class Hierarchy
// Find class inheritance hierarchy
            MATCH path = (child:Class)-[:INHERITS*]->(parent:Class)
            WHERE parent.name = $class_name
            RETURN path

// Find Similar Functions
// Find functions similar to a given function (requires embeddings)
            MATCH (target:Function {name: $function_name})
            WHERE target.embedding IS NOT NULL
            
            CALL db.index.vector.queryNodes('code_embeddings', 5, target.embedding)
            YIELD node, score
            WHERE node <> target
            RETURN node.name, node.file_path, node.github_url, score

// Find File Dependencies
// Find files that import from a specific module
            MATCH (file:File)-[:IMPORTS]->(imp:Import)-[:FROM_MODULE]->(mod:Module {name: $module_name})
            RETURN file.path, imp.content, imp.github_url

// Code Metrics
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

